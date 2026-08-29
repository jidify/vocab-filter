"""Évaluation réelle, opt-in, des juges S3 local/frontière sur Q0-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, llm_client, mwe_judge
from pipeline.llm_tasks import task_config

CASES_PATH = Path("fix_pipeline/s3_judge_eval_cases.json")
GOLD_MWE_PATH = Path("fix_pipeline/gold_corpus/the_humans_gold_v0.jsonl")
RESULTS_PATH = config.OUT_DIR / "s3_judge_model_results.json"
REPORT_PATH = config.OUT_DIR / "s3_judge_model_report.md"
# Résolus via le registre (Lot U6 du plan d'unification, ferme M7 —
# fix_pipeline/multi_models/report_multi_models.md §4bis/§5) : un override
# VOCAB_LLM_S3_JUDGE_OCCURRENCE / VOCAB_LLM_S6_TRANSLATE_FRONTIER doit
# atteindre cette évaluation comme la production, pas seulement un
# config.llm_model()/config.SENSE_FR_FRONTIER_MODEL figés à l'import.
# Indépendance juge/candidat de l'ablation (config.py:333-336) non concernée
# ici : "frontière" et "local" sont déjà deux tâches distinctes du registre.
LOCAL_TASK = task_config("S3-judge-occurrence")
FRONTIER_TASK = task_config("S6-translate-frontier")
LOCAL_MODEL = LOCAL_TASK.model
FRONTIER_MODEL = FRONTIER_TASK.model
MIN_PRECISION = 0.97
MIN_RECALL = 0.97
MAX_REVIEW_RATE = 0.05
MIN_LABEL_ACCURACY = 0.97


def _gold_cases(limit: int) -> list[dict]:
    """Extrait uniquement les annotations gold qui portent un label S3."""
    category_to_label = {
        "idiom": "idiome",
        "phrasal_verb_separable": "phrasal_verb",
        "phrasal_verb_inseparable": "phrasal_verb",
    }
    cases = []
    with GOLD_MWE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for index, span in enumerate(row.get("gold_spans", [])):
                label = category_to_label.get(span.get("category"))
                if not span.get("is_gold") or label is None:
                    continue
                cases.append({
                    "id": f'gold-{row["segment_idx"]}-{index}',
                    "stratum": f'gold_{span["category"]}',
                    "canonical_form": span["surface"],
                    "surface": span["surface"],
                    "context": row["text"],
                    "expected_label": label,
                    "gold_source": str(GOLD_MWE_PATH),
                })
                if len(cases) >= limit:
                    return cases
    return cases


def _prompt(case: dict) -> str:
    return f'''Expression candidate : "{case["canonical_form"]}"
Phrase : "{case["context"]}"
Span détecté : "{case["surface"]}"

Classe CETTE occurrence dans exactement une catégorie : "idiome",
"phrasal_verb", "semi_fige", "littéral" ou "incertain".

Réponds uniquement avec ce JSON compact, sans explication ni champ supplémentaire :
{{"label":"<catégorie>","canonical_form":"<canon>","pos":"<NOUN|VERB|ADJ|ADV|OTHER>","confidence":<0.0-1.0>}}'''


def _frontier_call(case: dict) -> tuple[dict, float]:
    """Client unifié (Lot U6) — cache dédié à cette évaluation (préfixe
    ``s3_frontier_``), invalidé par ce changement de mécanique HTTP : cache
    d'outil de diagnostic opt-in, pas l'un des 4 caches de production
    préservés octet pour octet au Lot U3 (voir report_multi_models.md §4bis)."""
    prompt = _prompt(case)
    metadata = {"protocol": mwe_judge.S3_PROMPT_VERSION,
                "schema": mwe_judge.S3_DECISION_SCHEMA_VERSION,
                "model": FRONTIER_MODEL, "canonical_form": case["canonical_form"],
                "context_signature": hashlib.sha256(case["context"].encode()).hexdigest()}
    raw, cost = llm_client.call(
        model=FRONTIER_MODEL, system=mwe_judge.OCC_SYSTEM_PROMPT, prompt=prompt,
        cache_key_fields=llm_client.build_cache_key(
            model=FRONTIER_MODEL, system=mwe_judge.OCC_SYSTEM_PROMPT, prompt=prompt, extra=metadata,
        ),
        cache_prefix="s3_frontier_", reasoning_effort="low", return_cost=True,
    )
    return raw, cost


def _normalize(raw: dict, canonical: str) -> dict:
    label = raw.get("label")
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    valid = label in mwe_judge.VALID_LABELS
    return {"label": label if valid else "incertain", "canonical_form": raw.get("canonical_form") or canonical,
            "confidence": max(0.0, min(1.0, confidence)),
            "schema_valid": valid and bool(raw.get("canonical_form")) and bool(raw.get("pos"))}


def _run_case(case: dict, model_kind: str) -> tuple[dict, float]:
    started = time.perf_counter()
    if model_kind == "local":
        prompt = _prompt(case)
        try:
            raw = llm_client.call(
                model=LOCAL_MODEL, system=mwe_judge.OCC_SYSTEM_PROMPT, prompt=prompt,
                timeout=config.CATGPT_TIMEOUT,
                cache_key_fields=llm_client.build_cache_key(
                    model=LOCAL_MODEL, system=mwe_judge.OCC_SYSTEM_PROMPT, prompt=prompt,
                    extra={"protocol": "s3-judge-eval-1-compact",
                          "schema": "label-canon-pos-confidence-v1"},
                ),
            )
            normalized = _normalize(raw, case["canonical_form"])
        except llm_client.LLMError:
            normalized = {"label": "incertain", "canonical_form": case["canonical_form"],
                          "confidence": 0.0, "schema_valid": False}
        cost = 0.0
    else:
        raw, cost = _frontier_call(case)
        normalized = _normalize(raw, case["canonical_form"])
    normalized["latency_seconds"] = round(time.perf_counter() - started, 4)
    return normalized, cost


def _run_local_batch(cases: list[dict], cache_variant: str = "default") -> tuple[dict[str, dict], float]:
    """Un seul appel CatGPT/Ollama pour tous les cas, réponse compacte indexée."""
    if len(cases) > config.S3_JUDGE_BATCH_SIZE:
        raise ValueError(
            f"lot S3 de {len(cases)} cas > limite validée {config.S3_JUDGE_BATCH_SIZE}"
        )
    items = "\n".join(
        f'{i + 1}. case_id={case["id"]!r}; expression={case["canonical_form"]!r}; '
        f'span={case["surface"]!r}; contexte={case["context"]!r}'
        for i, case in enumerate(cases)
    )
    prompt = f'''Classe séparément les {len(cases)} occurrences suivantes.
Applique strictement ces distinctions :
- "idiome" : sens conventionnel non compositionnel, notamment si la lecture
  littérale contredit le sens réellement communiqué ou inverse sa polarité ;
- "phrasal_verb" : verbe et particule/préposition forment une unité verbale
  lexicalisée dont le sens ou la construction est spécialisé ;
- "semi_fige" : collocation contrainte ou formulation conventionnelle, mais
  dont le sens global reste compositionnel et compatible avec les mots ;
- "littéral" : combinaison syntaxique libre et compositionnelle dans ce contexte ;
- "incertain" : le contexte ne permet pas de trancher sans forcer l'analyse.
En cas de conflit, la non-compositionnalité prime sur le caractère seulement figé.

{items}

Réponds uniquement avec un objet JSON compact, sans explication ni champ supplémentaire :
{{"decisions":[{{"case_id":"<id exact>","label":"<catégorie>","canonical_form":"<canon>","pos":"<NOUN|VERB|ADJ|ADV|OTHER>","confidence":<0.0-1.0>}}]}}
Il doit y avoir exactement une décision par case_id, dans le même ordre.'''
    started = time.perf_counter()
    raw = llm_client.call(
        model=LOCAL_MODEL, system=mwe_judge.OCC_SYSTEM_PROMPT, prompt=prompt,
        timeout=config.CATGPT_TIMEOUT,
        cache_key_fields=llm_client.build_cache_key(
            model=LOCAL_MODEL, system=mwe_judge.OCC_SYSTEM_PROMPT, prompt=prompt,
            extra={"protocol": "s3-judge-eval-1-compact-batch-prompt-2",
                  "schema": "batch-label-canon-pos-confidence-v1",
                  "cache_variant": cache_variant,
                  "case_ids": [case["id"] for case in cases]},
        ),
    )
    elapsed = round(time.perf_counter() - started, 4)
    expected_ids = {case["id"] for case in cases}
    received: dict[str, dict] = {}
    duplicates = set()
    for item in raw.get("decisions", []) if isinstance(raw, dict) else []:
        case_id = item.get("case_id") if isinstance(item, dict) else None
        if case_id in received:
            duplicates.add(case_id)
        elif case_id in expected_ids:
            received[case_id] = item
    predictions = {}
    for case in cases:
        item = received.get(case["id"])
        if item is None or case["id"] in duplicates:
            predictions[case["id"]] = {
                "label": "incertain", "canonical_form": case["canonical_form"],
                "confidence": 0.0, "schema_valid": False,
                "batch_error": "missing_or_duplicate",
            }
        else:
            predictions[case["id"]] = _normalize(item, case["canonical_form"])
        predictions[case["id"]]["latency_seconds"] = elapsed / len(cases)
        predictions[case["id"]]["batch_total_latency_seconds"] = elapsed
    return predictions, 0.0


def _metrics(rows: list[dict]) -> dict:
    tp = sum(r["expected_lexicalized"] and r["predicted_lexicalized"] for r in rows)
    fp = sum(not r["expected_lexicalized"] and r["predicted_lexicalized"] for r in rows)
    fn = sum(r["expected_lexicalized"] and not r["predicted_lexicalized"] for r in rows)
    abstentions = sum(r["prediction"]["label"] == "incertain" for r in rows)
    avoidable_review = sum(r["prediction"]["label"] == "incertain"
                           and r["expected_label"] != "incertain" for r in rows)
    return {"cases": len(rows), "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "label_accuracy": sum(r["prediction"]["label"] == r["expected_label"] for r in rows) / len(rows),
            "schema_valid_rate": sum(r["prediction"]["schema_valid"] for r in rows) / len(rows),
            "review_rate": avoidable_review / len(rows),
            "abstention_rate": abstentions / len(rows),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "latency_seconds": round(sum(r["prediction"]["latency_seconds"] for r in rows), 4)}


def _passes(metrics: dict) -> bool:
    return bool(metrics["precision"] is not None and metrics["precision"] >= MIN_PRECISION
                and metrics["recall"] >= MIN_RECALL and metrics["schema_valid_rate"] == 1.0
                and metrics["label_accuracy"] >= MIN_LABEL_ACCURACY
                and metrics["review_rate"] <= MAX_REVIEW_RATE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=("local", "frontier", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None,
                        help="Nombre maximal de cas, utile pour un diagnostic sans lancer le corpus entier.")
    parser.add_argument("--batch", action="store_true",
                        help="Envoie tous les cas dans un unique appel au backend local configuré.")
    parser.add_argument("--cache-variant", default="default",
                        help="Identifie une configuration externe du modèle et force un nouveau cache.")
    parser.add_argument("--gold-extra", type=int, default=0,
                        help="Ajoute N occurrences MWE distinctes du gold corpus au lot d'évaluation.")
    parser.add_argument("--gold-only", action="store_true",
                        help="Évalue seulement les cas demandés par --gold-extra.")
    args = parser.parse_args()
    cases = [] if args.gold_only else json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if args.gold_extra:
        cases.extend(_gold_cases(args.gold_extra))
    if args.limit is not None:
        cases = cases[:max(0, args.limit)]
    kinds = ["local", "frontier"] if args.models == "both" else [args.models]
    results = {}
    for kind in kinds:
        rows = []
        batch_predictions = None
        if args.batch:
            if kind != "local":
                raise SystemExit("--batch ne s'applique qu'au backend local/CatGPT configuré")
            batch_predictions, _batch_cost = _run_local_batch(cases, args.cache_variant)
        for case in cases:
            if batch_predictions is None:
                prediction, cost = _run_case(case, kind)
            else:
                prediction, cost = batch_predictions[case["id"]], 0.0
            expected_lexicalized = case["expected_label"] in mwe_judge.LEXICALIZED_LABELS
            rows.append({"case_id": case["id"], "stratum": case["stratum"],
                         "expected_label": case["expected_label"], "expected_lexicalized": expected_lexicalized,
                         "predicted_lexicalized": prediction["label"] in mwe_judge.LEXICALIZED_LABELS,
                         "prediction": prediction, "cost_usd": cost})
        strata = defaultdict(list)
        for row in rows:
            strata[row["stratum"]].append(row)
        metrics = _metrics(rows)
        results[kind] = {"model": LOCAL_MODEL if kind == "local" else FRONTIER_MODEL,
                         "backend": LOCAL_TASK.provider if kind == "local" else FRONTIER_TASK.provider,
                         "metrics": metrics, "passes": _passes(metrics), "by_stratum": {
                             name: _metrics(items) for name, items in sorted(strata.items())}, "rows": rows}
    if set(results) == {"local", "frontier"}:
        local_by_stratum = results["local"]["by_stratum"]
        frontier_by_stratum = results["frontier"]["by_stratum"]
        escalated_strata = sorted(
            name for name in local_by_stratum
            if frontier_by_stratum[name]["label_accuracy"] > local_by_stratum[name]["label_accuracy"]
        )
        frontier_rows = {r["case_id"]: r for r in results["frontier"]["rows"]}
        hybrid_rows = [frontier_rows[r["case_id"]] if r["stratum"] in escalated_strata else r
                       for r in results["local"]["rows"]]
        strata = defaultdict(list)
        for row in hybrid_rows:
            strata[row["stratum"]].append(row)
        metrics = _metrics(hybrid_rows)
        results["hybrid"] = {"model": f"{LOCAL_MODEL} + {FRONTIER_MODEL}",
                             "metrics": metrics, "passes": _passes(metrics),
                             "escalated_strata": escalated_strata,
                             "by_stratum": {name: _metrics(items) for name, items in sorted(strata.items())},
                             "rows": hybrid_rows}
    full_evaluation = args.limit is None
    passing = [(name, data) for name, data in results.items()
               if data["passes"] and full_evaluation]
    selected = min(passing, key=lambda item: item[1]["metrics"]["cost_usd"])[0] if passing else None
    payload = {"schema_version": 1, "protocol": mwe_judge.S3_PROMPT_VERSION,
               "evaluation_scope": "full" if full_evaluation else f"partial:{len(cases)}",
               "call_mode": "single_batch" if args.batch else "one_call_per_case",
               "cache_variant": args.cache_variant,
               "decision_schema": mwe_judge.S3_DECISION_SCHEMA_VERSION,
               "thresholds": {"precision": MIN_PRECISION, "recall": MIN_RECALL,
                              "label_accuracy": MIN_LABEL_ACCURACY,
                              "schema_valid_rate": 1.0, "max_review_rate": MAX_REVIEW_RATE},
               "selected_model": selected, "escalation_policy":
               "local par défaut ; frontière sur les strates où son gain Q0-2 est mesuré, "
               "ainsi que si incertain, confiance < 0.5, schéma invalide ou preuve absente",
               "models": results}
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Évaluation des juges S3", "", f"Modèle retenu : `{selected or 'aucun'}`.", "",
             "| Juge | Précision | Rappel | Exactitude label | Révision | Coût USD | Seuils |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, data in results.items():
        m = data["metrics"]
        precision = "—" if m["precision"] is None else f"{m['precision']:.3f}"
        recall = "—" if m["recall"] is None else f"{m['recall']:.3f}"
        lines.append(f"| {name} (`{data['model']}`) | {precision} | {recall} | "
                     f"{m['label_accuracy']:.3f} | {m['review_rate']:.3f} | {m['cost_usd']:.6f} | "
                     f"{'OK' if data['passes'] else 'ÉCHEC'} |")
    for name, data in results.items():
        lines += ["", f"## {name} par strate", "", "| Strate | N | Précision | Rappel | Label |", "|---|---:|---:|---:|---:|"]
        for stratum, m in data["by_stratum"].items():
            precision = "—" if m["precision"] is None else f"{m['precision']:.3f}"
            recall = "—" if m["recall"] is None else f"{m['recall']:.3f}"
            lines.append(f"| {stratum} | {m['cases']} | {precision} | {recall} | {m['label_accuracy']:.3f} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
