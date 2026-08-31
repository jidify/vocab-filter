"""S6-3 (plan §6, `fix_pipeline/plan_action_fix_pipeline.md`) — mesure de la
traduction française sur le benchmark SANS FUITE.

Comme `fix_pipeline/evaluate_fix_quality.py` (Q0-1), ce module lit le
benchmark `pipeline_out/vocab_corrige.csv` SEULEMENT APRÈS génération de
`pipeline_out/vocab.csv` par le pipeline de production (S6-1/S6-2 : le
traducteur ne reçoit jamais ce fichier, voir
`test_s6_3_translation_leakfree.py::NoBenchmarkLeakTests`). Il vit
délibérément hors de `pipeline/` et réutilise l'appariement multiensemble
`(canonical_form, unit_type)` de Q0-1 (`match_rows`) pour ne jamais comparer
un jeu de lignes différent entre les deux rapports.

Au-delà de l'égalité de chaîne souple de Q0-1 (`_soft_fr_equal` : égalité
normalisée, ou variante explicitement séparée par `/`/`;`, jamais un
jugement sémantique — voir sa propre section "Limites explicites"), ce
module ajoute un jugement LLM (CatGPT, en lot) qui distingue une VARIANTE
ACCEPTABLE (synonyme fidèle au sens visé) d'un CONTRESENS (sens différent),
sur les paires où les deux traductions sont non vides mais diffèrent
littéralement. Le rapport ventile la fidélité par statut (`fr_status`),
source (`agreement` du magasin `data/sense_fr.jsonl`) et modèle traducteur,
et écrit un échantillon auditable pour vérification humaine — un jugement
LLM sur lui-même n'est jamais une preuve absolue (plan §5.5).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config, llm_client
from pipeline.llm_tasks import task_config
from fix_pipeline.evaluate_fix_quality import _soft_fr_equal, match_rows, read_csv

# ============================================================
# Configuration du juge — CatGPT, en lot, timeout long (appel explicitement
# fixé par le plan, indépendant de config.CATGPT_TIMEOUT (300s par défaut,
# dimensionné pour les appels de production unitaires/lots plus courts) :
# un lot de JUDGE_BATCH_SIZE=50 paires à juger est un prompt nettement plus
# long, qui a mesurablement besoin de plus de marge.
# ============================================================

JUDGE_MODEL = f"catgpt/{config.CATGPT_MODEL}"
JUDGE_BATCH_SIZE = 50
JUDGE_TIMEOUT_SECONDS = 1200.0
JUDGE_PROTOCOL = "s6-3-translation-leakfree-judge-2"  # v2 : ajoute contexte_en au prompt (v1 jugeait sans phrase du livre)
JUDGE_CACHE_PREFIX = "s6_3_translation_judge_"

AUDIT_SAMPLE_SIZE = 30
AUDIT_SAMPLE_SEED = 20260830  # date de la correction -> échantillon reproductible

VERDICTS = {"equivalent", "synonyme_acceptable", "contresens", "incertain"}
ACCEPTABLE_VERDICTS = {"equivalent", "synonyme_acceptable"}


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"numerator": numerator, "denominator": denominator,
            "value": round(numerator / denominator, 6) if denominator else None}


# ============================================================
# Magasin data/sense_fr.jsonl — attribution source/modèle
# ============================================================

def load_sense_fr_store(path: Path) -> dict[str, dict]:
    store: dict[str, dict] = {}
    if not path.exists():
        return store
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            store[entry["key"]] = entry
    return store


def translation_source(entry: dict | None) -> str:
    """`agreement` du magasin : quelle(s) ressource(s) fondent la
    traduction (concordance dictionnaire, frontière, source unique...) —
    voir pipeline/sense_fr.py et pipeline/sense_fr_frontier.py."""
    if entry is None:
        return "absent_du_magasin"
    return entry.get("agreement") or "inconnu"


def translation_model(entry: dict | None, *, local_translate_model: str) -> str:
    """Modèle réellement responsable de la traduction retenue :
    - une entrée `validated` est humaine, quelle que soit la preuve LLM
      qui l'a précédée (S6-1 : la relecture humaine prime toujours) ;
    - `evidence.frontier_model` si une des 4 tâches S6 batchées (LiteLLM)
      a produit ou corroboré `fr` ;
    - le modèle de la tâche locale S6-translate-local si `evidence.llm_votes`
      est non vide (consensus/rétro-traduction locale a pesé sur la décision,
      voir pipeline/sense_fr.py::classify_synset_key/classify_mwe_key) ;
    - sinon la traduction vient uniquement des ressources dictionnaire
      (omw-fr/WoNeF), aucun modèle n'a été consulté."""
    if entry is None:
        return "absent_du_magasin"
    if entry.get("status") == "validated":
        return "humain"
    evidence = entry.get("evidence") or {}
    frontier_model = evidence.get("frontier_model")
    if frontier_model:
        return frontier_model
    if evidence.get("llm_votes"):
        return local_translate_model
    return "dictionnaire_seul"


# ============================================================
# Jugement LLM en lot — distingue variante acceptable et contresens
# ============================================================

def _judge_prompt(pairs: list[dict]) -> str:
    items = "\n".join(
        f'{i + 1}. id={p["id"]!r}; anglais={p["lemmas_en"]!r}; definition={p["definition_en"]!r}; '
        f'contexte_livre={(p["contexte_en"] or "(aucun)")!r}; '
        f'traduction_a_juger={p["actual_fr"]!r}; alternatives_a_juger={p["actual_fr_alt"]!r}; '
        f'traduction_reference={p["benchmark_fr"]!r}; alternatives_reference={p["benchmark_fr_alt"]!r}'
        for i, p in enumerate(pairs)
    )
    return f'''Tu es lexicographe bilingue anglais-français. Pour chacune des {len(pairs)} paires
suivantes, une "traduction_a_juger" (et ses "alternatives_a_juger") est comparée
à une "traduction_reference" (et ses "alternatives_reference"), toutes deux
censées traduire le MÊME sens anglais précis (voir "definition"). "contexte_livre"
donne une ou plusieurs phrases réelles où ce sens est employé (séparées par
" || " s'il y en a plusieurs) — utilise-le en priorité pour trancher un cas
ambigu (transitif/réfléchi, registre, polysémie) ; s'il vaut "(aucun)", juge
seulement sur la définition. Classe la relation entre les deux traductions
dans exactement une catégorie :
- "equivalent" : même mot/expression, ou variante orthographique/flexion triviale ;
- "synonyme_acceptable" : mot ou expression différent mais qui transmet
  fidèlement le même sens dans ce sens précis (paraphrase, registre différent,
  synonyme correct — y compris via une des alternatives) ;
- "contresens" : la traduction à juger change le sens (autre sens du mot,
  polarité inversée, sens trop large/étroit qui trahit la définition) ;
- "incertain" : impossible de trancher, même avec le contexte du livre.

{items}

Réponds uniquement avec ce JSON compact, sans explication ni champ supplémentaire :
{{"verdicts":[{{"id":"<id exact>","verdict":"<categorie>","reason":"<justification en une courte phrase>"}}]}}
Il doit y avoir exactement un verdict par id, dans le même ordre.'''


def _judge_batch(pairs: list[dict]) -> dict[str, dict]:
    if not pairs:
        return {}
    if len(pairs) > JUDGE_BATCH_SIZE:
        raise ValueError(f"lot de {len(pairs)} paires > JUDGE_BATCH_SIZE={JUDGE_BATCH_SIZE}")
    prompt = _judge_prompt(pairs)
    system = ("Tu juges la fidélité sémantique de traductions françaises face à une "
              "traduction de référence, jamais leur seule qualité stylistique.")
    try:
        raw = llm_client.call(
            model=JUDGE_MODEL, system=system, prompt=prompt, timeout=JUDGE_TIMEOUT_SECONDS,
            cache_key_fields=llm_client.build_cache_key(
                model=JUDGE_MODEL, system=system, prompt=prompt,
                extra={"protocol": JUDGE_PROTOCOL, "pair_ids": [p["id"] for p in pairs]},
            ),
            cache_prefix=JUDGE_CACHE_PREFIX,
        )
    except llm_client.LLMError:
        raw = {}
    expected_ids = {p["id"] for p in pairs}
    received: dict[str, dict] = {}
    duplicates: set[str] = set()
    for item in raw.get("verdicts", []) if isinstance(raw, dict) else []:
        pid = item.get("id") if isinstance(item, dict) else None
        if pid in received:
            duplicates.add(pid)
        elif pid in expected_ids:
            received[pid] = item
    out: dict[str, dict] = {}
    for p in pairs:
        item = received.get(p["id"])
        if item is None or p["id"] in duplicates or item.get("verdict") not in VERDICTS:
            out[p["id"]] = {"verdict": "incertain", "reason": "réponse manquante, dupliquée ou hors schéma"}
        else:
            out[p["id"]] = {"verdict": item["verdict"], "reason": item.get("reason") or ""}
    return out


def judge_pairs(pairs: list[dict]) -> dict[str, dict]:
    """Un appel CatGPT par tranche d'au plus JUDGE_BATCH_SIZE — jamais un
    appel par paire (voir la consigne de lot du plan)."""
    results: dict[str, dict] = {}
    for i in range(0, len(pairs), JUDGE_BATCH_SIZE):
        results.update(_judge_batch(pairs[i:i + JUDGE_BATCH_SIZE]))
    return results


# ============================================================
# Évaluation
# ============================================================

def _benchmark_variants(primary: str, alt: str) -> list[str]:
    variants = [primary] + [v.strip() for v in re.split(r"[/;]", alt) if v.strip()]
    return [v for v in variants if v]


def evaluate(
    actual_rows: list[dict[str, str]], benchmark_rows: list[dict[str, str]],
    sense_fr_store: dict[str, dict], *,
    judge=judge_pairs, limit: int | None = None,
) -> dict:
    pairs, _only_a, _only_e = match_rows(actual_rows, benchmark_rows)
    local_translate_model = task_config("S6-translate-local").model

    rows: list[dict] = []
    to_judge: list[dict] = []
    for i, p in enumerate(pairs):
        a, e = actual_rows[p.actual_index], benchmark_rows[p.expected_index]
        actual_fr = (a.get("meaning_fr_official") or "").strip()
        benchmark_fr = (e.get("meaning_fr_official") or "").strip()
        if not actual_fr or not benchmark_fr:
            # Couverture (traduction officielle vide) est mesurée par Q0-1 ;
            # une fidélité sémantique ne peut pas se juger contre du vide.
            continue
        entry = sense_fr_store.get(a.get("sense_id") or "")
        row = {
            "id": f"pair-{i}",
            "canonical_form": e.get("canonical_form", ""), "unit_type": e.get("unit_type", ""),
            "sense_id": a.get("sense_id") or e.get("sense_id") or "",
            "definition_en": a.get("definition_en") or e.get("definition_en") or "",
            "lemmas_en": a.get("canonical_form", ""),
            # Phrase(s) réelles du livre courant, déjà produites par S7/export.py
            # (colonne `contexte_en` de vocab.csv/vocab_corrige.csv, voir
            # pipeline/sense_fr.py::format_occurrences_en) — jamais recalculées
            # ici : indispensable pour trancher un cas ambigu (transitif/réfléchi,
            # polysémie) sans quoi ni le juge ni un relecteur humain ne le peuvent.
            "contexte_en": a.get("contexte_en") or e.get("contexte_en") or "",
            "actual_fr": actual_fr, "actual_fr_alt": a.get("meaning_fr_alt") or "",
            "benchmark_fr": benchmark_fr, "benchmark_fr_alt": e.get("meaning_fr_alt") or "",
            "status": a.get("fr_status") or "",
            "source": translation_source(entry),
            "model": translation_model(entry, local_translate_model=local_translate_model),
        }
        variants = _benchmark_variants(benchmark_fr, e.get("meaning_fr_alt") or "")
        if any(_soft_fr_equal(actual_fr, v) for v in variants):
            row.update(verdict="equivalent",
                       reason="égalité normalisée ou variante explicite (Q0-1::_soft_fr_equal)",
                       judged_by="deterministic")
        else:
            row["judged_by"] = "llm"
            to_judge.append(row)
        rows.append(row)

    if limit is not None:
        to_judge = to_judge[:limit]
        keep_ids = {r["id"] for r in to_judge}
        rows = [r for r in rows if r["judged_by"] == "deterministic" or r["id"] in keep_ids]

    verdicts = judge(to_judge) if to_judge else {}
    for row in to_judge:
        v = verdicts.get(row["id"]) or {"verdict": "incertain", "reason": "non jugé (réponse absente)"}
        row["verdict"] = v["verdict"]
        row["reason"] = v.get("reason", "")

    return _summarize(rows)


def _group_by(rows: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    out: dict[str, dict] = {}
    for key in sorted(groups):
        items = groups[key]
        n = len(items)
        acceptable = sum(item["verdict"] in ACCEPTABLE_VERDICTS for item in items)
        out[key] = {
            "n": n, "acceptable": acceptable,
            "contresens": sum(item["verdict"] == "contresens" for item in items),
            "incertain": sum(item["verdict"] == "incertain" for item in items),
            "acceptable_rate": round(acceptable / n, 6) if n else None,
        }
    return out


_CONTRESENS_FIELDS = (
    "canonical_form", "unit_type", "sense_id", "actual_fr", "benchmark_fr",
    "benchmark_fr_alt", "definition_en", "contexte_en", "status", "source", "model", "reason",
)


def _summarize(rows: list[dict]) -> dict:
    judged = [r for r in rows if r["judged_by"] == "llm"]
    total = len(rows)
    acceptable = sum(r["verdict"] in ACCEPTABLE_VERDICTS for r in rows)
    counts = Counter(r["verdict"] for r in rows)
    contresens = [{k: r[k] for k in _CONTRESENS_FIELDS} for r in rows if r["verdict"] == "contresens"]
    return {
        "schema_version": 1,
        "protocol": JUDGE_PROTOCOL,
        "judge_model": JUDGE_MODEL,
        "judge_batch_size": JUDGE_BATCH_SIZE,
        "judge_timeout_seconds": JUDGE_TIMEOUT_SECONDS,
        "counts": {
            "total_pairs_with_both_translations": total,
            "judged_by_llm": len(judged),
            "judged_deterministically": total - len(judged),
            **{f"verdict_{v}": counts.get(v, 0) for v in sorted(VERDICTS)},
        },
        "metrics": {"semantic_fidelity_rate": _ratio(acceptable, total)},
        "by_status": _group_by(rows, lambda r: r["status"] or "(vide)"),
        "by_source": _group_by(rows, lambda r: r["source"]),
        "by_model": _group_by(rows, lambda r: r["model"]),
        "contresens": contresens,
        "rows": rows,
    }


# ============================================================
# Échantillon auditable
# ============================================================

AUDIT_FIELDS = [
    "canonical_form", "unit_type", "sense_id", "definition_en", "contexte_en",
    "actual_fr", "actual_fr_alt", "benchmark_fr", "benchmark_fr_alt",
    "status", "source", "model", "verdict", "reason",
    "human_verdict", "human_note",
]


def build_audit_sample(rows: list[dict], *, size: int = AUDIT_SAMPLE_SIZE, seed: int = AUDIT_SAMPLE_SEED) -> list[dict]:
    """Tous les contresens jugés par le LLM (dans la limite de `size`),
    complétés par un tirage aléatoire REPRODUCTIBLE (graine fixe) des autres
    verdicts — un accord LLM/humain ne se vérifie que sur un échantillon des
    deux catégories, pas seulement sur les cas déjà signalés comme fautifs."""
    judged = [r for r in rows if r["judged_by"] == "llm"]
    contresens = [r for r in judged if r["verdict"] == "contresens"]
    others = [r for r in judged if r["verdict"] != "contresens"]
    sample = list(contresens[:size])
    remaining = max(0, size - len(sample))
    if remaining and others:
        sample += random.Random(seed).sample(others, min(remaining, len(others)))
    sample.sort(key=lambda r: r["id"])
    return sample


def write_audit_sample(rows: list[dict], path: Path, *, size: int = AUDIT_SAMPLE_SIZE, seed: int = AUDIT_SAMPLE_SEED) -> int:
    sample = build_audit_sample(rows, size=size, seed=seed)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sample:
            out = dict(row)
            out["human_verdict"] = ""
            out["human_note"] = ""
            writer.writerow(out)
    return len(sample)


# ============================================================
# Rapport
# ============================================================

def _table(title: str | None, group: dict[str, dict]) -> list[str]:
    lines = ([f"## {title}", ""] if title else []) + [
        "| Clé | N | Acceptable | Contresens | Incertain | Taux acceptable |", "|---|---:|---:|---:|---:|---:|"]
    for key, g in group.items():
        rate = "n/a" if g["acceptable_rate"] is None else f"{100 * g['acceptable_rate']:.1f} %"
        lines.append(f"| {key} | {g['n']} | {g['acceptable']} | {g['contresens']} | {g['incertain']} | {rate} |")
    lines.append("")
    return lines


def render_report(result: dict, actual_path: Path, benchmark_path: Path, audit_sample_path: Path) -> str:
    c, m = result["counts"], result["metrics"]
    fidelity = m["semantic_fidelity_rate"]
    fidelity_str = "n/a" if fidelity["value"] is None else f"{100 * fidelity['value']:.2f} %"
    lines = [
        "# Rapport S6-3 — fidélité de traduction (sans fuite)", "",
        f"Comparaison en lecture seule de `{actual_path.as_posix()}` (généré AVANT toute lecture du "
        f"benchmark — voir S6-1/S6-2) à `{benchmark_path.as_posix()}`, avec jugement sémantique "
        f"(`{result['judge_model']}`, lot de {result['judge_batch_size']}, timeout "
        f"{result['judge_timeout_seconds']:.0f}s) au-delà de l'égalité de chaîne souple de Q0-1.", "",
        f"- Paires comparées (traduction non vide des deux côtés) : {c['total_pairs_with_both_translations']}.",
        f"- Jugées par égalité déterministe (Q0-1) : {c['judged_deterministically']} ; "
        f"jugées par le LLM : {c['judged_by_llm']}.",
        f"- Fidélité sémantique (`equivalent` + `synonyme_acceptable`) : {fidelity_str} "
        f"({fidelity['numerator']}/{fidelity['denominator']}).",
        "- Verdicts : " + ", ".join(f"{v}={c.get('verdict_' + v, 0)}" for v in sorted(VERDICTS)), "",
    ]
    lines += _table("Par statut (`fr_status`)", result["by_status"])
    lines += _table("Par source (`agreement` du magasin)", result["by_source"])
    lines += ["## Par modèle", "",
              f"Modèle qui a PRODUIT la traduction évaluée (`pipeline/sense_fr*.py`), pas le juge S6-3 "
              f"— celui-ci est toujours `{result['judge_model']}`, voir l'en-tête ci-dessus.", ""]
    lines += _table(None, result["by_model"])
    lines += [f"## Contresens résiduels ({len(result['contresens'])})", "",
              "Traductions dont le sens diverge réellement du sens visé — jamais une simple "
              "différence de forme (déjà écartée par l'égalité déterministe ci-dessus).", ""]
    if result["contresens"]:
        for row in result["contresens"]:
            context = row.get("contexte_en") or "(aucun contexte)"
            if len(context) > 160:
                context = context[:157] + "..."
            lines.append(f"- `{row['canonical_form']}` ({row['sense_id'] or 'sans sense_id'}) : "
                         f"« {row['actual_fr']} » vs référence « {row['benchmark_fr']} » — {row['reason']} "
                         f"[contexte : {context}]")
    else:
        lines.append("(aucun)")
    lines += ["", "## Échantillon auditable", "",
              f"Tous les contresens détectés par le LLM, complétés par un tirage aléatoire reproductible "
              f"(graine {AUDIT_SAMPLE_SEED}) des autres verdicts jusqu'à {AUDIT_SAMPLE_SIZE} paires, sont "
              f"écrits dans `{audit_sample_path.as_posix()}` — colonnes `human_verdict`/`human_note` vides, "
              "à remplir par un relecteur humain pour mesurer l'accord réel avec ce jugement automatique.", "",
              "## Limites explicites", "",
              "Le jugement LLM n'est pas une source indépendante au sens du plan §5.5 : c'est un troisième "
              "avis, jamais une preuve absolue — d'où l'échantillon auditable ci-dessus. Les paires où l'une "
              "des deux traductions officielles est vide sont exclues de cette mesure de fidélité (couverture "
              "déjà mesurée séparément par Q0-1::official_fr_coverage).", ""]
    return "\n".join(lines)


# ============================================================
# Orchestration
# ============================================================

def run(
    actual_path: Path, benchmark_path: Path, sense_fr_store_path: Path,
    json_path: Path, report_path: Path, audit_sample_path: Path,
    *, judge=judge_pairs, limit: int | None = None,
) -> dict:
    actual_rows = read_csv(actual_path)
    benchmark_rows = read_csv(benchmark_path)
    sense_fr_store = load_sense_fr_store(sense_fr_store_path)
    result = evaluate(actual_rows, benchmark_rows, sense_fr_store, judge=judge, limit=limit)
    # Sorties disjointes des deux entrées ; le benchmark n'est jamais ouvert en écriture.
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result, actual_path, benchmark_path, audit_sample_path), encoding="utf-8")
    write_audit_sample(result["rows"], audit_sample_path)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=config.OUT_DIR / "vocab.csv")
    parser.add_argument("--benchmark", type=Path, default=config.OUT_DIR / "vocab_corrige.csv")
    parser.add_argument("--sense-fr-store", type=Path, default=config.SENSE_FR_STORE_PATH)
    parser.add_argument("--json", type=Path, default=config.OUT_DIR / "s6_3_translation_leakfree_metrics.json")
    parser.add_argument("--report", type=Path, default=config.OUT_DIR / "s6_3_translation_leakfree_report.md")
    parser.add_argument("--audit-sample", type=Path, default=config.OUT_DIR / "s6_3_translation_audit_sample.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite le nombre de paires envoyées au LLM (diagnostic, sans lancer le corpus entier).")
    args = parser.parse_args(argv)
    if args.benchmark.resolve() in {args.json.resolve(), args.report.resolve(), args.audit_sample.resolve()}:
        parser.error("output paths must not overwrite the benchmark")
    run(args.actual, args.benchmark, args.sense_fr_store, args.json, args.report, args.audit_sample, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
