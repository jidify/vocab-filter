"""Benchmark ciblé : pipeline hybride actuel vs décision LLM conjointe.

Ce module est volontairement isolé du pipeline de production. Il lit les
artefacts existants, écrit uniquement des fichiers ``frontier_benchmark_*``
dans ``pipeline_out/`` et ne modifie jamais ``data/sense_fr.jsonl``.

Usage::

    uv run python -m pipeline.eval_frontier_ablation --prepare-only
    uv run python -m pipeline.eval_frontier_ablation
    uv run python -m pipeline.eval_frontier_ablation --report-only
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from nltk.corpus import wordnet as nwn
from pydantic import BaseModel

from pipeline import config, llm_client, sense_fr, senses


CASES_PATH = config.OUT_DIR / "frontier_benchmark_cases.jsonl"
JOINT_PATH = config.OUT_DIR / "frontier_benchmark_joint.jsonl"
JUDGMENTS_PATH = config.OUT_DIR / "frontier_benchmark_judgments.jsonl"
REPORT_PATH = config.OUT_DIR / "frontier_benchmark_report.md"

DEFAULT_CANDIDATE_MODEL = config.SENSE_FR_FRONTIER_MODEL
DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-sol"
EXPORTABLE_STATUSES = {
    "validated", "auto_strong", "auto_llm", "auto_corroborated", "auto_judged", "auto_joint",
}
STRUCTURAL_AGREEMENTS = {
    "sense_id_suspect", "sense_id_douteux", "frontier_reformulation",
    "frontier_explicitation", "frontier_sans_ressource",
}
EVALUATION_DIMENSIONS = ("pos_correct", "sense_correct", "fr_acceptable", "overall")


class JointDecision(BaseModel):
    case_id: str
    pos: Literal["n", "v", "a", "s", "r", "mwe", "other"]
    sense_id: str | None
    meaning_fr: str
    translation_type: Literal["equivalence_directe", "reformulation", "explicitation"]
    confidence: Literal["high", "medium", "low"]
    reason: str


class JointBatch(BaseModel):
    decisions: list[JointDecision]


class AnswerEvaluation(BaseModel):
    pos_correct: bool
    sense_correct: bool
    fr_acceptable: bool
    overall: bool
    reason: str


class PairJudgment(BaseModel):
    case_id: str
    x: AnswerEvaluation
    y: AnswerEvaluation
    preferred: Literal["x", "y", "tie", "neither"]


class JudgmentBatch(BaseModel):
    judgments: list[PairJudgment]


JOINT_SYSTEM = """Tu es lexicographe bilingue anglais-français. Pour chaque occurrence,
décide CONJOINTEMENT sa catégorie grammaticale, son sens et sa traduction française.
Le POS fourni par l'analyseur et la décision actuelle ne sont pas des autorités. Choisis
un sense_id uniquement dans l'inventaire WordNet fourni. Si aucun sens ne décrit l'usage
réel, renvoie sense_id=null et donne quand même le sens contextuel français naturel.
Ne force jamais une occurrence nominale dans un inventaire verbal, ou inversement.
Renvoie exactement une décision par case_id."""


JUDGE_SYSTEM = """Tu es le juge indépendant d'un benchmark lexicographique anglais-français.
Deux réponses anonymisées X et Y décrivent la même occurrence. Évalue chacune séparément.
Le POS doit correspondre à l'usage réel. Le sense_id doit correspondre à sa définition ;
null est correct si WordNet ne contient aucun sens exact. La traduction doit être naturelle
et fidèle dans le contexte. overall est vrai seulement si POS, sens et français sont tous
acceptables. Ignore l'ordre X/Y et ne suppose jamais qu'une réponse vient du pipeline.
Renvoie exactement un jugement par case_id."""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    decoder = json.JSONDecoder(strict=False)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(decoder.decode(line))
                except json.JSONDecodeError:
                    # Même tolérance que senses.load_occurrences_by_sense :
                    # un artefact source corrompu ne doit pas invalider les
                    # milliers d'autres occurrences exploitables.
                    continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    config.ensure_out_dir()
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _cache_key_fields(model: str, system: str, user: str) -> dict:
    return {"model": model, "system": system, "user": user}


def _completion(model: str, system: str, user: str, schema: type[BaseModel],
                prefix: str, reasoning_effort: str) -> tuple[BaseModel, float]:
    """Client unifié (Lot U6 du plan d'unification, ferme M7 —
    fix_pipeline/multi_models/report_multi_models.md §4bis/§5). Ne touche
    PAS à ``DEFAULT_CANDIDATE_MODEL``/``DEFAULT_JUDGE_MODEL`` : ce benchmark
    exige un juge indépendant du modèle candidat par construction
    (config.py:333-336), donc jamais résolu via task_config/
    ALLOWED_FRONTIER_MODELS — seule la mécanique d'appel change ici."""
    return llm_client.call(
        model=model, system=system, prompt=user, response_model=schema,
        cache_key_fields=_cache_key_fields(model, system, user), cache_prefix=f"{prefix}_",
        reasoning_effort=reasoning_effort, max_tokens=16000, return_cost=True,
    )


def _completions(model: str, system: str, users: list[str], schema: type[BaseModel],
                 prefix: str, reasoning_effort: str,
                 max_workers: int = 5) -> tuple[list[BaseModel], float]:
    """Version parallèle et ordonnée de _completion, avec le même cache."""
    items = [
        llm_client.BatchItem(system=system, user=user,
                             cache_key_fields=_cache_key_fields(model, system, user),
                             cache_prefix=f"{prefix}_")
        for user in users
    ]

    def _on_error(_i, _item, exc):
        raise exc

    results, total_cost = llm_client.call_batch_completion(
        items, model=model, response_model=schema,
        reasoning_effort=reasoning_effort, max_tokens=16000, max_workers=max_workers,
        on_error=_on_error,
    )
    if any(result is None for result in results):
        raise RuntimeError("Réponse LLM absente après traitement parallèle.")
    return [result for result in results if result is not None], total_cost


def _inventory(lemma: str, existing_candidates: list[dict] | None = None) -> list[dict]:
    found = {}
    for candidate in existing_candidates or []:
        key = candidate.get("synset")
        if key:
            found[key] = {
                "sense_id": key, "pos": key.split(".")[-2],
                "definition": candidate.get("definition") or "",
            }
    lookup = lemma.strip().replace(" ", "_")
    for synset in nwn.synsets(lookup):
        found[synset.name()] = {
            "sense_id": synset.name(), "pos": synset.pos(),
            "definition": synset.definition(),
        }
    return sorted(found.values(), key=lambda x: (x["pos"], x["sense_id"]))


def _case_from_store(entry: dict, occurrence: dict, stratum: str) -> dict:
    lemma = (entry.get("lemmas_en") or [occurrence.get("target_surface") or ""])[0]
    evidence = entry.get("evidence") or {}
    fr_candidates = list(dict.fromkeys(
        list(evidence.get("omw_fr") or []) + list(evidence.get("wonef") or [])
    ))
    segment_idx = int(occurrence.get("segment_idx", 0))
    return {
        "case_id": f"store:{entry['key']}:{segment_idx}",
        "stratum": stratum,
        "source_key": entry["key"],
        "source_status": entry.get("status"),
        "source_agreement": entry.get("agreement"),
        "lemma": lemma,
        "target_surface": occurrence.get("target_surface") or lemma,
        "context": occurrence.get("context") or "",
        "analyzer_pos": entry.get("pos") or "mwe",
        "inventory": _inventory(lemma),
        "fr_candidates": fr_candidates,
        "current": {
            "pos": entry.get("pos") or "mwe",
            "sense_id": entry["key"] if entry.get("kind") == "synset" else None,
            "meaning_fr": entry.get("fr") or "",
            "translation_type": entry.get("translation_type") or "",
        },
    }


def _case_from_none(record: dict) -> dict:
    lemma = record.get("word") or record.get("target_surface") or ""
    segment_idx = int(record.get("segment_idx", 0))
    return {
        "case_id": f"none:{lemma}:{record.get('pos')}:{segment_idx}",
        "stratum": "aucun_sens_adapte",
        "source_key": "aucun_sens_adapte",
        "source_status": "rejected_s5",
        "source_agreement": "aucun_sens_adapte",
        "lemma": lemma,
        "target_surface": record.get("target_surface") or lemma,
        "context": record.get("context") or "",
        "analyzer_pos": record.get("pos") or "other",
        "inventory": _inventory(lemma, record.get("candidates") or []),
        "fr_candidates": [],
        "current": {
            "pos": record.get("pos") or "other", "sense_id": None,
            "meaning_fr": "", "translation_type": "",
        },
    }


def _pick(rows: list, n: int, rng: random.Random) -> list:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:min(n, len(rows))]


def _fill_store_cases(entries: list[dict], occurrences: dict[str, list[dict]],
                      n: int, rng: random.Random, stratum: str) -> list[dict]:
    candidates = []
    for entry in entries:
        occs = occurrences.get(entry["key"]) or []
        if not occs:
            continue
        occ = rng.choice(sorted(occs, key=lambda o: o.get("segment_idx", 0)))
        candidates.append(_case_from_store(entry, occ, stratum))
    return _pick(candidates, n, rng)


def prepare_cases(sample_size: int = 150, seed: int = 42) -> list[dict]:
    """Construit l'échantillon sans appel réseau et sans mutation du magasin."""
    if sample_size != 150:
        raise ValueError("Ce benchmark ciblé est calibré pour exactement 150 cas.")
    rng = random.Random(seed)
    store = list(sense_fr.load_store().values())
    occurrences = senses.load_occurrences_by_sense()

    structural = [
        e for e in store if e.get("status") == "pending"
        and e.get("agreement") in STRUCTURAL_AGREEMENTS
    ]
    structural_cases = _fill_store_cases(
        sorted(structural, key=lambda e: e["key"]), occurrences, len(structural), rng,
        "pending_structurel",
    )
    disagreement = [
        e for e in store if e.get("status") == "pending"
        and e.get("agreement") == "frontier_desaccord"
    ]
    pending_cases = structural_cases + _fill_store_cases(
        disagreement, occurrences, 75 - len(structural_cases), rng, "pending_desaccord",
    )

    accepted_cases = []
    quotas = {"auto_strong": 20, "auto_llm": 15, "auto_corroborated": 15}
    for status, quota in quotas.items():
        accepted_cases.extend(_fill_store_cases(
            [e for e in store if e.get("status") == status], occurrences, quota, rng,
            f"accepte_{status}",
        ))

    raw_senses = _read_jsonl(config.SENSES_PATH)
    none_records = [
        r for r in raw_senses if r.get("best_sense") == "aucun_sens_adapte"
        and r.get("context") and r.get("target_surface")
    ]
    none_cases = [_case_from_none(r) for r in _pick(none_records, 25, rng)]
    cases = pending_cases + accepted_cases + none_cases
    if len(cases) != sample_size:
        raise RuntimeError(
            f"Échantillon incomplet: {len(cases)}/{sample_size}. "
            "Vérifier les occurrences disponibles par strate."
        )
    rng.shuffle(cases)
    _write_jsonl(CASES_PATH, cases)
    return cases


def _joint_prompt(batch: list[dict]) -> str:
    public = [{
        "case_id": c["case_id"], "lemma": c["lemma"],
        "target_surface": c["target_surface"], "context": c["context"],
        "analyzer_pos_hint": c["analyzer_pos"], "wordnet_inventory": c["inventory"],
        "french_candidates_untrusted": c["fr_candidates"],
    } for c in batch]
    return "Cas à analyser :\n" + json.dumps(public, ensure_ascii=False)


def run_joint(cases: list[dict], model: str, batch_size: int = 10) -> tuple[list[dict], float]:
    rows, total_cost = [], 0.0
    allowed = {c["case_id"]: {i["sense_id"] for i in c["inventory"]} for c in cases}
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        parsed, cost = _completion(
            model, JOINT_SYSTEM, _joint_prompt(batch), JointBatch,
            "frontier_benchmark_joint", "low",
        )
        total_cost += cost
        by_id = {d.case_id: d for d in parsed.decisions}
        for case in batch:
            decision = by_id.get(case["case_id"])
            if decision is None:
                raise RuntimeError(f"Réponse conjointe absente pour {case['case_id']}")
            if decision.sense_id is not None and decision.sense_id not in allowed[case["case_id"]]:
                raise RuntimeError(
                    f"sense_id inventé pour {case['case_id']}: {decision.sense_id}"
                )
            rows.append(decision.model_dump())
        print(f"  décision conjointe: {min(start + batch_size, len(cases))}/{len(cases)}")
    _write_jsonl(JOINT_PATH, rows)
    return rows, total_cost


def _answer_payload(answer: dict) -> dict:
    return {
        "pos": answer.get("pos"), "sense_id": answer.get("sense_id"),
        "meaning_fr": answer.get("meaning_fr") or "",
        "translation_type": answer.get("translation_type") or "",
    }


def _judge_prompt(batch: list[dict], joint: dict[str, dict], seed: int,
                  pass_number: int) -> tuple[str, dict[str, bool]]:
    ordering, public = {}, []
    for case in batch:
        first_joint = bool(random.Random(f"{seed}:{case['case_id']}").getrandbits(1))
        if pass_number == 2:
            first_joint = not first_joint
        ordering[case["case_id"]] = first_joint
        current = _answer_payload(case["current"])
        proposed = _answer_payload(joint[case["case_id"]])
        x, y = (proposed, current) if first_joint else (current, proposed)
        public.append({
            "case_id": case["case_id"], "target_surface": case["target_surface"],
            "context": case["context"], "wordnet_inventory": case["inventory"],
            "X": x, "Y": y,
        })
    return "Paires à juger :\n" + json.dumps(public, ensure_ascii=False), ordering


def _normalize_judgment(judgment: PairJudgment, first_joint: bool) -> dict:
    joint_eval = judgment.x if first_joint else judgment.y
    current_eval = judgment.y if first_joint else judgment.x
    preferred = judgment.preferred
    if preferred in ("x", "y"):
        preferred = "joint" if (preferred == "x") == first_joint else "current"
    return {
        "case_id": judgment.case_id,
        "current": current_eval.model_dump(), "joint": joint_eval.model_dump(),
        "preferred": preferred,
    }


def run_judge(cases: list[dict], joint_rows: list[dict], model: str,
              seed: int = 42, batch_size: int = 10) -> tuple[list[dict], float]:
    joint = {r["case_id"]: r for r in joint_rows}
    passes: list[dict[str, dict]] = []
    total_cost = 0.0
    for pass_number in (1, 2):
        normalized = {}
        tasks = []
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            prompt, ordering = _judge_prompt(batch, joint, seed, pass_number)
            tasks.append((batch, prompt, ordering))
        parsed_batches, cost = _completions(
            model, JUDGE_SYSTEM, [task[1] for task in tasks], JudgmentBatch,
            f"frontier_benchmark_judge_p{pass_number}", "high",
        )
        total_cost += cost
        for task_index, ((batch, _prompt, ordering), parsed) in enumerate(
            zip(tasks, parsed_batches), start=1
        ):
            by_id = {j.case_id: j for j in parsed.judgments}
            for case in batch:
                judgment = by_id.get(case["case_id"])
                if judgment is None:
                    raise RuntimeError(f"Jugement absent pour {case['case_id']}")
                normalized[case["case_id"]] = _normalize_judgment(
                    judgment, ordering[case["case_id"]],
                )
            print(
                f"  jugement passe {pass_number}: "
                f"{min(task_index * batch_size, len(cases))}/{len(cases)}"
            )
        passes.append(normalized)

    rows = []
    by_case = {c["case_id"]: c for c in cases}
    for case_id in by_case:
        a, b = passes[0][case_id], passes[1][case_id]
        stable = all(
            a[side][dimension] == b[side][dimension]
            for side in ("current", "joint")
            for dimension in EVALUATION_DIMENSIONS
        )
        rows.append({
            "case_id": case_id, "stratum": by_case[case_id]["stratum"],
            "source_status": by_case[case_id]["source_status"],
            "pass_1": a, "pass_2": b, "stable": stable,
            "preference_stable": a["preferred"] == b["preferred"],
        })
    _write_jsonl(JUDGMENTS_PATH, rows)
    return rows, total_cost


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def _previous_report_cost() -> float:
    if not REPORT_PATH.exists():
        return 0.0
    for line in REPORT_PATH.read_text(encoding="utf-8").splitlines():
        if "Coût API non caché constaté" in line and "$" in line:
            try:
                return float(line.rsplit("$", 1)[1])
            except ValueError:
                return 0.0
    return 0.0


def build_report(cases: list[dict], judgments: list[dict], candidate_model: str,
                 judge_model: str, api_cost: float = 0.0) -> str:
    stable = [r for r in judgments if r["stable"]]
    current_ok = sum(r["pass_1"]["current"]["overall"] for r in stable)
    joint_ok = sum(r["pass_1"]["joint"]["overall"] for r in stable)
    unstable_rate = (len(judgments) - len(stable)) / len(judgments) if judgments else 1.0

    pending = [r for r in stable if r["stratum"].startswith("pending_")]
    accepted = [r for r in stable if r["stratum"].startswith("accepte_")]
    p_current = sum(r["pass_1"]["current"]["overall"] for r in pending)
    p_joint = sum(r["pass_1"]["joint"]["overall"] for r in pending)
    accepted_regressions = sum(
        r["pass_1"]["current"]["overall"] and not r["pass_1"]["joint"]["overall"]
        for r in accepted
    )
    overall_delta = (joint_ok - current_ok) / len(stable) if stable else -1.0
    pending_delta = (p_joint - p_current) / len(pending) if pending else -1.0
    recommend_joint = (
        overall_delta >= 0.10 and pending_delta >= 0.20
        and accepted_regressions <= 2 and unstable_rate < 0.10
    )

    strata = Counter(r["stratum"] for r in cases)
    dimensions = EVALUATION_DIMENSIONS
    dimension_labels = {
        "pos_correct": "POS correct", "sense_correct": "Sens correct",
        "fr_acceptable": "Français acceptable", "overall": "Réussite de bout en bout",
    }
    gate_rows = [
        ("Gain global ≥10 points", overall_delta >= 0.10,
         f"{100 * overall_delta:.1f} points"),
        ("Gain pending ≥20 points", pending_delta >= 0.20,
         f"{100 * pending_delta:.1f} points"),
        ("Régressions contrôles ≤2", accepted_regressions <= 2,
         str(accepted_regressions)),
        ("Jugements instables <10 %", unstable_rate < 0.10,
         f"{100 * unstable_rate:.1f}%"),
    ]
    lines = [
        "# Benchmark : mécanique hybride vs décision conjointe", "",
        "## Configuration", "",
        f"- Candidat conjoint : `{candidate_model}`",
        f"- Juge : `{judge_model}` (deux passes, ordre X/Y inversé)",
        f"- Cas : {len(cases)} — " + ", ".join(f"{k}={v}" for k, v in sorted(strata.items())),
        f"- Jugements stables : {len(stable)}/{len(judgments)} ({_pct(len(stable), len(judgments))})",
        f"- Préférences X/Y stables : "
        f"{sum(r.get('preference_stable', r['pass_1']['preferred'] == r['pass_2']['preferred']) for r in judgments)}/{len(judgments)}",
        f"- Coût API non caché constaté : ${api_cost:.4f}", "",
        "## Résultats", "",
        "| Mesure | Pipeline actuel | Décision conjointe |", "|---|---:|---:|",
        f"| Réussite globale, cas stables | {current_ok}/{len(stable)} ({_pct(current_ok, len(stable))}) | {joint_ok}/{len(stable)} ({_pct(joint_ok, len(stable))}) |",
        f"| Réussite sur les pending | {p_current}/{len(pending)} ({_pct(p_current, len(pending))}) | {p_joint}/{len(pending)} ({_pct(p_joint, len(pending))}) |",
        "",
        "### Par dimension", "",
        "| Dimension | Pipeline actuel | Décision conjointe |", "|---|---:|---:|",
    ]
    for dimension in dimensions:
        n_current = sum(r["pass_1"]["current"][dimension] for r in stable)
        n_joint = sum(r["pass_1"]["joint"][dimension] for r in stable)
        lines.append(
            f"| {dimension_labels[dimension]} | {n_current}/{len(stable)} "
            f"({_pct(n_current, len(stable))}) | {n_joint}/{len(stable)} "
            f"({_pct(n_joint, len(stable))}) |"
        )
    lines.extend(["", "### Par strate", "",
                  "| Strate | Stables/total | Pipeline actuel | Décision conjointe |",
                  "|---|---:|---:|---:|"])
    for stratum in sorted(strata):
        stratum_rows = [r for r in stable if r["stratum"] == stratum]
        cur = sum(r["pass_1"]["current"]["overall"] for r in stratum_rows)
        joi = sum(r["pass_1"]["joint"]["overall"] for r in stratum_rows)
        lines.append(
            f"| {stratum} | {len(stratum_rows)}/{strata[stratum]} | "
            f"{cur}/{len(stratum_rows)} ({_pct(cur, len(stratum_rows))}) | "
            f"{joi}/{len(stratum_rows)} ({_pct(joi, len(stratum_rows))}) |"
        )
    lines.extend([
        "",
        f"- Bons résultats actuellement bloqués (`pending`) : **{p_current}**.",
        f"- Mauvais résultats acceptés dans le groupe témoin : **{len(accepted) - sum(r['pass_1']['current']['overall'] for r in accepted)}**.",
        f"- Régressions de la décision conjointe sur les contrôles acceptés : **{accepted_regressions}**.",
        "",
        "## Décision", "",
        ("**Les seuils sont atteints : recommander la décision conjointe POS/sens/traduction.**"
         if recommend_joint else
         "**Les seuils ne sont pas tous atteints : conserver l'architecture actuelle ou cibler seulement les strates gagnantes.**"),
        "", "| Seuil | Valeur | Résultat |", "|---|---:|---:|",
    ])
    for label, passed, value in gate_rows:
        lines.append(f"| {label} | {value} | {'PASS' if passed else 'FAIL'} |")
    lines.extend([
        "",
        "## Limite", "",
        "Le juge est un LLM OpenAI : ce rapport mesure son appréciation comparative, "
        "pas une précision humaine absolue. L'anonymisation et l'inversion de l'ordre "
        "réduisent la circularité sans l'éliminer.", "",
    ])
    return "\n".join(lines)


def run(sample_size: int = 150, seed: int = 42,
        candidate_model: str = DEFAULT_CANDIDATE_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        prepare_only: bool = False, report_only: bool = False) -> int:
    if report_only:
        cases, judgments = _read_jsonl(CASES_PATH), _read_jsonl(JUDGMENTS_PATH)
        if not cases or not judgments:
            raise RuntimeError("Artefacts de benchmark absents ; exécuter d'abord le benchmark.")
        report = build_report(
            cases, judgments, candidate_model, judge_model, _previous_report_cost(),
        )
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(REPORT_PATH)
        return 0

    cases = prepare_cases(sample_size, seed)
    print(f"Échantillon écrit: {CASES_PATH} ({len(cases)} cas)")
    if prepare_only:
        return 0
    joint, joint_cost = run_joint(cases, candidate_model)
    judgments, judge_cost = run_judge(cases, joint, judge_model, seed)
    api_cost = max(joint_cost + judge_cost, _previous_report_cost())
    report = build_report(cases, judgments, candidate_model, judge_model, api_cost)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Rapport: {REPORT_PATH}")
    print(f"Coût API non caché constaté: ${api_cost:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    return run(**vars(args))


if __name__ == "__main__":
    raise SystemExit(main())
