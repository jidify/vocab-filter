"""Phase 3 — mesure ``rules_plus`` (``plan_detection_benchmark_funnel.md``,
section Phase 3) : union de la Baseline 2 (l'ensemble réel du pipeline
actuel, Phase 2 — jamais réimplémentée, mêmes fonctions de production que
``phase2_run_baselines.py``) avec les générateurs neufs de
``rules_plus.py`` (WordNet candidats, PARSEME, patron phrasal verb,
règles de bornes). spaCy ne reçoit jamais de pouvoir de rejet : on ajoute
à sa sortie, on n'en retire jamais rien.

Comme Phase 2, tourne UNIQUEMENT sur les 99 segments du corpus gold, jamais
le livre entier. Un second passage ``nlp.pipe`` (même tokenizer que la
production, ``pipeline.analyze.get_nlp()``, cas spéciaux e-mail/lexique
custom inclus) donne accès aux ``Doc`` spaCy dont ``analyze_segments`` ne
laisse rien échapper à l'appelant — nécessaire aux scanners lemme/POS de
``rules_plus.py``. Coût négligeable (99 segments courts, ~1s).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

from pipeline import analyze, config, mwe
from pipeline.corpus import load_segments

from fix_pipeline.detection_benchmark import normalize_adapter, rules_plus, scorer

OUT_DIR = config.ROOT / "pipeline_out" / "detection_benchmark"
REPORT_PATH = Path(__file__).resolve().parent / "phase3_rules_plus_report.md"

_TOKEN_RE = re.compile(r"\w+")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _dedupe(candidates: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for c in candidates:
        key = (c["segment_idx"], scorer.resolve_span(c))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_idxs = {g.segment_idx for g in scorer.load_gold()}
    all_segments = load_segments()
    play_segments = [s for s in all_segments if s.kind != "hors_oeuvre"]
    subset = [s for s in play_segments if s.idx in gold_idxs]
    assert len(subset) == len(gold_idxs), (
        "segment_idx manquants entre le corpus gold et pipeline.corpus.load_segments()"
    )
    total_tokens = sum(len(_TOKEN_RE.findall(s.en)) for s in subset)

    # ------------------------------------------------------------------
    # Baseline 2 (reprise à l'identique de phase2_run_baselines.py) — la
    # base sur laquelle rules_plus s'ajoute par union.
    # ------------------------------------------------------------------
    vpc_sink: list[dict] = []
    zone_sink: list[int] = []
    multi_token_sink: list[dict] = []

    t0 = time.perf_counter()
    occurrences = list(analyze.analyze_segments(subset, vpc_sink, zone_sink, multi_token_sink))
    analyze_elapsed = time.perf_counter() - t0

    _write_jsonl(OUT_DIR / "phase3_vpc_candidates.jsonl", vpc_sink)

    idiomatch_raw = list(mwe.find_candidates(subset))
    vpc_path_for_benchmark = OUT_DIR / "phase3_vpc_candidates.jsonl"
    with patch.object(config, "VPC_CANDIDATES_PATH", vpc_path_for_benchmark):
        vpc_normalized = mwe.load_vpc_candidates(subset)
    merged = mwe.merge_candidate_sources(idiomatch_raw, vpc_normalized)
    mwe_filtered = mwe.structural_prefilter(merged)

    simple_word_candidates_raw = [
        occ for occ in occurrences if normalize_adapter.is_simple_word_candidate(occ)
    ]

    baseline2_candidates = normalize_adapter.combine(
        normalize_adapter.normalize_multi_token(multi_token_sink),
        normalize_adapter.normalize_mwe(mwe_filtered),
        normalize_adapter.normalize_simple_words(simple_word_candidates_raw),
    )
    report_baseline2 = scorer.score(
        baseline2_candidates,
        total_tokens=total_tokens,
        elapsed_seconds=analyze_elapsed,
        source="full_pipeline_ensemble_phase3run",
    )

    # ------------------------------------------------------------------
    # Second passage spaCy (même tokenizer que la production) — donne accès
    # aux Doc pour les scanners de rules_plus.py. analyze_segments() ne les
    # expose pas à l'appelant (générateur d'occurrences aplaties).
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    nlp = analyze.get_nlp()
    docs_by_segment = {
        seg.idx: doc for seg, doc in zip(subset, nlp.pipe((s.en for s in subset), batch_size=64))
    }
    rules_plus_new = rules_plus.build_new_candidates(subset, docs_by_segment, multi_token_sink)
    rules_plus_elapsed = time.perf_counter() - t0

    rules_plus_candidates = _dedupe(baseline2_candidates + rules_plus_new)
    report_rules_plus = scorer.score(
        rules_plus_candidates,
        total_tokens=total_tokens,
        elapsed_seconds=analyze_elapsed + rules_plus_elapsed,
        source="rules_plus",
    )
    report_new_only = scorer.score(
        _dedupe(rules_plus_new),
        total_tokens=total_tokens,
        elapsed_seconds=rules_plus_elapsed,
        source="rules_plus_new_generators_only",
    )

    _write_jsonl(OUT_DIR / "phase3_baseline2_candidates.jsonl", baseline2_candidates)
    _write_jsonl(OUT_DIR / "phase3_rules_plus_new_candidates.jsonl", rules_plus_new)
    _write_jsonl(OUT_DIR / "phase3_rules_plus_union_candidates.jsonl", rules_plus_candidates)
    (OUT_DIR / "phase3_baseline2_report.json").write_text(
        json.dumps(report_baseline2, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "phase3_rules_plus_report.json").write_text(
        json.dumps(report_rules_plus, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "phase3_new_only_report.json").write_text(
        json.dumps(report_new_only, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "baseline2": report_baseline2,
        "rules_plus": report_rules_plus,
        "new_generators_only": report_new_only,
    }


if __name__ == "__main__":
    reports = main()
    print(json.dumps(reports, indent=2, ensure_ascii=False))
