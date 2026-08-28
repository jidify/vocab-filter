"""Probe hors plan : rejoue Baseline 2 (Phase 2) et `rules_plus` (Phase 3)
à l'identique, mais avec `tokenizer_boundary_fix.patch_dash_after_punctuation`
appliqué au tokenizer spaCy AVANT toute analyse — pour chiffrer l'effet du
fix de tokenizer découvert en discussion (voir sa docstring), séparément de
tout ce que `rules_plus` fait déjà.

Ne modifie AUCUN fichier de production : le patch est appliqué en mémoire
sur le `nlp` retourné par `pipeline.analyze.get_nlp()`, pour la durée de ce
script seulement. Mêmes 99 segments gold que Phase 2/3, jamais le livre
entier pour le SCORING (le scan diagnostic sur les 2535 segments a déjà été
fait à la main, voir la conversation et la docstring de
`tokenizer_boundary_fix.py` — pas refait ici).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

from pipeline import analyze, config, mwe
from pipeline.corpus import load_segments

from fix_pipeline.detection_benchmark import (
    normalize_adapter,
    rules_plus,
    scorer,
    tokenizer_boundary_fix,
)

OUT_DIR = config.ROOT / "pipeline_out" / "detection_benchmark"
REPORT_PATH = Path(__file__).resolve().parent / "phase_tokenizer_fix_probe_report.md"

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

    # ------------------------------------------------------------------
    # Patch AVANT tout usage du tokenizer -- get_nlp() est un singleton mis
    # en cache par pipeline/analyze.py : le patcher une fois ici suffit,
    # tout appel ultérieur (y compris depuis analyze_segments/mwe.py) réutilise
    # le même objet nlp déjà patché.
    # ------------------------------------------------------------------
    nlp = analyze.get_nlp()
    tokenizer_boundary_fix.patch_dash_after_punctuation(nlp)

    gold_idxs = {g.segment_idx for g in scorer.load_gold()}
    all_segments = load_segments()
    play_segments = [s for s in all_segments if s.kind != "hors_oeuvre"]
    subset = [s for s in play_segments if s.idx in gold_idxs]
    assert len(subset) == len(gold_idxs)
    total_tokens = sum(len(_TOKEN_RE.findall(s.en)) for s in subset)

    # ------------------------------------------------------------------
    # Baseline 2, tokenizer patché (reprise identique de phase2/phase3).
    # ------------------------------------------------------------------
    vpc_sink: list[dict] = []
    zone_sink: list[int] = []
    multi_token_sink: list[dict] = []

    t0 = time.perf_counter()
    occurrences = list(analyze.analyze_segments(subset, vpc_sink, zone_sink, multi_token_sink))
    analyze_elapsed = time.perf_counter() - t0

    vpc_path_for_benchmark = OUT_DIR / "probe_vpc_candidates.jsonl"
    _write_jsonl(vpc_path_for_benchmark, vpc_sink)

    idiomatch_raw = list(mwe.find_candidates(subset))
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
        source="baseline2_tokenizer_fix_probe",
    )

    # ------------------------------------------------------------------
    # rules_plus, tokenizer patché -- second passage nlp.pipe (même nlp
    # patché, singleton) pour les scanners lemme/POS de rules_plus.py.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
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
        source="rules_plus_tokenizer_fix_probe",
    )

    _write_jsonl(OUT_DIR / "probe_baseline2_candidates.jsonl", baseline2_candidates)
    _write_jsonl(OUT_DIR / "probe_rules_plus_candidates.jsonl", rules_plus_candidates)
    (OUT_DIR / "probe_baseline2_report.json").write_text(
        json.dumps(report_baseline2, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "probe_rules_plus_report.json").write_text(
        json.dumps(report_rules_plus, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {"baseline2": report_baseline2, "rules_plus": report_rules_plus}


if __name__ == "__main__":
    reports = main()
    print(json.dumps(reports, indent=2, ensure_ascii=False))
