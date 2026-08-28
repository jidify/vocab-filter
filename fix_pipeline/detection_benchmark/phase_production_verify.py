"""Q0-3 Phase 6 — vérification de l'intégration en production (voir le
plan d'implémentation, section "Vérification", point 2) : appelle le VRAI
code de production désormais modifié (`pipeline.analyze.analyze_segments`,
`pipeline.multi_token.detect` via lui, `pipeline.mwe.merge_candidate_sources`
à 3 sources) sur les 99 segments gold, et score avec le scorer de Phase 1 —
à comparer aux chiffres déjà mesurés par le prototype de benchmark
(`phase3_rules_plus_report.md` : 67,6% rappel MWE exact, 85,7% phrasal
verbs séparables) pour confirmer que le port en production (conversion de
schéma `member_spans` -> `member_char_spans` comprise) ne régresse rien.

Ne modifie AUCUN fichier de production. N'écrit que dans
`pipeline_out/detection_benchmark/` (gitignored, régénérable)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

from pipeline import analyze, config, mwe
from pipeline.corpus import load_segments

from fix_pipeline.detection_benchmark import normalize_adapter, scorer

OUT_DIR = config.ROOT / "pipeline_out" / "detection_benchmark"
_TOKEN_RE = re.compile(r"\w+")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_idxs = {g.segment_idx for g in scorer.load_gold()}
    all_segments = load_segments()
    play_segments = [s for s in all_segments if s.kind != "hors_oeuvre"]
    subset = [s for s in play_segments if s.idx in gold_idxs]
    assert len(subset) == len(gold_idxs)
    total_tokens = sum(len(_TOKEN_RE.findall(s.en)) for s in subset)

    vpc_sink: list[dict] = []
    zone_sink: list[int] = []
    multi_token_sink: list[dict] = []
    rules_plus_sink: list[dict] = []

    t0 = time.perf_counter()
    occurrences = list(analyze.analyze_segments(
        subset, vpc_sink, zone_sink, multi_token_sink, rules_plus_sink
    ))
    analyze_elapsed = time.perf_counter() - t0
    print(f"analyze_segments (production, tel que modifié) : {len(occurrences)} occurrences, "
          f"{len(multi_token_sink)} candidats multi_token, {len(vpc_sink)} candidats VPC bruts, "
          f"{len(rules_plus_sink)} candidats rules_plus, {analyze_elapsed:.1f}s.")

    vpc_path = OUT_DIR / "prodverify_vpc_candidates.jsonl"
    rules_plus_path = OUT_DIR / "prodverify_rules_plus_candidates.jsonl"
    _write_jsonl(vpc_path, vpc_sink)
    _write_jsonl(rules_plus_path, rules_plus_sink)

    idiomatch_raw = list(mwe.find_candidates(subset))
    with patch.object(config, "VPC_CANDIDATES_PATH", vpc_path):
        vpc_normalized = mwe.load_vpc_candidates(subset)
    with patch.object(config, "RULES_PLUS_CANDIDATES_PATH", rules_plus_path):
        rules_plus_normalized = mwe.load_rules_plus_candidates(subset)

    merged = mwe.merge_candidate_sources(idiomatch_raw, vpc_normalized, rules_plus_normalized)
    mwe_filtered = mwe.structural_prefilter(merged)
    print(f"idiomatch={len(idiomatch_raw)} vpc={len(vpc_normalized)} "
          f"rules_plus={len(rules_plus_normalized)} -> fusionné+filtré={len(mwe_filtered)}.")

    simple_word_candidates_raw = [
        occ for occ in occurrences if normalize_adapter.is_simple_word_candidate(occ)
    ]

    production_candidates = normalize_adapter.combine(
        normalize_adapter.normalize_multi_token(multi_token_sink),
        normalize_adapter.normalize_mwe(mwe_filtered),
        normalize_adapter.normalize_simple_words(simple_word_candidates_raw),
    )
    report = scorer.score(
        production_candidates,
        total_tokens=total_tokens,
        elapsed_seconds=analyze_elapsed,
        source="production_rules_plus_integrated",
    )

    _write_jsonl(OUT_DIR / "prodverify_candidates.jsonl", production_candidates)
    (OUT_DIR / "prodverify_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    report = main()
    print(json.dumps(report, indent=2, ensure_ascii=False))
