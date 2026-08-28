"""Phase 2 — mesure les deux baselines demandées par le plan
(``plan_detection_benchmark_funnel.md``, section Phase 2) :

1. ``pipeline.multi_token`` seul ;
2. l'ensemble des détecteurs actuels du pipeline (spaCy NER/compound via
   multi_token, détecteur VPC, idiomatch, extraction des mots simples).

Ne lance JAMAIS le pipeline de production sur le livre entier : seuls les
99 segments couverts par le corpus gold (``fix_pipeline/gold_corpus/
the_humans_gold_v0.jsonl``) sont analysés, via les fonctions de production
elles-mêmes (``pipeline.analyze.analyze_segments``, ``pipeline.mwe.
find_candidates``/``merge_candidate_sources``/``structural_prefilter``,
``pipeline.multi_token.detect``) — jamais une réimplémentation. Le gold
corpus n'est lu que pour ses ``segment_idx`` (jamais copié dans un
magasin permanent, jamais utilisé pour guider un détecteur).

Artefacts bruts (candidats produits, occurrences) -> pipeline_out/
detection_benchmark/ (gitignored, régénérable). Rapport de scores ->
fix_pipeline/detection_benchmark/phase2_baselines_report.md (écrit par
ce script).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

from pipeline import analyze, config, mwe, multi_token as multi_token_module
from pipeline.corpus import load_segments

from fix_pipeline.detection_benchmark import normalize_adapter, scorer

OUT_DIR = config.ROOT / "pipeline_out" / "detection_benchmark"
REPORT_PATH = Path(__file__).resolve().parent / "phase2_baselines_report.md"

_TOKEN_RE = re.compile(r"\w+")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_gold_segment_idxs() -> set[int]:
    return {g.segment_idx for g in scorer.load_gold()}


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_idxs = load_gold_segment_idxs()
    all_segments = load_segments()
    play_segments = [s for s in all_segments if s.kind != "hors_oeuvre"]
    subset = [s for s in play_segments if s.idx in gold_idxs]
    print(f"{len(subset)} segments du livre correspondant aux {len(gold_idxs)} "
          f"segment_idx du corpus gold (attendu : 99).")
    assert len(subset) == len(gold_idxs), (
        "segment_idx manquants entre le corpus gold et pipeline.corpus.load_segments() "
        "— le corpus gold n'est plus alignable avec la production."
    )

    total_tokens = sum(len(_TOKEN_RE.findall(s.en)) for s in subset)

    # ------------------------------------------------------------------
    # Un seul run de la boucle spaCy de production (analyze_segments) sur
    # le sous-ensemble gold : produit occurrences + multi_token_candidates
    # + vpc_candidates en une passe, exactement comme pipeline/analyze.py::run().
    # ------------------------------------------------------------------
    vpc_sink: list[dict] = []
    zone_sink: list[int] = []
    multi_token_sink: list[dict] = []

    t0 = time.perf_counter()
    occurrences = list(analyze.analyze_segments(subset, vpc_sink, zone_sink, multi_token_sink))
    analyze_elapsed = time.perf_counter() - t0
    print(f"analyze_segments : {len(occurrences)} occurrences, "
          f"{len(multi_token_sink)} candidats multi_token, "
          f"{len(vpc_sink)} candidats VPC bruts en {analyze_elapsed:.1f}s.")

    _write_jsonl(OUT_DIR / "occurrences.jsonl", occurrences)
    _write_jsonl(OUT_DIR / "multi_token_candidates.jsonl", multi_token_sink)
    _write_jsonl(OUT_DIR / "vpc_candidates.jsonl", vpc_sink)

    # ------------------------------------------------------------------
    # idiomatch, sur le même sous-ensemble (pipeline.mwe.find_candidates
    # tourne segment par segment, comme en production).
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    idiomatch_raw = list(mwe.find_candidates(subset))
    idiomatch_elapsed = time.perf_counter() - t0
    print(f"idiomatch : {len(idiomatch_raw)} candidats bruts en {idiomatch_elapsed:.1f}s.")

    # mwe.load_vpc_candidates() lit VPC_CANDIDATES_PATH sur disque (c'est le
    # contrat de production réel, vpc_candidates.jsonl étant un artefact
    # intermédiaire entre analyze.py et mwe.py) — on la pointe vers notre
    # copie sous pipeline_out/detection_benchmark/, jamais vers le vrai
    # pipeline_out/vpc_candidates.jsonl d'un run complet.
    vpc_path_for_benchmark = OUT_DIR / "vpc_candidates.jsonl"
    with patch.object(config, "VPC_CANDIDATES_PATH", vpc_path_for_benchmark):
        vpc_normalized = mwe.load_vpc_candidates(subset)
    print(f"VPC (normalisé au schéma mwe, rejets exclus) : {len(vpc_normalized)} candidats.")

    merged = mwe.merge_candidate_sources(idiomatch_raw, vpc_normalized)
    mwe_filtered = mwe.structural_prefilter(merged)
    print(f"mwe fusionné+filtré (idiomatch ∪ VPC, structural_prefilter) : "
          f"{len(mwe_filtered)} candidats.")
    _write_jsonl(OUT_DIR / "mwe_filtered_candidates.jsonl", mwe_filtered)

    # ------------------------------------------------------------------
    # Mots simples : mêmes 4 conditions que select.py::iter_content_occurrences
    # (sans is_covered — voir normalize_adapter.py).
    # ------------------------------------------------------------------
    simple_word_candidates_raw = [
        occ for occ in occurrences if normalize_adapter.is_simple_word_candidate(occ)
    ]
    print(f"mots simples (filtre select.py sans is_covered) : "
          f"{len(simple_word_candidates_raw)} candidats.")

    # ------------------------------------------------------------------
    # Baseline 1 — pipeline.multi_token seul.
    # ------------------------------------------------------------------
    baseline1_candidates = normalize_adapter.normalize_multi_token(multi_token_sink)
    report1 = scorer.score(
        baseline1_candidates,
        total_tokens=total_tokens,
        elapsed_seconds=analyze_elapsed,
        source="multi_token_only",
    )

    # ------------------------------------------------------------------
    # Baseline 2 — ensemble complet actuel du pipeline.
    # ------------------------------------------------------------------
    baseline2_candidates = normalize_adapter.combine(
        normalize_adapter.normalize_multi_token(multi_token_sink),
        normalize_adapter.normalize_mwe(mwe_filtered),
        normalize_adapter.normalize_simple_words(simple_word_candidates_raw),
    )
    report2 = scorer.score(
        baseline2_candidates,
        total_tokens=total_tokens,
        elapsed_seconds=analyze_elapsed + idiomatch_elapsed,
        source="full_pipeline_ensemble",
    )

    _write_jsonl(OUT_DIR / "baseline1_multi_token_candidates.jsonl", baseline1_candidates)
    _write_jsonl(OUT_DIR / "baseline2_full_ensemble_candidates.jsonl", baseline2_candidates)
    (OUT_DIR / "baseline1_report.json").write_text(
        json.dumps(report1, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "baseline2_report.json").write_text(
        json.dumps(report2, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {"baseline1": report1, "baseline2": report2}


if __name__ == "__main__":
    reports = main()
    print(json.dumps(reports, indent=2, ensure_ascii=False))
