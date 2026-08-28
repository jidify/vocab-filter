"""Span-level scorer for the Q0-3 detection-architecture benchmark (Phase 1).

Scores a detector's candidate spans against the frozen gold corpus
(``fix_pipeline/gold_corpus/the_humans_gold_v0.jsonl``, gelé en v0 en Phase 0).
This module never imports anything under :mod:`pipeline` and never runs a
detector itself — Phase 2+ is responsible for producing the candidate list
this scorer consumes.

Candidate input format (one dict per span a detector produced)::

    {
      "segment_idx": 75,
      "surface": "New York City",
      "start_char": 73,
      "end_char": 86,
      "category": "multi_token_entity",   # optional, informational only
      "source": "spacy",                  # optional, informational only
    }

For phrasal verbs with discontinuous members (separable phrasal verbs), a
candidate may instead (or additionally) provide::

    {
      "segment_idx": 197,
      "surface": "cracks the bathroom door open",
      "category": "phrasal_verb_separable",
      "source": "rules_plus",
      "full_span": {"start_char": 7, "end_char": 36},
      "member_spans": [{"start_char": 7, "end_char": 13},
                        {"start_char": 31, "end_char": 36}]
    }

A candidate must supply at least one of: (``start_char`` and ``end_char``),
``full_span``, or ``member_spans``. When only ``member_spans`` is given, the
resolved span is synthesized as ``(min(member starts), max(member ends))`` —
this is how a detector that only locates the verb and the particle
separately can still be matched against the gold corpus's single contiguous
annotation of the whole construction, without the scorer inventing offsets
for the material in between.

Metrics produced (see plan_detection_benchmark_funnel.md, Phase 1):

- rappel exact des spans (``exact_recall``) ;
- rappel avec chevauchement (``overlap_recall``) — diagnostic uniquement,
  jamais assimilé à une réussite : un span qui chevauche le gold sans en
  partager les bornes est précisément le type d'erreur que ce benchmark
  cherche à repérer (ex. ``floor apartment`` vs ``ground-floor apartment``) ;
- exactitude des bornes (``boundary_accuracy``) = exact_recall / overlap_recall
  parmi les spans effectivement chevauchés ;
- rappel par catégorie et par rôle (mêmes trois métriques, regroupées) ;
- taux de capture des ``hard_negative`` (correspondance EXACTE avec le piège —
  un chevauchement seul ne suffit pas, sinon le vrai positif voisin
  déclencherait le piège par accident) ;
- candidats produits pour 1000 tokens (proxy de sur-génération) ;
- temps d'exécution du détecteur, transmis tel quel par l'appelant.

Délibérément absent : une métrique de « précision exacte » globale. Le
corpus (109 spans sur 2535 segments) n'est pas exhaustif : un candidat hors
gold n'est pas nécessairement une erreur. Le seul signal de précision
interprétable ici est le taux de capture des ``hard_negative``, conçus
explicitement comme des pièges.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_GOLD_PATH = (
    Path(__file__).resolve().parent.parent / "gold_corpus" / "the_humans_gold_v0.jsonl"
)

POSITIVE_CATEGORIES = (
    "simple_word",
    "nominal_compound",
    "multi_token_entity",
    "phrasal_verb_separable",
    "phrasal_verb_inseparable",
    "idiom",
)
ROLES = ("lexical_candidate", "protective_span", "pedagogical_word")

_TOKEN_RE = re.compile(r"\w+")


# --------------------------------------------------------------------------
# Gold corpus loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldSpan:
    segment_idx: int
    surface: str
    start_char: int
    end_char: int
    category: str
    is_gold: bool
    role: str | None = None


@dataclass(frozen=True)
class GoldSegment:
    segment_idx: int
    text: str
    spans: tuple[GoldSpan, ...]


def load_gold(path: Path | str = DEFAULT_GOLD_PATH) -> list[GoldSegment]:
    """Load the frozen gold corpus. Read-only: never writes back to it."""
    path = Path(path)
    segments: list[GoldSegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        spans = tuple(
            GoldSpan(
                segment_idx=rec["segment_idx"],
                surface=s["surface"],
                start_char=s["start_char"],
                end_char=s["end_char"],
                category=s["category"],
                is_gold=s["is_gold"],
                role=s.get("role"),
            )
            for s in rec["gold_spans"]
        )
        segments.append(GoldSegment(segment_idx=rec["segment_idx"], text=rec["text"], spans=spans))
    return segments


# --------------------------------------------------------------------------
# Candidate span resolution
# --------------------------------------------------------------------------


class CandidateError(ValueError):
    """Raised when a candidate dict cannot be resolved to a (start, end) span."""


def resolve_span(candidate: dict[str, Any]) -> tuple[int, int]:
    """Return ``(start_char, end_char)`` for a candidate, per the formats
    documented in the module docstring. Priority when several are present:
    ``full_span`` > flat ``start_char``/``end_char`` > synthesized from
    ``member_spans``.
    """
    full_span = candidate.get("full_span")
    if full_span is not None:
        return full_span["start_char"], full_span["end_char"]
    if "start_char" in candidate and "end_char" in candidate:
        return candidate["start_char"], candidate["end_char"]
    member_spans = candidate.get("member_spans")
    if member_spans:
        starts = [m["start_char"] for m in member_spans]
        ends = [m["end_char"] for m in member_spans]
        return min(starts), max(ends)
    raise CandidateError(
        "candidate has none of start_char/end_char, full_span, member_spans: "
        f"{candidate!r}"
    )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# --------------------------------------------------------------------------
# Recall bookkeeping
# --------------------------------------------------------------------------


@dataclass
class RecallStats:
    n_gold: int = 0
    n_exact: int = 0
    n_overlap: int = 0

    def record(self, exact: bool, overlap: bool) -> None:
        self.n_gold += 1
        self.n_exact += int(exact)
        self.n_overlap += int(overlap)

    @property
    def exact_recall(self) -> float | None:
        return self.n_exact / self.n_gold if self.n_gold else None

    @property
    def overlap_recall(self) -> float | None:
        return self.n_overlap / self.n_gold if self.n_gold else None

    @property
    def boundary_accuracy(self) -> float | None:
        """Among gold spans that were at least overlapped, the share found exactly."""
        return self.n_exact / self.n_overlap if self.n_overlap else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_gold": self.n_gold,
            "exact_recall": self.exact_recall,
            "overlap_recall": self.overlap_recall,
            "boundary_accuracy": self.boundary_accuracy,
        }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(
    candidates: Iterable[dict[str, Any]],
    *,
    gold_path: Path | str = DEFAULT_GOLD_PATH,
    total_tokens: int | None = None,
    elapsed_seconds: float | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Score ``candidates`` (as produced by one detector) against the gold corpus.

    ``total_tokens``: token count of the corpus the detector actually ran
    over. If omitted, falls back to counting tokens across the gold
    segments' own text only — a likely undercount when the detector was run
    over the full play (2535 segments) rather than just the 99 annotated
    ones; pass it explicitly in that case.

    ``elapsed_seconds``: wall-clock time the detector took to produce
    ``candidates``, measured by the caller. This function does not measure
    detector execution time itself — it never runs a detector.
    """
    gold_segments = load_gold(gold_path)

    candidates = list(candidates)
    resolved_by_segment: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for c in candidates:
        resolved_by_segment[c["segment_idx"]].append(resolve_span(c))

    overall = RecallStats()
    by_category: dict[str, RecallStats] = {cat: RecallStats() for cat in POSITIVE_CATEGORIES}
    by_role: dict[str, RecallStats] = {role: RecallStats() for role in ROLES}
    hard_negative_total = 0
    hard_negative_captured = 0

    for gseg in gold_segments:
        cand_spans = resolved_by_segment.get(gseg.segment_idx, [])
        for gspan in gseg.spans:
            gold_range = (gspan.start_char, gspan.end_char)
            if gspan.is_gold:
                exact = gold_range in cand_spans
                overlap = any(_overlaps(gold_range, c) for c in cand_spans)
                overall.record(exact, overlap)
                by_category[gspan.category].record(exact, overlap)
                if gspan.role is not None:
                    by_role[gspan.role].record(exact, overlap)
            else:
                hard_negative_total += 1
                if gold_range in cand_spans:
                    hard_negative_captured += 1

    if total_tokens is None:
        total_tokens = sum(len(_TOKEN_RE.findall(g.text)) for g in gold_segments)
        total_tokens_source = (
            "gold_segments_only (fallback — pass total_tokens explicitly "
            "when scoring a run over the full corpus)"
        )
    else:
        total_tokens_source = "provided"

    candidates_per_1000_tokens = (
        1000 * len(candidates) / total_tokens if total_tokens else None
    )

    return {
        "source": source,
        "n_candidates": len(candidates),
        "overall": overall.as_dict(),
        "by_category": {cat: stats.as_dict() for cat, stats in by_category.items()},
        "by_role": {role: stats.as_dict() for role, stats in by_role.items()},
        "hard_negatives": {
            "n_total": hard_negative_total,
            "n_captured": hard_negative_captured,
            "capture_rate": (
                hard_negative_captured / hard_negative_total if hard_negative_total else None
            ),
        },
        "generation_rate": {
            "candidates_per_1000_tokens": candidates_per_1000_tokens,
            "total_tokens": total_tokens,
            "total_tokens_source": total_tokens_source,
        },
        "execution_time_seconds": elapsed_seconds,
    }


# --------------------------------------------------------------------------
# Self-test: score the gold corpus against itself (must give 100% everywhere
# a "found" metric applies, and 0% hard_negative capture).
# --------------------------------------------------------------------------


def _build_selftest_candidates(gold_segments: list[GoldSegment]) -> list[dict[str, Any]]:
    """A 'perfect' detector: emit exactly the positive gold spans, as flat
    candidates, and nothing else (no hard_negative should ever be produced by
    a perfect detector).
    """
    candidates = []
    for gseg in gold_segments:
        for gspan in gseg.spans:
            if not gspan.is_gold:
                continue
            candidates.append(
                {
                    "segment_idx": gseg.segment_idx,
                    "surface": gspan.surface,
                    "start_char": gspan.start_char,
                    "end_char": gspan.end_char,
                    "category": gspan.category,
                    "source": "gold_selftest",
                }
            )
    return candidates


def _selftest_member_span_synthesis() -> None:
    """Unit check, independent of the real corpus: a candidate given only via
    discontinuous member_spans (no start_char/end_char/full_span) must
    resolve to the span enclosing both members — this is the format
    phrasal_verb_separable candidates are allowed to use.
    """
    candidate = {
        "segment_idx": 197,
        "surface": "cracks the bathroom door open",
        "category": "phrasal_verb_separable",
        "source": "unit_test",
        "member_spans": [
            {"start_char": 7, "end_char": 13},  # "cracks"
            {"start_char": 31, "end_char": 36},  # "open"
        ],
    }
    resolved = resolve_span(candidate)
    expected = (7, 36)
    if resolved != expected:
        raise AssertionError(f"member_spans synthesis failed: {resolved} != {expected}")


def _run_selftest(gold_path: Path | str) -> dict[str, Any]:
    _selftest_member_span_synthesis()

    gold_segments = load_gold(gold_path)

    build_start = time.perf_counter()
    candidates = _build_selftest_candidates(gold_segments)
    elapsed = time.perf_counter() - build_start

    report = score(candidates, gold_path=gold_path, elapsed_seconds=elapsed, source="gold_selftest")

    failures: list[str] = []
    overall = report["overall"]
    for metric in ("exact_recall", "overlap_recall", "boundary_accuracy"):
        if overall[metric] != 1.0:
            failures.append(f"overall {metric} = {overall[metric]}, expected 1.0")
    for cat, stats in report["by_category"].items():
        if stats["n_gold"] and stats["exact_recall"] != 1.0:
            failures.append(f"category {cat}: exact_recall = {stats['exact_recall']}, expected 1.0")
    for role, stats in report["by_role"].items():
        if stats["n_gold"] and stats["exact_recall"] != 1.0:
            failures.append(f"role {role}: exact_recall = {stats['exact_recall']}, expected 1.0")
    hn = report["hard_negatives"]
    if hn["capture_rate"] != 0.0:
        failures.append(
            f"hard_negative capture_rate = {hn['capture_rate']}, expected 0.0 "
            "(a perfect detector must not fall into the traps)"
        )

    report["_selftest_failures"] = failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the gold corpus against itself (Phase 1 sanity check).")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH), help="Path to the gold corpus JSONL.")
    args = parser.parse_args()

    report = _run_selftest(args.gold)
    failures = report.pop("_selftest_failures")

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if failures:
        print("\nSELF-TEST FAILED:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)

    print(
        "\nSELF-TEST OK — gold corpus scored against itself: 100% recall "
        "(exact & overlap) and 100% boundary_accuracy on every category and "
        "role, 0% hard_negative capture."
    )


if __name__ == "__main__":
    main()
