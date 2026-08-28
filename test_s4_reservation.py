"""S4-2 — réservation exacte, occurrence-scoped et comptabilisée."""

from __future__ import annotations

import unittest

from pipeline import mwe_judge, select


def word(occurrence_id: str, start: int, end: int, segment_idx: int = 1) -> dict:
    return {"occurrence_id": occurrence_id, "start_char": start, "end_char": end,
            "segment_idx": segment_idx}


def occurrence(
    occurrence_id: str, surface: str, members: list[list[int]], label: str,
    *, segment_idx: int = 1, confidence: float = .9,
) -> dict:
    return {
        "occurrence_id": occurrence_id,
        "segment_idx": segment_idx,
        "start_char": min(a for a, _ in members),
        "end_char": max(b for _, b in members),
        "surface": surface,
        "n_tokens_span": len(members),
        "member_char_spans": members,
        "ambiguous_alignment": False,
        "occurrence_decision": {
            "label": label,
            "confidence": confidence,
            "canonical_form": surface.casefold(),
            "pos": "PROPN" if surface in {"New York", "Virgin Mary"} else "VERB",
            "contextual_paraphrase": surface,
            "sense_id": f"fixture:{surface.casefold().replace(' ', '-')}",
        },
    }


def entry(idiom: str, *occurrences: dict) -> dict:
    return {"idiom": idiom, "label": "incertain", "confidence": 0.0,
            "occurrences": list(occurrences)}


class ExactCoverageTests(unittest.TestCase):
    def test_discontinuous_span_reserves_members_not_interior(self):
        span = {"start_char": 0, "end_char": 22,
                "member_char_spans": [[0, 6], [19, 22]]}
        self.assertTrue(select.is_covered(word("turned", 0, 6), [span]))
        self.assertFalse(select.is_covered(word("lantern", 11, 18), [span]))
        self.assertTrue(select.is_covered(word("off", 19, 22), [span]))

    def test_legacy_envelope_without_exact_members_reserves_nothing(self):
        self.assertFalse(select.is_covered(
            word("latch", 10, 15), [{"start_char": 0, "end_char": 20}]
        ))

    def test_rejected_lift_your_latch_returns_latch_token(self):
        rejected = occurrence("m:latch", "lift your latch", [[0, 4], [10, 15]], "littéral")
        spans = mwe_judge.select_mwe_spans([entry("lift one's latch", rejected)])
        self.assertEqual(spans.get(1, []), [])
        self.assertFalse(select.is_covered(word("w:latch", 10, 15), spans.get(1, [])))


class DecisionAndCompetitionTests(unittest.TestCase):
    def test_uncertain_hypothesis_reserves_nothing(self):
        uncertain = occurrence("m:uncertain", "look up", [[0, 4], [5, 7]], "incertain")
        self.assertEqual(mwe_judge.select_mwe_spans([entry("look up", uncertain)]), {})

    def test_richer_confirmed_competing_hypothesis_wins(self):
        short = occurrence("m:short", "put up", [[0, 3], [4, 6]], "phrasal_verb")
        rich = occurrence("m:rich", "put up with", [[0, 3], [4, 6], [7, 11]], "phrasal_verb")
        spans = mwe_judge.select_mwe_spans([entry("put up", short), entry("put up with", rich)])
        self.assertEqual([s["occurrence_id"] for s in spans[1]], ["m:rich"])

    def test_confirmed_compounds_reserve_components_only_in_their_occurrence(self):
        new_york = occurrence("m:ny", "New York", [[0, 3], [4, 8]], "semi_fige", segment_idx=1)
        virgin_mary = occurrence("m:vm", "Virgin Mary", [[0, 6], [7, 11]], "semi_fige", segment_idx=2)
        spans = mwe_judge.select_mwe_spans([
            entry("New York", new_york), entry("Virgin Mary", virgin_mary)
        ])
        self.assertTrue(select.is_covered(word("w:york-compound", 4, 8), spans[1]))
        self.assertTrue(select.is_covered(word("w:virgin-compound", 0, 6), spans[2]))
        self.assertFalse(select.is_covered(word("w:york-autonomous", 4, 8), spans.get(3, [])))
        self.assertFalse(select.is_covered(word("w:virgin-autonomous", 0, 6), spans.get(3, [])))


class ReservationAccountingTests(unittest.TestCase):
    def test_before_equals_reserved_plus_after(self):
        before = [word("w:turn", 0, 4), word("w:radio", 5, 10), word("w:off", 11, 14)]
        after = [before[1]]
        spans = {1: [{"occurrence_id": "m:turn-off",
                      "start_char": 0, "end_char": 14,
                      "member_char_spans": [[0, 4], [11, 14]]}]}
        report = select.build_reservation_report(before, after, spans)
        self.assertEqual(report["word_occurrences_before_reservation"], 3)
        self.assertEqual(report["word_occurrences_reserved"], 2)
        self.assertEqual(report["word_occurrences_after_reservation"], 1)
        self.assertEqual(report["legacy_envelope_would_reserve"], 3)
        self.assertEqual(report["tokens_returned_vs_legacy_envelope"], 1)
        self.assertTrue(report["invariant_before_equals_reserved_plus_after"])


if __name__ == "__main__":
    unittest.main()
