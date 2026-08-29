from __future__ import annotations

import unittest

from pipeline import senses


def candidates(scores, fr_winner=None):
    return [
        {"synset": f"sense.{i}", "final_score": score,
         "fr_score": 0.2 if i == fr_winner else 0.0}
        for i, score in enumerate(scores)
    ]


class CalibratedPolicyTests(unittest.TestCase):
    def test_margin_is_not_the_primary_gate(self):
        # Meme marge historique (> .15), mais absence de localisation : revision.
        decision = senses.calibrated_resolution_policy(
            candidates([0.9, 0.6]), located=False, pos_compatible=True,
            has_bilingual=True,
        )
        self.assertTrue(decision["needs_arbitration"])
        self.assertTrue(decision["needs_review"])
        self.assertIn("target_not_localized", decision["reasons"])

    def test_convergent_localized_bilingual_evidence_can_be_accepted(self):
        decision = senses.calibrated_resolution_policy(
            candidates([3.0, 0.0, -1.0], fr_winner=0), located=True,
            pos_compatible=True, has_bilingual=True,
        )
        self.assertFalse(decision["needs_arbitration"])
        self.assertFalse(decision["needs_review"])
        self.assertGreaterEqual(decision["confidence"], senses.POLICY_ACCEPT_CONFIDENCE)

    def test_pos_compound_and_entity_conflicts_force_review(self):
        for kwargs, reason in [
            ({"pos_compatible": False}, "pos_incompatible"),
            ({"structural_conflict": True}, "entity_or_compound_conflict"),
        ]:
            decision = senses.calibrated_resolution_policy(
                candidates([4.0, 0.0], fr_winner=0), located=True,
                has_bilingual=True, **kwargs,
            )
            self.assertTrue(decision["needs_review"])
            self.assertIn(reason, decision["reasons"])

    def test_entropy_and_model_disagreement_route_to_arbitration(self):
        decision = senses.calibrated_resolution_policy(
            candidates([0.51, 0.50, 0.49], fr_winner=1), located=True,
            pos_compatible=True, has_bilingual=True, model_disagreement=True,
        )
        self.assertTrue(decision["needs_arbitration"])
        self.assertIn("high_candidate_entropy", decision["reasons"])
        self.assertIn("model_disagreement", decision["reasons"])

    def test_no_bilingual_signal_is_explicit(self):
        decision = senses.calibrated_resolution_policy(
            candidates([1.0, 0.9]), located=True, pos_compatible=True,
            has_bilingual=False,
        )
        self.assertIn("bilingual_unavailable", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
