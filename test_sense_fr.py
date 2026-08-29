from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline import sense_fr


class CollectTargetsMweIdentityTests(unittest.TestCase):
    def test_mwe_target_carries_pos_through(self):
        """S6-1 : identité complète — le POS calculé par S3/S4
        (mwe_judge.py/select.py, ex. "VERB") se perdait entre
        score.build_mwe_units() et le target consommé par S6 (frontier/
        reassign ne recevaient jamais que pos=None pour toute MWE)."""
        fake_mwe_unit = {
            "canonical_form": "give out", "pos": "VERB", "sense_id": "phrasal_verb",
            "unit_key": "mwe:give out:phrasal_verb", "occurrences": 1,
            "definition_en": "To break down.", "label": "phrasal_verb",
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[]), \
             patch("pipeline.score.build_mwe_units", return_value=[fake_mwe_unit]):
            targets = sense_fr.collect_targets()

        self.assertEqual(targets["mwe:give out:phrasal_verb"]["pos"], "VERB")

    def test_falls_back_to_none_when_unit_has_no_pos(self):
        fake_mwe_unit = {
            "canonical_form": "wing it", "pos": None, "sense_id": "idiome",
            "unit_key": "mwe:wing it:idiome", "occurrences": 1,
            "definition_en": None, "label": "idiome",
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[]), \
             patch("pipeline.score.build_mwe_units", return_value=[fake_mwe_unit]):
            targets = sense_fr.collect_targets()

        self.assertIsNone(targets["mwe:wing it:idiome"]["pos"])


class ClassifyMweKeyPosTests(unittest.TestCase):
    def test_uses_target_pos_instead_of_hardcoded_none(self):
        target = {
            "key": "mwe:give out:phrasal_verb", "lemmas_en": ["give out"],
            "occurrences": 1, "definition_en": "To break down.", "pos": "VERB",
        }
        with patch("pipeline.sense_fr.llm_is_available", return_value=False):
            entry = sense_fr.classify_mwe_key(target)
        self.assertEqual(entry["pos"], "VERB")


if __name__ == "__main__":
    unittest.main()
