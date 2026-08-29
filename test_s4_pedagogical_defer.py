"""S4-3 — les signaux pédagogiques attendent le sens choisi par S5."""

from __future__ import annotations

import unittest
from unittest import mock

from pipeline import lexicon, select


def entry(lemma: str = "water", wn_pos: str = "n") -> dict:
    return {
        "lemma": lemma,
        "wn_pos": wn_pos,
        "occurrences": [{"upos": "NOUN"}],
    }


class SenseDependentGateTests(unittest.TestCase):
    def test_common_lemma_with_potential_rare_sense_survives_basic_cefr(self):
        prevalence = lexicon.PrevalenceRow("water", .999, 1000, 1.0, 5.5)
        with mock.patch.object(lexicon, "load_prevalence", return_value={"water": prevalence}), \
             mock.patch.object(lexicon, "cefr_levels_for", return_value={"A1"}):
            keep, meta = select.gate(entry())

        self.assertTrue(keep, "S4 must leave the rare-sense decision to S5")
        self.assertEqual(meta["cefr_signal"], "basic_only")
        self.assertTrue(meta["pedagogical_filter_deferred"])
        self.assertNotIn("drop_reason", meta)

    def test_low_prevalence_is_metadata_not_an_early_exclusion(self):
        prevalence = lexicon.PrevalenceRow("arcane", .1, 1000, .1, 2.0)
        with mock.patch.object(lexicon, "load_prevalence", return_value={"arcane": prevalence}), \
             mock.patch.object(lexicon, "cefr_levels_for", return_value=set()):
            keep, meta = select.gate(entry("arcane"))

        self.assertTrue(keep)
        self.assertEqual(meta["prevalence_signal"], "low_prevalence")
        self.assertTrue(meta["pedagogical_filter_deferred"])

    def test_certain_named_entity_noise_remains_excluded(self):
        proper = entry("inventedname")
        proper["occurrences"] = [{"upos": "PROPN"}]
        with mock.patch.object(lexicon, "load_prevalence", return_value={}), \
             mock.patch.object(lexicon, "cefr_levels_for", return_value=set()), \
             mock.patch.object(select, "has_common_synset", return_value=False):
            keep, meta = select.gate(proper)

        self.assertFalse(keep)
        self.assertEqual(meta["drop_reason"], "named_entity")


if __name__ == "__main__":
    unittest.main()
