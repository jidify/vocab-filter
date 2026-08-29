from __future__ import annotations

import unittest

from pipeline.sense_fr_adjudicate import decide_stage_a, eligible_candidates


def _entry(**overrides) -> dict:
    base = dict(key="a", status="pending", fr="mot", sense_fit=None, translation_type=None)
    base.update(overrides)
    return base


def _signals(**overrides) -> dict:
    base = dict(
        n_corroborating_signals=2, deterministic_ok=True,
        resource_match=True, dbnary_match=True, apertium_match=False,
        wordfreq_ok=True, polysemy_collision_with=None,
    )
    base.update(overrides)
    return base


class EligibleCandidatesTests(unittest.TestCase):
    def test_excludes_mismatch_and_doubtful_pending_entries(self):
        store = {
            "a": _entry(key="a", sense_fit="mismatch"),
            "b": _entry(key="b", sense_fit="doubtful"),
            "c": _entry(key="c", sense_fit="ok"),
            "d": _entry(key="d", status="validated"),  # ni pending ni auto_llm : hors périmètre
        }
        candidates, excluded = eligible_candidates(store)
        self.assertEqual({e["key"] for e in candidates}, {"c"})
        self.assertEqual({e["key"] for e in excluded}, {"a", "b"})

    def test_excludes_non_literal_translation_type(self):
        store = {"a": _entry(key="a", translation_type="reformulation")}
        candidates, excluded = eligible_candidates(store)
        self.assertEqual(candidates, [])
        self.assertEqual([e["key"] for e in excluded], ["a"])

    def test_legacy_entries_without_sense_fit_field_remain_eligible(self):
        store = {"a": _entry(key="a", status="auto_llm", sense_fit=None, translation_type=None)}
        candidates, excluded = eligible_candidates(store)
        self.assertEqual([e["key"] for e in candidates], ["a"])
        self.assertEqual(excluded, [])


class DecideStageATests(unittest.TestCase):
    def test_mismatch_entry_is_never_promoted_even_with_two_corroborating_signals(self):
        """Reproduit le bug réel : sans cette porte, Stage A promouvait en
        auto_corroborated une entrée déjà signalée mismatch/doubtful par
        sense_fr_frontier.py dès que 2 ressources hors ligne corroboraient
        la MÊME traduction, verrouillant la contradiction plutôt que de la
        laisser en révision (voir pipeline/verify_sense_coherence.py)."""
        entry = _entry(key="a", status="pending", sense_fit="mismatch")
        decision = decide_stage_a(entry, _signals())
        self.assertIsNone(decision)

    def test_doubtful_entry_is_never_promoted(self):
        entry = _entry(key="a", status="auto_llm", sense_fit="doubtful")
        decision = decide_stage_a(entry, _signals())
        self.assertIsNone(decision)

    def test_coherent_entry_still_promotes_normally(self):
        entry = _entry(key="a", status="pending", sense_fit="ok", translation_type="equivalence_directe")
        decision = decide_stage_a(entry, _signals())
        self.assertIsNotNone(decision)
        self.assertEqual(decision[0], "auto_corroborated")


if __name__ == "__main__":
    unittest.main()
