from __future__ import annotations

import unittest

from pipeline.sense_fr_adjudicate import decide_stage_a, decide_stage_c, eligible_candidates


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

    def test_excludes_structurally_unresolved_entries_without_any_fr(self):
        """S6-2 (plan §6) : `fr is None` (sense_id WordNet introuvable, ou
        bifurcation S5-3 `aucun_sens_adapte` jamais rattachée — clés
        `unresolved.human_review.<hash>`) doit être exclu AVANT même
        sense_fit/translation_type : il n'y a aucune proposition ni
        définition à corroborer ou juger, contrairement à un `pending` de
        traduction normal."""
        store = {
            "a": _entry(key="a", fr=None, sense_fit=None, translation_type=None),
            "b": _entry(key="b", fr="mot"),
        }
        candidates, excluded = eligible_candidates(store)
        self.assertEqual([e["key"] for e in candidates], ["b"])
        self.assertEqual([e["key"] for e in excluded], ["a"])


class DecideStageCTests(unittest.TestCase):
    """S6-2 : `no_equivalent` doit obéir à la MÊME porte de confiance que
    `auto_judged` — un verdict de juge à confiance basse/moyenne, qu'il
    propose une traduction ou déclare qu'aucune n'existe, doit rester
    `pending` (révisable via sense_fr_review.csv), jamais verrouillé."""

    def _verdict(self, **overrides):
        base = dict(no_equivalent=False, confidence="high")
        base.update(overrides)
        return base

    def test_no_equivalent_high_confidence_locks_terminal(self):
        status, reason = decide_stage_c(self._verdict(no_equivalent=True, confidence="high"))
        self.assertEqual(status, "no_equivalent")
        self.assertEqual(reason, "stage_c:no_equivalent")

    def test_no_equivalent_medium_confidence_stays_pending(self):
        status, _ = decide_stage_c(self._verdict(no_equivalent=True, confidence="medium"))
        self.assertEqual(status, "pending")

    def test_no_equivalent_low_confidence_stays_pending(self):
        status, _ = decide_stage_c(self._verdict(no_equivalent=True, confidence="low"))
        self.assertEqual(status, "pending")

    def test_translation_high_confidence_promotes_to_auto_judged(self):
        status, reason = decide_stage_c(self._verdict(no_equivalent=False, confidence="high"))
        self.assertEqual(status, "auto_judged")
        self.assertEqual(reason, "stage_c:judge_high_confidence")

    def test_translation_low_confidence_stays_pending(self):
        status, _ = decide_stage_c(self._verdict(no_equivalent=False, confidence="low"))
        self.assertEqual(status, "pending")


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
