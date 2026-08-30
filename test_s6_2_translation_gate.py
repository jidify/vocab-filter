"""Correction S6-2 (plan §6) : une ligne de vocab.csv dont la traduction
officielle est encore `pending`/`rejected`/`no_equivalent`
(`meaning_fr_official` vide, voir score.resolve_official_fr) ne doit
jamais paraître finalisée. Avant ce correctif, `needs_review` ignorait
totalement `meaning_fr_official` : une occurrence par ailleurs pleinement
confiante (S5) produisait une ligne vide dans vocab.csv avec
`needs_review=False`, absente de review_queue.csv (write_review_queue ne
filtre que sur `needs_review`) — cas mesuré : 131 lignes le 2026-08-30,
voir fix_pipeline/plan_action_fix_pipeline.md §6 S6-2."""

from __future__ import annotations

import unittest
from unittest import mock

from pipeline import score


def _word_record(**overrides):
    base = dict(
        key=("dog", "n", "dog.n.01"), surface="dog", lemma="dog", wn_pos="n",
        sense_id="dog.n.01", definition="a domesticated animal", fr_hits=[],
        zipf=3.5, pknown=None, cefr_levels=[], segment_idx=0, needs_review=False,
        margin=1.0, context="The dog barked.", candidate_senses=["dog.n.01"],
        recovery_route=None, recovery_reason=None, review_action=None,
    )
    base.update(overrides)
    return base


class WordUnitNeedsReviewTests(unittest.TestCase):
    def _units(self, store: dict, records: list[dict]) -> list[dict]:
        with mock.patch.object(score.sense_fr, "load_store", return_value=store), \
             mock.patch.object(score.senses, "load_occurrences_by_sense", return_value={}), \
             mock.patch.object(score, "load_manual_corrections", return_value={}):
            return score.aggregate_and_score(records)

    def test_pending_translation_forces_needs_review_even_with_confident_occurrence(self):
        store = {"dog.n.01": {"status": "pending", "fr": "chien", "fr_alt": []}}
        units = self._units(store, [_word_record(needs_review=False)])
        self.assertEqual(len(units), 1)
        self.assertIsNone(units[0]["meaning_fr_official"])
        self.assertTrue(units[0]["needs_review"])

    def test_locked_translation_leaves_confident_occurrence_unflagged(self):
        store = {"dog.n.01": {"status": "auto_strong", "fr": "chien", "fr_alt": []}}
        units = self._units(store, [_word_record(needs_review=False)])
        self.assertEqual(units[0]["meaning_fr_official"], "chien")
        self.assertFalse(units[0]["needs_review"])

    def test_no_equivalent_status_also_forces_needs_review(self):
        store = {"dog.n.01": {"status": "no_equivalent", "fr": "chien", "fr_alt": []}}
        units = self._units(store, [_word_record(needs_review=False)])
        self.assertIsNone(units[0]["meaning_fr_official"])
        self.assertTrue(units[0]["needs_review"])


class MweUnitNeedsReviewTests(unittest.TestCase):
    def test_pending_translation_forces_needs_review_for_mwe(self):
        raw = [{
            "canonical_form": "kick the bucket", "surface_forms": ["kicked the bucket"],
            "pos": "v", "sense_id": "custom.mwe.idiom.kick_the_bucket.01",
            "unit_key": "mwe:kick the bucket:idiom:custom.mwe.idiom.kick_the_bucket.01",
            "definition_en": "to die", "label": "idiome", "book_count": 1, "dispersion": 1,
            "confidence": 0.95,
        }]
        store = {raw[0]["unit_key"]: {"status": "pending", "fr": "casser sa pipe", "fr_alt": []}}
        fake_path = mock.Mock()
        fake_path.exists.return_value = True
        fake_path.open.return_value = _JsonlLines(raw)
        with mock.patch.object(score.config, "SELECTED_MWE_PATH", fake_path), \
             mock.patch.object(score.sense_fr, "load_store", return_value=store), \
             mock.patch.object(score.senses, "load_occurrences_by_sense", return_value={}):
            units = score.build_mwe_units()
        self.assertEqual(len(units), 1)
        self.assertIsNone(units[0]["meaning_fr_official"])
        self.assertTrue(units[0]["needs_review"])


class _JsonlLines:
    """Contexte fichier minimal : itère les dicts déjà prêts comme des
    lignes JSON (évite de dépendre de json.dumps/round-trip pour ce test)."""

    def __init__(self, rows: list[dict]):
        import json
        self._lines = [json.dumps(r) for r in rows]

    def __enter__(self):
        return self._lines

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    unittest.main()
