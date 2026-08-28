from __future__ import annotations

import json
import io
import unittest
from unittest.mock import patch

from pipeline import analyze, config, inventory, multi_token, select


class MultiTokenCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = analyze.get_nlp()

    def test_named_entities_and_nominal_compounds_are_proposed(self):
        text = ("I moved from New York and prayed to the Virgin Mary. "
                "Serve ranch dip by the observation deck of the nursing home "
                "and gaze into a crystal ball.")
        rows = multi_token.detect(self.nlp(text), 7)
        by_surface = {row["surface"]: row for row in rows}
        expected = {"New York", "Virgin Mary", "ranch dip", "observation deck",
                    "nursing home", "crystal ball"}
        self.assertTrue(expected <= by_surface.keys(), by_surface.keys())
        for surface in expected:
            row = by_surface[surface]
            self.assertEqual(text[row["start_char"]:row["end_char"]], surface)
            self.assertGreater(row["score"], 0)
            self.assertTrue(row["provenance"])
            self.assertEqual(row["schema_version"], multi_token.SCHEMA_VERSION)
        multi_token.validate(rows, {7: text})

    def test_plain_adjective_and_separate_names_are_not_joined(self):
        text = "We made a new plan and Mary observed crystal."
        self.assertEqual(multi_token.detect(self.nlp(text), 2), [])

    def test_hypothesis_does_not_reserve_tokens(self):
        text = "Try ranch dip."
        row = multi_token.detect(self.nlp(text), 3)[0]
        ranch = {"segment_idx": 3, "start_char": row["start_char"],
                 "end_char": row["start_char"] + len("ranch")}
        self.assertEqual(multi_token.covering(ranch, [row]), [row])
        # S4 réserve uniquement mwe_confirmed_spans, jamais ces hypothèses S1.
        self.assertFalse(select.is_covered(ranch, []))

    def test_artifact_and_s4_consumer_preserve_hypotheses(self):
        candidate = multi_token.detect(self.nlp("ranch dip"), 1)[0]
        occurrence = {
            "occurrence_id": "w:1:0", "segment_idx": 1, "kind": "dialogue",
            "surface": "ranch", "lemma": "ranch", "upos": "NOUN", "wn_pos": "n",
            "is_alpha": True, "is_stop": False, "start_char": 0, "end_char": 5,
            "multi_token_candidates": [candidate],
        }
        class JsonlPath:
            def __init__(self, row):
                self.payload = json.dumps(row) + "\n"
            def exists(self):
                return True
            def open(self, **_kwargs):
                return io.StringIO(self.payload)

        with patch.object(config, "MULTI_TOKEN_CANDIDATES_PATH", JsonlPath(candidate)):
            self.assertEqual(multi_token.load_by_segment()[1][0]["surface"], "ranch dip")
        with patch.object(config, "OCCURRENCES_PATH", JsonlPath(occurrence)):
            built = select.build_types({})[("ranch", "n")]
            self.assertEqual(built["occurrences"][0]["multi_token_candidates"][0]
                             ["candidate_id"], candidate["candidate_id"])

    def test_candidate_change_invalidates_downstream_digest(self):
        base = {"occurrence_id": "w:1:0", "unit_key": "ranch:n", "analysis": {},
                "multi_token_candidates": [{"candidate_id": "mt:1:0:9", "score": .82}]}
        changed = json.loads(json.dumps(base))
        changed["multi_token_candidates"][0]["score"] = .95
        self.assertNotEqual(inventory.compute_hash([base]), inventory.compute_hash([changed]))


class RulesPlusGroupATests(unittest.TestCase):
    """Q0-3 Phase 6 (fix_pipeline/detection_benchmark/phase6_decision.md) :
    les trois règles de bornes structurelles de pipeline/rules_plus.py
    (chaîne à trait d'union libre, extension à travers un trait d'union,
    troncature du possessif), consommées par multi_token.detect() — mêmes
    cas que ceux mesurés dans phase3_rules_plus_report.md."""

    @classmethod
    def setUpClass(cls):
        cls.nlp = analyze.get_nlp()

    def test_free_hyphen_chain_with_no_base_candidate(self):
        text = "A turn-of-the-century apartment."
        rows = multi_token.detect(self.nlp(text), 1)
        by_surface = {row["surface"]: row for row in rows}
        self.assertIn("turn-of-the-century", by_surface)
        multi_token.validate(rows, {1: text})

    def test_hyphen_extend_recovers_full_compound(self):
        text = "ground-floor apartment—"
        rows = multi_token.detect(self.nlp(text), 2)
        by_surface = {row["surface"]: row for row in rows}
        self.assertIn("ground-floor apartment", by_surface)
        multi_token.validate(rows, {2: text})

    def test_possessive_trim_adds_a_corrected_candidate_alongside_the_original(self):
        """rules_plus (Groupe A) n'a jamais de pouvoir de rejet : le
        candidat NER d'origine ("New York City's", possessif inclus) reste
        présent — c'est un candidat CORRIGÉ qui s'ajoute, rien n'est
        retiré (même principe que le reste de multi_token.py :
        "hypothèses auditables, jamais des réservations")."""
        text = "New York City’s Chinatown."
        rows = multi_token.detect(self.nlp(text), 3)
        by_surface = {row["surface"]: row for row in rows}
        self.assertIn("New York City", by_surface)
        self.assertEqual(by_surface["New York City"]["candidate_types"], ["multi_token_entity"])
        multi_token.validate(rows, {3: text})


if __name__ == "__main__":
    unittest.main()
