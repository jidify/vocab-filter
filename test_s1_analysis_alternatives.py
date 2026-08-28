from __future__ import annotations

import copy
import unittest

from pipeline import analyze, inventory
from pipeline.corpus import Segment


class Token:
    def __init__(self, text, lemma, pos, tag):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.tag_ = tag


class MorphosyntacticAlternativesTests(unittest.TestCase):
    def analyses(self, text, lemma, pos, tag):
        result = analyze.morphosyntactic_analysis(Token(text, lemma, pos, tag))
        return {(a["lemma"], a["wn_pos"]) for a in result["alternatives"]}, result

    def test_named_regression_cases_keep_required_inventory_openings(self):
        cases = [
            ("frosting", "frost", "VERB", "VBG", ("frosting", "n")),
            ("creeping", "creep", "VERB", "VBG", ("creeping", "a")),
            ("facilities", "facility", "NOUN", "NNS", ("facilities", "n")),
            ("stressing", "stress", "VERB", "VBG", ("destress", "v")),
            ("bitch", "bitch", "VERB", "VB", ("bitch", "n")),
        ]
        missing = []
        for surface, lemma, upos, tag, expected in cases:
            alternatives, result = self.analyses(surface, lemma, upos, tag)
            if expected not in alternatives:
                missing.append((surface, expected, sorted(alternatives)))
            self.assertEqual(result["primary"]["lemma"], lemma)
            self.assertEqual(result["version"], analyze.ANALYSIS_VERSION)
        self.assertFalse(missing, missing)

    def test_morphology_indicators_are_explicit(self):
        _, result = self.analyses("frosting", "frost", "VERB", "VBG")
        self.assertTrue(result["morphology"]["is_inflected"])
        self.assertTrue(result["morphology"]["is_participle"])
        self.assertTrue(result["morphology"]["is_nominalization_candidate"])

    def test_schema_validator_checks_exact_offsets_and_primary_projection(self):
        analysis = analyze.morphosyntactic_analysis(Token("frosting", "frost", "VERB", "VBG"))
        row = {
            "occurrence_id": "w:1:0", "segment_idx": 1, "surface": "frosting",
            "start_char": 5, "end_char": 13, "lemma": "frost", "upos": "VERB",
            "wn_pos": "v", "analysis_version": analyze.ANALYSIS_VERSION,
            "analysis": analysis,
        }
        segment = Segment(idx=1, kind="dialogue", speaker=None,
                          en="with frosting", fr="")
        analyze.validate_occurrence_schema([row], [segment])
        broken = copy.deepcopy(row)
        broken["start_char"] = 4
        with self.assertRaisesRegex(ValueError, "offsets invalides"):
            analyze.validate_occurrence_schema([broken], [segment])


class AnalysisDigestMigrationTests(unittest.TestCase):
    def test_legacy_rows_keep_the_historical_digest_contract(self):
        rows = [{"occurrence_id": "w:1:0", "unit_key": "walk:v"}]
        self.assertEqual(
            inventory.compute_hash(rows),
            __import__("hashlib").sha256(b"w:1:0\twalk:v\n").hexdigest(),
        )

    def test_alternative_change_invalidates_digest(self):
        row = {
            "occurrence_id": "w:1:0", "unit_key": "walk:v",
            "analysis": {"version": "v1", "primary": {"lemma": "walk", "wn_pos": "v"},
                         "alternatives": []},
        }
        changed = copy.deepcopy(row)
        changed["analysis"]["alternatives"].append({"lemma": "walk", "wn_pos": "n"})
        self.assertNotEqual(inventory.compute_hash([row]), inventory.compute_hash([changed]))


if __name__ == "__main__":
    unittest.main()
