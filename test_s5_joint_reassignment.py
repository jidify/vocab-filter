from __future__ import annotations

import copy
import unittest
from unittest import mock

from pipeline import inventory, senses
from pipeline.corpus import Segment
import run_pipeline


class FakeSynset:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class ControlledInventoryTests(unittest.TestCase):
    def test_named_fixtures_can_open_only_the_expected_real_ids(self):
        cases = [
            ("frost", "v", "frosting", "n", "frosting.n.01"),
            ("creep", "v", "creeping", "a", "creeping.s.01"),
            ("facility", "n", "facilities", "n", "facilities.n.02"),
            ("stress", "v", "destress", "v", "de_stress.v.01"),
            ("bitch", "v", "bitch", "n", "bitch.n.01"),
        ]
        for primary_lemma, primary_pos, alt_lemma, alt_pos, expected in cases:
            row = {
                "canonical_form": primary_lemma, "pos": primary_pos,
                "analysis": {
                    "primary": {"lemma": primary_lemma, "wn_pos": primary_pos, "source": "spacy"},
                    "alternatives": [{"lemma": alt_lemma, "wn_pos": alt_pos, "source": "fixture"}],
                },
            }

            def fake_get_synsets(lemma, pos, *, target=expected, alt=alt_lemma):
                return [FakeSynset(target)] if lemma == alt else [FakeSynset(f"{lemma}.{pos}.01")]

            with mock.patch.object(senses, "get_synsets", side_effect=fake_get_synsets):
                opened = senses.controlled_analysis_inventory(row)
            allowed = {sid for hypothesis in opened for sid in hypothesis["synset_ids"]}
            self.assertIn(expected, allowed)

    def test_joint_choice_keeps_initial_analysis_and_rejects_unlisted_id(self):
        row = {
            "canonical_form": "frost", "pos": "v", "surface": "frosting",
            "segment_idx": 1,
            "analysis": {
                "primary": {"lemma": "frost", "wn_pos": "v", "source": "spacy"},
                "alternatives": [{"lemma": "frosting", "wn_pos": "n", "source": "gerund"}],
            },
        }
        opened = [
            {"lemma": "frost", "pos": "v", "source": "spacy", "analysis_rank": 0,
             "synset_ids": ["frost.v.01"]},
            {"lemma": "frosting", "pos": "n", "source": "gerund", "analysis_rank": 1,
             "synset_ids": ["frosting.n.01"]},
        ]

        def analyzed(word, pos, *_args, **_kwargs):
            sense_id = "frosting.n.01" if pos == "n" else "frost.v.01"
            score = 0.9 if pos == "n" else 0.1
            return {"word": word, "pos": pos, "best_sense": sense_id,
                    "candidates": [{"synset": sense_id, "final_score": score}]}

        with mock.patch.object(senses, "controlled_analysis_inventory", return_value=opened), \
             mock.patch.object(senses, "analyze_occurrence", side_effect=analyzed):
            record = senses.resolve_joint_occurrence(row, [Segment(1, "dialogue", None, "frosting", "")])
        self.assertEqual(record["best_sense"], "frosting.n.01")
        self.assertEqual(record["word"], "frost")
        self.assertEqual(record["resolved_lemma"], "frosting")
        self.assertTrue(record["joint_resolution"]["reassigned"])
        self.assertEqual(record["joint_resolution"]["initial_analysis"]["source"], "spacy")


class ResumeDigestTests(unittest.TestCase):
    def test_policy_version_participates_in_resume_digest(self):
        old = senses.SENSE_RESOLUTION_VERSION
        first = senses.resolution_digest("inventory")
        try:
            senses.SENSE_RESOLUTION_VERSION = old + "-changed"
            self.assertNotEqual(first, senses.resolution_digest("inventory"))
        finally:
            senses.SENSE_RESOLUTION_VERSION = old

    def test_old_or_other_policy_records_are_not_reused(self):
        rows = [{"occurrence_id": "w:1:0", "inventory_digest": "inv",
                 "resolution_digest": "old", "best_sense": "frost.v.01"}]
        data = "".join(__import__("json").dumps(r) + "\n" for r in rows)
        path = mock.Mock()
        path.exists.return_value = True
        path.open.side_effect = lambda **_kwargs: __import__("io").StringIO(data)
        with mock.patch.object(senses.config, "SENSES_PATH", path):
            self.assertEqual(senses.load_existing_senses("inv", "new"), {})

    def test_surface_change_invalidates_inventory_digest(self):
        row = {"occurrence_id": "w:1:0", "unit_key": "word:frost:v:unresolved",
               "surface": "frosting", "analysis": {"primary": {"lemma": "frost"}}}
        changed = copy.deepcopy(row)
        changed["surface"] = "frosted"
        self.assertNotEqual(inventory.compute_hash([row]), inventory.compute_hash([changed]))


class StandardOrchestrationTests(unittest.TestCase):
    def test_structural_reassignment_is_in_the_standard_path_before_export(self):
        names = [name for name, _ in run_pipeline.STAGES]
        self.assertLess(names.index("senses"), names.index("sense_fr_reassign"))
        self.assertLess(names.index("sense_fr_reassign"), names.index("export"))
        self.assertIn("sense_fr_reassign", run_pipeline.FRONTIER_STAGES)


if __name__ == "__main__":
    unittest.main()
