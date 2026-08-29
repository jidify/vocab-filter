from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from pipeline import score, senses
from pipeline.corpus import Segment


def occurrence(word: str, surface: str, candidate: dict | None, token_i: int = 0) -> dict:
    return {
        "occurrence_id": f"w:0:{token_i}", "canonical_form": word, "pos": "n",
        "segment_idx": 0, "surface": surface, "start_char": 0,
        "end_char": len(surface), "analysis": {
            "primary": {"lemma": word, "wn_pos": "n", "source": "spacy"},
            "alternatives": [],
        },
        "multi_token_candidates": [candidate] if candidate else [],
    }


class CompoundFragmentResolutionTests(unittest.TestCase):
    def setUp(self):
        self.segments = [Segment(idx=0, kind="dialogue", speaker=None, en="context", fr="contexte")]

    def test_six_contexts_block_simple_wordnet_senses(self):
        cases = [
            ("york", "York", "New York", "named_entity:GPE", .95),
            ("virgin", "Virgin", "Virgin Mary", "named_entity:PERSON", .95),
            ("ranch", "ranch", "ranch dip", "nominal_compound", .82),
            ("observation", "observation", "observation deck", "nominal_compound", .82),
            ("nursing", "nursing", "nursing home", "nominal_compound", .82),
            ("crystal", "crystal", "crystal ball", "nominal_compound", .82),
        ]
        for i, (word, surface, compound, kind, confidence) in enumerate(cases):
            with self.subTest(compound=compound):
                candidate = {"candidate_id": f"mt:0:{i}", "surface": compound,
                             "candidate_types": [kind], "score": confidence}
                record = senses.resolve_joint_occurrence(
                    occurrence(word, surface, candidate, i), self.segments
                )
                self.assertEqual(record["best_sense"], senses.MULTI_TOKEN_FRAGMENT_SENSE_ID)
                self.assertEqual(record["resolution_status"], "excluded")
                self.assertEqual(record["covering_surfaces"], [compound])

    def test_same_lemma_autonomous_occurrence_is_not_suppressed(self):
        covered = occurrence("ranch", "ranch", {
            "candidate_id": "mt:0:0:9", "surface": "ranch dip",
            "candidate_types": ["nominal_compound"], "score": .82,
        })
        autonomous = occurrence("ranch", "ranch", None, 1)
        self.assertIsNotNone(senses.multi_token_fragment_record(covered, self.segments))
        self.assertIsNone(senses.multi_token_fragment_record(autonomous, self.segments))

    def test_weak_hypothesis_cannot_delete_a_word(self):
        weak = occurrence("crystal", "crystal", {
            "candidate_id": "mt:0:weak", "surface": "crystal clear",
            "candidate_types": ["multi_token_entity"], "score": .70,
        })
        self.assertIsNone(senses.multi_token_fragment_record(weak, self.segments))

    def test_export_filter_is_occurrence_scoped(self):
        excluded = {
            "word": "ranch", "pos": "n", "segment_idx": 0,
            "resolution_status": "excluded",
            "exclusion_reason": "covered_by_confirmed_multi_token",
        }
        autonomous = {
            "word": "ranch", "pos": "n", "segment_idx": 1,
            "target_surface": "ranch", "best_sense": "ranch.n.01",
            "candidates": [{"synset": "ranch.n.01", "definition": "farm", "fr_hits": []}],
            "needs_review": False, "margin": 1.0,
        }
        selected = json.dumps({"lemma": "ranch", "wn_pos": "n", "zipf": 3.0}) + "\n"
        paths = lambda text: mock.Mock(open=mock.Mock(side_effect=lambda **_: io.StringIO(text)))
        with mock.patch.object(score.config, "SELECTED_TYPES_PATH", paths(selected)), \
             mock.patch.object(score.config, "SENSES_PATH", paths(
                 json.dumps(excluded) + "\n" + json.dumps(autonomous) + "\n")), \
             mock.patch.object(score, "load_manual_corrections", return_value={}):
            records = score.build_records()
        self.assertEqual([(r["lemma"], r["segment_idx"]) for r in records], [("ranch", 1)])


if __name__ == "__main__":
    unittest.main()
