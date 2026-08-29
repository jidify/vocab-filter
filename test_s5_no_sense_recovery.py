from __future__ import annotations

import io
import json
import csv
from contextlib import contextmanager
import unittest
from unittest import mock

from pipeline import export, score, senses


def path_with(text: str):
    path = mock.Mock()
    path.exists.return_value = True
    path.open.side_effect = lambda **_kwargs: io.StringIO(text)
    return path


class RecoveryOrderTests(unittest.TestCase):
    def test_covering_compound_precedes_custom_and_review(self):
        occurrence = {"canonical_form": "latch", "pos": "n", "segment_idx": 7,
                      "multi_token_candidates": [{"candidate_id": "compound:7"}]}
        record = {"best_sense": "aucun_sens_adapte", "candidates": [],
                  "arbitration": {"custom_definition_en": "A fastening device.",
                                  "evidence": "the latch", "confidence": 1.0}}
        recovered = senses.recover_no_sense(occurrence, record)
        self.assertEqual(recovered["recovery"]["route"], "mwe_or_compound")
        self.assertTrue(recovered["needs_review"])
        self.assertEqual([a["branch"] for a in recovered["recovery"]["attempts"]],
                         ["alternate_lemma_pos", "mwe_or_compound", "human_review"])

    def test_justified_custom_precedes_review_and_is_stable(self):
        occurrence = {"canonical_form": "latch", "pos": "n", "segment_idx": 7}
        base = {"best_sense": "aucun_sens_adapte", "candidates": [],
                "arbitration": {"custom_definition_en": "A context-specific fastening device.",
                                "evidence": "It closes the gate.", "confidence": .91}}
        first = senses.recover_no_sense(occurrence, json.loads(json.dumps(base)))
        second = senses.recover_no_sense(occurrence, json.loads(json.dumps(base)))
        self.assertEqual(first["best_sense"], second["best_sense"])
        self.assertTrue(first["best_sense"].startswith("custom.word."))
        self.assertFalse(first["needs_review"])
        self.assertEqual(first["recovery"]["route"], "custom_justified")


class LatchBlockingGateTests(unittest.TestCase):
    def test_legacy_no_sense_survives_build_score_and_review_queue(self):
        selected = json.dumps({"lemma": "latch", "wn_pos": "n", "zipf": 3.2}) + "\n"
        occurrence = {
            "word": "latch", "pos": "n", "segment_idx": 42, "target_surface": "latch",
            "context": "She checks the latch.", "best_sense": "aucun_sens_adapte",
            "candidates": [
                {"synset": "latch.n.01", "definition": "a fastener", "fr_hits": []},
                {"synset": "latch.n.02", "definition": "a lock", "fr_hits": []},
            ],
            "needs_review": True, "margin": 0.0,
        }
        with mock.patch.object(score.config, "SELECTED_TYPES_PATH", path_with(selected)), \
             mock.patch.object(score.config, "SENSES_PATH", path_with(json.dumps(occurrence) + "\n")), \
             mock.patch.object(score, "load_manual_corrections", return_value={}):
            records = score.build_records()
        self.assertEqual(len(records), 1, "gate bloquant: latch ne doit jamais disparaître")
        self.assertEqual(records[0]["lemma"], "latch")
        self.assertTrue(records[0]["needs_review"])
        self.assertEqual(records[0]["candidate_senses"], ["latch.n.01", "latch.n.02"])

        with mock.patch.object(score.sense_fr, "load_store", return_value={}), \
             mock.patch.object(score.senses, "load_occurrences_by_sense", return_value={}), \
             mock.patch.object(score, "load_manual_corrections", return_value={}):
            units = score.aggregate_and_score(records)
        export.assert_no_uncertain_occurrence_lost(records, units)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["canonical_form"], "latch")
        self.assertTrue(units[0]["needs_review"])
        self.assertIn("She checks the latch", units[0]["contexte_en"])
        self.assertEqual(units[0]["review_action"],
                         "select another analysis or justify a custom sense")

        # Artefacts aval : JSONL exhaustif, CSV final et review queue. Les
        # sinks atomiques sont capturés en mémoire pour rester hors réseau et
        # indépendants des ACL temporaires Windows.
        json_sink = []
        with mock.patch.object(export.atomic, "atomic_write_jsonl",
                               side_effect=lambda _path, rows: json_sink.extend(rows)):
            export.write_jsonl(units, "vocab.jsonl")
        self.assertEqual(json_sink[0]["canonical_form"], "latch")

        for writer_fn, name in ((export.write_csv, "vocab.csv"),
                                (export.write_review_queue, "review_queue.csv")):
            buffer = io.StringIO()

            @contextmanager
            def sink(*_args, **_kwargs):
                yield buffer

            with mock.patch.object(export.atomic, "atomic_open", side_effect=sink):
                writer_fn(units, name)
            buffer.seek(0)
            rows = list(csv.DictReader(buffer))
            self.assertEqual(rows[0]["canonical_form"], "latch")
            self.assertEqual(rows[0]["candidate_senses"], "latch.n.01/latch.n.02")
            self.assertEqual(rows[0]["recovery_route"], "human_review")

    def test_accounting_invariant_rejects_silent_loss(self):
        records = [{"key": ("latch", "n", "unresolved.x"), "needs_review": True}]
        with self.assertRaises(RuntimeError):
            export.assert_no_uncertain_occurrence_lost(records, [])


if __name__ == "__main__":
    unittest.main()
