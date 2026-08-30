"""tools/migrate_llm_cache.py — script one-shot de reprise du cache disque
par prompt de lot entier (pipeline_out/cache/) vers le magasin LLM unitaire
(pipeline/llm_store.py, plan de décorrélation lot/stockage).

Fixtures minimales isolées (pas les vraies données du dépôt) : un candidat
S3-judge-occurrence + un cluster S3-definition-cluster, plus un fichier de
cache unitaire et un fichier de cache S5-arbitrate en lot, pour vérifier
qu'ils sont comptés et ignorés (hors périmètre du script — voir sa docstring)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import config, llm_store
from tools import migrate_llm_cache


class MigrateLlmCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        self._db_patch = patch.object(config, "LLM_RESULTS_DB_PATH", root / "llm_results.sqlite3")
        self._db_patch.start()
        self.addCleanup(self._db_patch.stop)

        self._cache_dir = root / "cache"
        self._cache_dir.mkdir()

        candidates_path = root / "mwe_candidates.jsonl"
        candidates_path.write_text(json.dumps({
            "idiom": "turn off",
            "occurrences": [{
                "occurrence_id": "m:1:0:8", "segment_idx": 1, "surface": "turn off",
                "source": "idiomatch", "member_char_spans": [[0, 4], [5, 8]],
            }],
        }) + "\n", encoding="utf-8")
        self._candidates_patch = patch.object(config, "MWE_CANDIDATES_PATH", candidates_path)
        self._candidates_patch.start()
        self.addCleanup(self._candidates_patch.stop)

        decisions_path = root / "mwe_decisions.jsonl"
        decisions_path.write_text(json.dumps({
            "idiom": "turn off",
            "occurrences": [{
                "occurrence_id": "m:1:0:8", "segment_idx": 1,
                "occurrence_decision": {
                    "label": "phrasal_verb", "canonical_form": "turn off", "pos": "VERB",
                    "sense_id": "mwe-custom-v1:deadbeef", "contextual_paraphrase": "switch a device off",
                },
            }],
        }) + "\n", encoding="utf-8")
        self._decisions_patch = patch.object(config, "MWE_DECISIONS_PATH", decisions_path)
        self._decisions_patch.start()
        self.addCleanup(self._decisions_patch.stop)

        self._segments_patch = patch(
            "tools.migrate_llm_cache.load_segments",
            return_value=[type("Segment", (), {"idx": 1, "en": "He turned off the lamp."})()],
        )
        self._segments_patch.start()
        self.addCleanup(self._segments_patch.stop)

        self._wn_patch = patch(
            "pipeline.mwe_judge.wordnet_synset_candidates", return_value=[],
        )
        self._wn_patch.start()
        self.addCleanup(self._wn_patch.stop)

    def _write_cache_file(self, name: str, payload: dict) -> None:
        (self._cache_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_batch_occurrence_file_is_recovered(self):
        self._write_cache_file("occ_batch.json", {"decisions": [
            {"occurrence_id": "m:1:0:8", "label": "phrasal_verb", "canonical_form": "turn off",
             "pos": "VERB", "contextual_paraphrase": "switch a device off", "confidence": 0.9,
             "evidence": ["specialized meaning"], "wordnet_sense_id": None, "reason": "fixture"},
        ]})
        report = migrate_llm_cache.migrate(model="catgpt/x", dry_run=False, cache_dir=self._cache_dir)
        self.assertEqual(report["S3-judge-occurrence_recovered"], 1)
        self.assertEqual(report["rows_written"], 1)

        stats = llm_store.stats(task_id="S3-judge-occurrence")
        self.assertEqual(stats[0]["count"], 1)

    def test_batch_definition_cluster_file_is_recovered(self):
        self._write_cache_file("def_batch.json", {"decisions": [
            {"cluster_id": "turn off|VERB|mwe-custom-v1:deadbeef", "candidate_id": None,
             "custom_definition": "to switch a device off", "reason": "fixture",
             "occurrence_checks": [{"occurrence_id": "m:1:0:8", "contradicts": False}]},
        ]})
        report = migrate_llm_cache.migrate(model="catgpt/x", dry_run=False, cache_dir=self._cache_dir)
        self.assertEqual(report["S3-definition-cluster_recovered"], 1)

        stats = llm_store.stats(task_id="S3-definition-cluster")
        self.assertEqual(stats[0]["count"], 1)

    def test_unmatched_occurrence_id_is_skipped_not_stored(self):
        self._write_cache_file("occ_batch.json", {"decisions": [
            {"occurrence_id": "m:999:0:0", "label": "phrasal_verb", "canonical_form": "unknown",
             "pos": "VERB", "contextual_paraphrase": "x", "confidence": 0.9,
             "evidence": [], "wordnet_sense_id": None, "reason": "fixture"},
        ]})
        report = migrate_llm_cache.migrate(model="catgpt/x", dry_run=False, cache_dir=self._cache_dir)
        self.assertEqual(report["S3-judge-occurrence_recovered"], 0)
        self.assertEqual(report["S3-judge-occurrence_skipped_no_match"], 1)
        self.assertEqual(llm_store.stats(), [])

    def test_unit_mode_file_has_no_identifier_and_is_skipped(self):
        self._write_cache_file("unit.json", {
            "label": "phrasal_verb", "canonical_form": "turn off", "pos": "VERB",
            "contextual_paraphrase": "switch off", "confidence": 0.9,
            "evidence": [], "wordnet_sense_id": None, "reason": "fixture",
        })
        report = migrate_llm_cache.migrate(model="catgpt/x", dry_run=False, cache_dir=self._cache_dir)
        self.assertEqual(report["files_unit_mode_skipped"], 1)
        self.assertEqual(report["rows_written"], 0)

    def test_s5_arbitrate_batch_file_is_recognized_but_not_migrated(self):
        self._write_cache_file("arb_batch.json", {"decisions": [
            {"request_id": "r1", "selected_sense": "turn_off.v.01", "usage_type": "litteral",
             "contextual_meaning_fr": "éteindre", "custom_definition_en": "", "evidence": "x",
             "confidence": 0.9},
        ]})
        report = migrate_llm_cache.migrate(model="catgpt/x", dry_run=False, cache_dir=self._cache_dir)
        self.assertEqual(report["S5-arbitrate_files_seen_not_migrated"], 1)
        self.assertEqual(report["rows_written"], 0)

    def test_dry_run_writes_nothing(self):
        self._write_cache_file("occ_batch.json", {"decisions": [
            {"occurrence_id": "m:1:0:8", "label": "phrasal_verb", "canonical_form": "turn off",
             "pos": "VERB", "contextual_paraphrase": "switch a device off", "confidence": 0.9,
             "evidence": [], "wordnet_sense_id": None, "reason": "fixture"},
        ]})
        report = migrate_llm_cache.migrate(model="catgpt/x", dry_run=True, cache_dir=self._cache_dir)
        self.assertEqual(report["rows_would_write"], 1)
        self.assertEqual(report["rows_written"], 0)
        self.assertEqual(llm_store.stats(), [])

    def test_unparseable_json_file_is_skipped_without_crashing(self):
        (self._cache_dir / "broken.json").write_text("{not json", encoding="utf-8")
        report = migrate_llm_cache.migrate(model="catgpt/x", dry_run=False, cache_dir=self._cache_dir)
        self.assertEqual(report["files_unrecognized_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
