"""Tests du magasin de résultats LLM unitaire — pipeline/llm_store.py.

Isolé de la vraie data/llm_results.sqlite3 comme test_llm_client.py isole
pipeline_out/cache/ (patch de config.LLM_RESULTS_DB_PATH vers un tmpdir)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import config, llm_store


class LlmStoreIsolatedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            config, "LLM_RESULTS_DB_PATH", Path(self._tmp.name) / "llm_results.sqlite3",
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class PayloadSigTests(unittest.TestCase):
    def test_deterministic_and_order_independent_keys(self):
        a = llm_store.payload_sig({"b": 2, "a": 1})
        b = llm_store.payload_sig({"a": 1, "b": 2})
        self.assertEqual(a, b)

    def test_different_payload_different_sig(self):
        a = llm_store.payload_sig({"a": 1})
        b = llm_store.payload_sig({"a": 2})
        self.assertNotEqual(a, b)


class GetPutManyTests(LlmStoreIsolatedTests):
    def test_miss_then_hit_round_trip(self):
        sig = llm_store.payload_sig({"surface": "turn off"})
        hits = llm_store.get_many(
            task_id="S3-judge-occurrence", model="ollama/mistral-small:24b",
            protocol="s3-judge-prompt-5", wanted=[("m:1:0:8", sig)],
        )
        self.assertEqual(hits, {})

        n = llm_store.put_many([
            llm_store.ResultRow(
                task_id="S3-judge-occurrence", model="ollama/mistral-small:24b",
                protocol="s3-judge-prompt-5", unit_id="m:1:0:8",
                payload={"surface": "turn off"},
                result={"label": "phrasal_verb", "confidence": 0.9},
                batch_size=50, mode_batch=True, source="live",
            ),
        ])
        self.assertEqual(n, 1)

        hits = llm_store.get_many(
            task_id="S3-judge-occurrence", model="ollama/mistral-small:24b",
            protocol="s3-judge-prompt-5", wanted=[("m:1:0:8", sig)],
        )
        self.assertEqual(hits, {"m:1:0:8": {"label": "phrasal_verb", "confidence": 0.9}})

    def test_stale_payload_signature_is_a_miss_not_a_silent_hit(self):
        old_sig = llm_store.payload_sig({"sentence": "He turned off the light."})
        llm_store.put_many([
            llm_store.ResultRow(
                task_id="S3-judge-occurrence", model="m", protocol="p1", unit_id="m:1:0:8",
                payload={"sentence": "He turned off the light."},
                result={"label": "phrasal_verb", "confidence": 0.9},
            ),
        ])
        new_sig = llm_store.payload_sig({"sentence": "He turned off the radio."})
        hits = llm_store.get_many(
            task_id="S3-judge-occurrence", model="m", protocol="p1",
            wanted=[("m:1:0:8", new_sig)],
        )
        self.assertEqual(hits, {})
        # l'ancienne entrée reste consultable avec son ancienne signature —
        # sert au diagnostic (payload conservé), pas d'écrasement silencieux
        hits_old = llm_store.get_many(
            task_id="S3-judge-occurrence", model="m", protocol="p1",
            wanted=[("m:1:0:8", old_sig)],
        )
        self.assertEqual(hits_old["m:1:0:8"]["label"], "phrasal_verb")

    def test_different_model_is_isolated(self):
        sig = llm_store.payload_sig({"x": 1})
        llm_store.put_many([
            llm_store.ResultRow(
                task_id="S6-translate-frontier", model="openai/gpt-5-mini", protocol="v1",
                unit_id="beat.n.08", payload={"x": 1}, result={"fr": ["battement"]},
            ),
        ])
        hits_same = llm_store.get_many(
            task_id="S6-translate-frontier", model="openai/gpt-5-mini", protocol="v1",
            wanted=[("beat.n.08", sig)],
        )
        hits_other_model = llm_store.get_many(
            task_id="S6-translate-frontier", model="openai/gpt-5-nano", protocol="v1",
            wanted=[("beat.n.08", sig)],
        )
        self.assertEqual(hits_same, {"beat.n.08": {"fr": ["battement"]}})
        self.assertEqual(hits_other_model, {})

    def test_batch_size_never_affects_hit(self):
        """Le coeur du besoin : une unité stockée depuis un lot de 50 doit
        rester un hit quand on la redemande comme si elle venait d'un lot de 10
        — batch_size/mode_batch ne sont que des colonnes d'audit."""
        sig = llm_store.payload_sig({"occurrence_id": "m:1:0:8"})
        llm_store.put_many([
            llm_store.ResultRow(
                task_id="S3-judge-occurrence", model="m", protocol="p1", unit_id="m:1:0:8",
                payload={"occurrence_id": "m:1:0:8"}, result={"label": "phrasal_verb"},
                batch_size=50, mode_batch=True,
            ),
        ])
        hits = llm_store.get_many(
            task_id="S3-judge-occurrence", model="m", protocol="p1",
            wanted=[("m:1:0:8", sig)],
        )
        self.assertEqual(hits, {"m:1:0:8": {"label": "phrasal_verb"}})

    def test_put_many_persists_partial_batch_results(self):
        """Si seules 2 unités sur 3 d'un lot ont réussi, put_many ne reçoit
        que ces 2-là et les écrit quand même — pas de tout-ou-rien par lot."""
        n = llm_store.put_many([
            llm_store.ResultRow(
                task_id="t", model="m", protocol="p", unit_id="u1",
                payload={"i": 1}, result={"ok": True},
            ),
            llm_store.ResultRow(
                task_id="t", model="m", protocol="p", unit_id="u2",
                payload={"i": 2}, result={"ok": True},
            ),
        ])
        self.assertEqual(n, 2)
        hits = llm_store.get_many(
            task_id="t", model="m", protocol="p",
            wanted=[("u1", llm_store.payload_sig({"i": 1})),
                    ("u2", llm_store.payload_sig({"i": 2})),
                    ("u3", llm_store.payload_sig({"i": 3}))],
        )
        self.assertEqual(set(hits), {"u1", "u2"})

    def test_get_many_empty_wanted_returns_empty_without_query(self):
        self.assertEqual(llm_store.get_many(task_id="t", model="m", protocol="p", wanted=[]), {})

    def test_put_many_empty_rows_is_noop(self):
        self.assertEqual(llm_store.put_many([]), 0)

    def test_stats_groups_by_task_model_protocol(self):
        llm_store.put_many([
            llm_store.ResultRow(task_id="A", model="m1", protocol="p", unit_id="u1",
                                payload={"i": 1}, result={}),
            llm_store.ResultRow(task_id="A", model="m1", protocol="p", unit_id="u2",
                                payload={"i": 2}, result={}),
            llm_store.ResultRow(task_id="B", model="m2", protocol="p", unit_id="u3",
                                payload={"i": 3}, result={}),
        ])
        rows = llm_store.stats()
        by_task = {(r["task_id"], r["model"]): r["count"] for r in rows}
        self.assertEqual(by_task[("A", "m1")], 2)
        self.assertEqual(by_task[("B", "m2")], 1)
        filtered = llm_store.stats(task_id="A")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["count"], 2)

    def test_reinserting_same_key_overwrites_not_duplicates(self):
        row = llm_store.ResultRow(
            task_id="t", model="m", protocol="p", unit_id="u1",
            payload={"i": 1}, result={"v": 1},
        )
        llm_store.put_many([row])
        row2 = llm_store.ResultRow(
            task_id="t", model="m", protocol="p", unit_id="u1",
            payload={"i": 1}, result={"v": 2},
        )
        llm_store.put_many([row2])
        rows = llm_store.stats(task_id="t")
        self.assertEqual(rows[0]["count"], 1)
        hits = llm_store.get_many(
            task_id="t", model="m", protocol="p",
            wanted=[("u1", llm_store.payload_sig({"i": 1}))],
        )
        self.assertEqual(hits["u1"]["v"], 2)


if __name__ == "__main__":
    unittest.main()
