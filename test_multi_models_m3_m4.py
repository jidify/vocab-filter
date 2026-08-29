"""Contrats hors réseau des lots M3/M4 (prompts S3 unitaires et lots)."""

from __future__ import annotations

import json
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import mwe_judge
from pipeline.llm_tasks import TaskLlmConfig


def _task(task_id: str, *, batch: bool, size: int, model: str = "ollama/test-model"):
    provider, bare_model = model.split("/", 1)
    return TaskLlmConfig(
        task_id=task_id, batch_allowed=True, model=model, provider=provider,
        bare_model=bare_model, mode_batch=batch, batch_size=size,
    )


def _occurrence(index: int) -> dict:
    return {
        "occurrence_id": f"m:{index}:0:7", "segment_idx": index,
        "surface": "turn off", "source": "fixture", "vpc_decision_reason": "fixture",
    }


def _occurrence_reply(label: str = "phrasal_verb") -> dict:
    return {
        "label": label, "canonical_form": "turn off", "pos": "VERB",
        "contextual_paraphrase": "switch a device off", "confidence": 0.9,
        "evidence": ["specialized particle meaning"], "reason": "fixture",
        "wordnet_sense_id": None,
    }


class S3JudgeOccurrencePromptTests(unittest.TestCase):
    def test_unit_prompt_uses_task_slot_and_scalar_schema(self):
        task = _task("S3-judge-occurrence", batch=False, size=1, model="catgpt/judge-a")
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm, "call_json", return_value=_occurrence_reply()) as call:
            got = mwe_judge.judge_occurrence("turn off", _occurrence(1), {})

        self.assertEqual(got["label"], "phrasal_verb")
        self.assertEqual(call.call_args.kwargs["model"], "judge-a")
        self.assertEqual(call.call_args.kwargs["backend"], "catgpt")
        self.assertEqual(call.call_args.kwargs["cache_metadata"]["task_id"], "S3-judge-occurrence")
        self.assertFalse(call.call_args.kwargs["cache_metadata"]["mode_batch"])
        self.assertNotIn('"decisions"', call.call_args.args[0])

    def test_batch_prompt_returns_one_normalized_decision_per_occurrence(self):
        task = _task("S3-judge-occurrence", batch=True, size=3)
        batch = [("turn off", _occurrence(i)) for i in (1, 2, 3)]
        reply = {"decisions": [
            {"occurrence_id": "m:1:0:7", **_occurrence_reply("phrasal_verb")},
            {"occurrence_id": "m:2:0:7", **_occurrence_reply("littéral")},
            {"occurrence_id": "m:3:0:7", **_occurrence_reply("idiome")},
        ]}
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm, "call_json", return_value=reply) as call:
            got = mwe_judge.judge_occurrences_batch(batch, {})

        self.assertEqual([got[occ["occurrence_id"]]["label"] for _, occ in batch],
                         ["phrasal_verb", "littéral", "idiome"])
        prompt = call.call_args.args[0]
        self.assertIn('"decisions"', prompt)
        self.assertIn("exactement une décision par occurrence_id", prompt)
        self.assertTrue(call.call_args.kwargs["cache_metadata"]["mode_batch"])
        self.assertEqual(call.call_args.kwargs["cache_metadata"]["batch_size"], 3)

    def test_batch_detects_missing_and_duplicate_ids_as_uncertain_failures(self):
        task = _task("S3-judge-occurrence", batch=True, size=3)
        batch = [("turn off", _occurrence(i)) for i in (1, 2, 3)]
        reply = {"decisions": [
            {"occurrence_id": "m:1:0:7", **_occurrence_reply()},
            {"occurrence_id": "m:1:0:7", **_occurrence_reply()},
        ]}
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm, "call_json", return_value=reply):
            got = mwe_judge.judge_occurrences_batch(batch, {})

        self.assertTrue(all(item["label"] == "incertain" for item in got.values()))
        self.assertTrue(all(mwe_judge.is_llm_failure(item) for item in got.values()))

    def test_occurrence_store_key_distinguishes_unit_and_batch(self):
        unit = mwe_judge.occurrence_store_key(
            "turn off", "m:1", model="ollama/test", mode_batch=False, batch_size=1,
        )
        batch = mwe_judge.occurrence_store_key(
            "turn off", "m:1", model="ollama/test", mode_batch=True, batch_size=3,
        )
        self.assertNotEqual(unit, batch)

    def test_run_groups_pending_occurrences_and_persists_non_failures(self):
        task = _task("S3-judge-occurrence", batch=True, size=2)
        types = [
            {"idiom": "turn off", "occurrences": [_occurrence(1), _occurrence(2), _occurrence(3)]},
        ]
        candidates_path = SimpleNamespace()
        candidates_path.open = lambda **_: _Context(StringIO("\n".join(json.dumps(row) for row in types)))
        saved_stores = []
        batch_sizes = []

        def fake_batch(items, _segments, **_kwargs):
            batch_sizes.append(len(items))
            return {occ["occurrence_id"]: _normalize_for_run(idiom) for idiom, occ in items}

        with patch("pipeline.corpus.load_segments", return_value=[
            SimpleNamespace(idx=i, en=f"sentence {i}") for i in (1, 2, 3)
        ]), patch.object(mwe_judge.config, "MWE_CANDIDATES_PATH", candidates_path), \
             patch.object(mwe_judge.config, "ensure_out_dir"), \
             patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.mwe_stores, "load_occurrence_store", return_value={}), \
             patch.object(mwe_judge.mwe_stores, "write_occurrence_store", side_effect=saved_stores.append), \
             patch.object(mwe_judge, "judge_occurrences_batch", side_effect=fake_batch), \
             patch.object(mwe_judge, "assign_sense_ids"), \
             patch.object(mwe_judge, "assign_cluster_definitions"), \
             patch.object(mwe_judge.atomic, "atomic_write_jsonl"), \
             patch.object(mwe_judge, "write_confirmed_spans"):
            self.assertEqual(mwe_judge.run(), 0)

        self.assertEqual(batch_sizes, [2, 1])
        self.assertEqual(len(saved_stores[0]), 3)


class _Context:
    def __init__(self, value): self.value = value
    def __enter__(self): return self.value
    def __exit__(self, *_): return False


def _normalize_for_run(idiom: str) -> dict:
    return {
        "label": "phrasal_verb", "verdict": "lexicalisé", "canonical_form": idiom,
        "pos": "VERB", "contextual_paraphrase": "fixture", "model_confidence": 0.9,
        "confidence": 0.9, "confidence_features": [], "evidence": ["fixture"], "reason": "fixture",
    }


class S3DefinitionClusterPromptTests(unittest.TestCase):
    def _records(self):
        return [{"occurrences": [
            {"occurrence_id": "a:1", "segment_idx": 1, "occurrence_decision": {
                "label": "phrasal_verb", "canonical_form": "turn off", "pos": "VERB",
                "sense_id": "sense-a", "contextual_paraphrase": "switch a device off",
            }},
            {"occurrence_id": "b:1", "segment_idx": 2, "occurrence_decision": {
                "label": "phrasal_verb", "canonical_form": "give out", "pos": "VERB",
                "sense_id": "sense-b", "contextual_paraphrase": "stop working",
            }},
        ]}]

    def test_unit_definition_prompt_uses_its_task_slot(self):
        task = _task("S3-definition-cluster", batch=False, size=1, model="ollama/definitions")
        candidates = [{"candidate_id": "c1", "definition": "switch a device off", "source": "fixture"}]
        reply = {"candidate_id": "c1", "custom_definition": "",
                 "occurrence_checks": [{"occurrence_id": "a:1", "contradicts": False}]}
        occurrences = self._records()[0]["occurrences"][:1]
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge, "definition_candidates", return_value=candidates), \
             patch.object(mwe_judge.llm, "call_json", return_value=reply) as call:
            got = mwe_judge.choose_cluster_definition("turn off", "VERB", occurrences, {})

        self.assertEqual(got["definition_candidate_id"], "c1")
        self.assertEqual(call.call_args.kwargs["model"], "definitions")
        self.assertFalse(call.call_args.kwargs["cache_metadata"]["mode_batch"])
        self.assertNotIn('"decisions"', call.call_args.args[0])

    def test_batch_definition_prompt_updates_each_existing_cluster_without_reclustering(self):
        task = _task("S3-definition-cluster", batch=True, size=2)
        candidates = [
            [{"candidate_id": "turn", "definition": "switch a device off", "source": "fixture"}],
            [{"candidate_id": "give", "definition": "stop working", "source": "fixture"}],
        ]
        reply = {"decisions": [
            {"cluster_id": "turn off|VERB|sense-a", "candidate_id": "turn", "custom_definition": "",
             "occurrence_checks": [{"occurrence_id": "a:1", "contradicts": False}]},
            {"cluster_id": "give out|VERB|sense-b", "candidate_id": "give", "custom_definition": "",
             "occurrence_checks": [{"occurrence_id": "b:1", "contradicts": False}]},
        ]}
        records = self._records()
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge, "definition_candidates", side_effect=candidates), \
             patch.object(mwe_judge.llm, "call_json", return_value=reply) as call:
            mwe_judge.assign_cluster_definitions(records, {})

        decisions = [item["occurrence_decision"] for item in records[0]["occurrences"]]
        self.assertEqual([item["definition_candidate_id"] for item in decisions], ["turn", "give"])
        self.assertIn('"decisions"', call.call_args.args[0])
        self.assertTrue(call.call_args.kwargs["cache_metadata"]["mode_batch"])


if __name__ == "__main__":
    unittest.main()
