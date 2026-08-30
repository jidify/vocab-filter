"""Contrats hors réseau des lots M3/M4 (prompts S3 unitaires et lots).

Réécrit pour le plan de décorrélation lot/stockage : S3-judge-occurrence et
S3-definition-cluster n'appellent plus `llm_client.call` (cache disque par
prompt de lot entier) mais `llm_client.run_units` (cache unitaire, voir
pipeline/llm_store.py) — les mocks ciblent donc `litellm.batch_completion`,
et les clés de cache disque (`cache_key_fields["extra"]`) n'existent plus."""

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import config, mwe_judge
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


def _completion_response(payload: dict):
    """Réponse litellm.completion factice — S3 n'utilise pas de
    response_model Pydantic (JSON libre)."""
    message = type("Message", (), {"content": json.dumps(payload)})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


class LlmStoreIsolatedTests(unittest.TestCase):
    """run_units (pipeline/llm_client.py) stocke chaque décision unitaire
    dans pipeline/llm_store.py — isolé de la vraie data/llm_results.sqlite3
    comme test_llm_store.py."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            config, "LLM_RESULTS_DB_PATH", Path(self._tmp.name) / "llm_results.sqlite3",
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class S3JudgeOccurrencePromptTests(LlmStoreIsolatedTests):
    def test_unit_prompt_uses_task_slot_and_scalar_schema(self):
        task = _task("S3-judge-occurrence", batch=False, size=1, model="catgpt/judge-a")
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(_occurrence_reply())]) as call:
            got = mwe_judge.judge_occurrence("turn off", _occurrence(1), {})

        self.assertEqual(got["label"], "phrasal_verb")
        self.assertEqual(call.call_args.kwargs["model"], "catgpt/judge-a")
        sent_user = call.call_args.kwargs["messages"][0][1]["content"]
        self.assertNotIn('"decisions"', sent_user)

    def test_batch_prompt_returns_one_normalized_decision_per_occurrence(self):
        task = _task("S3-judge-occurrence", batch=True, size=3)
        batch = [("turn off", _occurrence(i)) for i in (1, 2, 3)]
        reply = {"decisions": [
            {"occurrence_id": "m:1:0:7", **_occurrence_reply("phrasal_verb")},
            {"occurrence_id": "m:2:0:7", **_occurrence_reply("littéral")},
            {"occurrence_id": "m:3:0:7", **_occurrence_reply("idiome")},
        ]}
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            got = mwe_judge.judge_occurrences_batch(batch, {})

        self.assertEqual([got[occ["occurrence_id"]]["label"] for _, occ in batch],
                         ["phrasal_verb", "littéral", "idiome"])
        sent_user = call.call_args.kwargs["messages"][0][1]["content"]
        self.assertIn('"decisions"', sent_user)
        self.assertIn("exactement une décision par occurrence_id", sent_user)
        # les 3 occurrences tiennent dans UNE seule tranche (batch_size=3)
        # -> un seul message envoyé, pas 3 appels séparés
        self.assertEqual(len(call.call_args.kwargs["messages"]), 1)

    def test_batch_detects_missing_and_duplicate_ids_as_uncertain_failures(self):
        task = _task("S3-judge-occurrence", batch=True, size=3)
        batch = [("turn off", _occurrence(i)) for i in (1, 2, 3)]
        reply = {"decisions": [
            {"occurrence_id": "m:1:0:7", **_occurrence_reply()},
            {"occurrence_id": "m:1:0:7", **_occurrence_reply()},
        ]}
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]):
            got = mwe_judge.judge_occurrences_batch(batch, {})

        self.assertTrue(all(item["label"] == "incertain" for item in got.values()))
        self.assertTrue(all(mwe_judge.is_llm_failure(item) for item in got.values()))

    def test_occurrence_store_key_no_longer_distinguishes_unit_and_batch(self):
        """Plan de décorrélation lot/stockage : le magasin métier ne porte
        plus mode_batch/batch_size — changer la taille de lot n'invalide
        plus data/mwe_occurrence_decisions.jsonl."""
        same_context = mwe_judge.occurrence_store_key("turn off", "m:1", model="ollama/test")
        also_same = mwe_judge.occurrence_store_key("turn off", "m:1", model="ollama/test")
        self.assertEqual(same_context, also_same)
        different_model = mwe_judge.occurrence_store_key("turn off", "m:1", model="ollama/other")
        self.assertNotEqual(same_context, different_model)

    def test_run_groups_pending_occurrences_in_one_call_and_persists_non_failures(self):
        """Lot de décorrélation : run() ne boucle plus manuellement par
        tranche de occurrence_batch_size — un seul appel à
        _judge_occurrence_units couvre TOUTES les occurrences en attente ;
        le découpage/parallélisme est interne à llm_client.run_units."""
        task = _task("S3-judge-occurrence", batch=True, size=2)
        types = [
            {"idiom": "turn off", "occurrences": [_occurrence(1), _occurrence(2), _occurrence(3)]},
        ]
        candidates_path = SimpleNamespace()
        candidates_path.open = lambda **_: _Context(StringIO("\n".join(json.dumps(row) for row in types)))
        saved_stores = []
        seen_batch_sizes = []

        def fake_judge_units(units, **kwargs):
            seen_batch_sizes.append(kwargs["batch_size"])
            return {unit.unit_id: _normalize_for_run(unit.data[0]) for unit in units}

        with patch("pipeline.corpus.load_segments", return_value=[
            SimpleNamespace(idx=i, en=f"sentence {i}") for i in (1, 2, 3)
        ]), patch.object(mwe_judge.config, "MWE_CANDIDATES_PATH", candidates_path), \
             patch.object(mwe_judge.config, "ensure_out_dir"), \
             patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.mwe_stores, "load_occurrence_store", return_value={}), \
             patch.object(mwe_judge.mwe_stores, "write_occurrence_store", side_effect=saved_stores.append), \
             patch.object(mwe_judge, "_judge_occurrence_units", side_effect=fake_judge_units), \
             patch.object(mwe_judge, "assign_sense_ids"), \
             patch.object(mwe_judge, "assign_cluster_definitions"), \
             patch.object(mwe_judge.atomic, "atomic_write_jsonl"), \
             patch.object(mwe_judge, "write_confirmed_spans"):
            self.assertEqual(mwe_judge.run(), 0)

        # UN seul appel à _judge_occurrence_units, avec le batch_size de la
        # tâche (2) — le découpage en tranches de 2/1 est désormais interne
        # à run_units, plus une responsabilité de run().
        self.assertEqual(seen_batch_sizes, [2])
        self.assertEqual(len(saved_stores[0]), 3)

    def test_run_passes_fully_qualified_model_not_bare_model(self):
        """Régression : run() passait task.bare_model ("catgpt-browser") à
        judge_occurrences_batch/judge_occurrence, qui depuis le Lot U2
        (client unifié) attendent "provider/modèle". Un modèle nu ne matche
        jamais "catgpt/..." dans llm_litellm_catgpt.call_kwargs -> le
        provider custom n'est jamais enregistré -> litellm.exceptions.
        BadRequestError ("LLM Provider NOT provided") sur CHAQUE appel réel
        (mesuré en conditions réelles avec catgpt/catgpt-browser)."""
        task = _task("S3-judge-occurrence", batch=True, size=2, model="catgpt/catgpt-browser")
        types = [
            {"idiom": "turn off", "occurrences": [_occurrence(1)]},
        ]
        candidates_path = SimpleNamespace()
        candidates_path.open = lambda **_: _Context(StringIO("\n".join(json.dumps(row) for row in types)))
        seen_models = []

        def fake_judge_units(units, **kwargs):
            seen_models.append(kwargs.get("model"))
            return {unit.unit_id: _normalize_for_run(unit.data[0]) for unit in units}

        with patch("pipeline.corpus.load_segments", return_value=[
            SimpleNamespace(idx=1, en="sentence 1")
        ]), patch.object(mwe_judge.config, "MWE_CANDIDATES_PATH", candidates_path), \
             patch.object(mwe_judge.config, "ensure_out_dir"), \
             patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.mwe_stores, "load_occurrence_store", return_value={}), \
             patch.object(mwe_judge.mwe_stores, "write_occurrence_store"), \
             patch.object(mwe_judge, "_judge_occurrence_units", side_effect=fake_judge_units), \
             patch.object(mwe_judge, "assign_sense_ids"), \
             patch.object(mwe_judge, "assign_cluster_definitions"), \
             patch.object(mwe_judge.atomic, "atomic_write_jsonl"), \
             patch.object(mwe_judge, "write_confirmed_spans"):
            self.assertEqual(mwe_judge.run(), 0)

        self.assertEqual(seen_models, ["catgpt/catgpt-browser"])


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


class S3DefinitionClusterPromptTests(LlmStoreIsolatedTests):
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
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            got = mwe_judge.choose_cluster_definition("turn off", "VERB", occurrences, {})

        self.assertEqual(got["definition_candidate_id"], "c1")
        self.assertEqual(call.call_args.kwargs["model"], "ollama/definitions")
        sent_user = call.call_args.kwargs["messages"][0][1]["content"]
        self.assertNotIn('"decisions"', sent_user)

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
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            mwe_judge.assign_cluster_definitions(records, {})

        decisions = [item["occurrence_decision"] for item in records[0]["occurrences"]]
        self.assertEqual([item["definition_candidate_id"] for item in decisions], ["turn", "give"])
        sent_user = call.call_args.kwargs["messages"][0][1]["content"]
        self.assertIn('"decisions"', sent_user)


if __name__ == "__main__":
    unittest.main()
