"""Contrats hors réseau du lot M5 (prompts S5-arbitrate unitaire et lot).

Réécrit pour le plan de décorrélation lot/stockage : S5-arbitrate n'appelle
plus `llm_client.call` (cache disque par prompt de lot entier) mais
`llm_client.run_units` (cache unitaire, voir pipeline/llm_store.py) — les
mocks ciblent donc `litellm.batch_completion`."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import config, senses
from pipeline.llm_tasks import TaskLlmConfig


def _task(task_id: str, *, batch: bool, size: int, model: str = "ollama/test-model"):
    provider, bare_model = model.split("/", 1)
    return TaskLlmConfig(
        task_id=task_id, batch_allowed=True, model=model, provider=provider,
        bare_model=bare_model, mode_batch=batch, batch_size=size,
    )


class _Synset:
    def __init__(self, name: str, definition: str):
        self._name, self._definition = name, definition

    def name(self) -> str:
        return self._name

    def definition(self) -> str:
        return self._definition


def _synsets() -> list[_Synset]:
    return [_Synset("turn_off.v.01", "switch a device off"),
            _Synset("turn_off.v.02", "cause to feel repugnance")]


def _reply(selected: str = "turn_off.v.01") -> dict:
    return {
        "selected_sense": selected, "usage_type": "litteral",
        "contextual_meaning_fr": "éteindre", "custom_definition_en": "",
        "evidence": "the lamp went dark", "confidence": 0.9,
    }


def _completion_response(payload: dict):
    """Réponse litellm.completion factice — S5-arbitrate n'utilise pas de
    response_model Pydantic (JSON libre)."""
    message = type("Message", (), {"content": json.dumps(payload)})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


class LlmStoreIsolatedTests(unittest.TestCase):
    """run_units (pipeline/llm_client.py) stocke chaque décision dans
    pipeline/llm_store.py — isolé de la vraie data/llm_results.sqlite3 comme
    test_llm_store.py."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            config, "LLM_RESULTS_DB_PATH", Path(self._tmp.name) / "llm_results.sqlite3",
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class S5ArbitrateUnitTests(LlmStoreIsolatedTests):
    def test_unit_prompt_uses_task_slot_and_scalar_schema(self):
        task = _task("S5-arbitrate", batch=False, size=1, model="catgpt/arbiter-a")
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(_reply())]) as call:
            got = senses.arbitrate("turn off", "VERB", "He turned off the lamp.", _synsets())

        self.assertEqual(got["selected_sense"], "turn_off.v.01")
        self.assertEqual(call.call_args.kwargs["model"], "catgpt/arbiter-a")
        sent_user = call.call_args.kwargs["messages"][0][1]["content"]
        self.assertNotIn('"decisions"', sent_user)

    def test_llm_failure_returns_none_selection_like_before(self):
        task = _task("S5-arbitrate", batch=False, size=1)
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[RuntimeError("down")]):
            got = senses.arbitrate("turn off", "VERB", "He turned off the lamp.", _synsets())

        self.assertIsNone(got["selected_sense"])
        self.assertEqual(got["confidence"], 0.0)
        self.assertIn("down", got["error"])

    def test_same_word_different_context_never_collides(self):
        """unit_id est dérivé du contenu (pas d'identifiant externe
        disponible dans analyze_occurrence) — deux contextes distincts pour
        le même mot ne doivent jamais partager une ligne du magasin."""
        task = _task("S5-arbitrate", batch=False, size=1)
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(_reply("turn_off.v.01"))]):
            first = senses.arbitrate("turn off", "VERB", "He turned off the lamp.", _synsets())
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(_reply("turn_off.v.02"))]) as call2:
            second = senses.arbitrate("turn off", "VERB", "The joke turned him off.", _synsets())
        call2.assert_called_once()  # pas de hit fantôme depuis le premier contexte
        self.assertEqual(first["selected_sense"], "turn_off.v.01")
        self.assertEqual(second["selected_sense"], "turn_off.v.02")


class S5ArbitrateBatchTests(LlmStoreIsolatedTests):
    def _requests(self):
        return [
            ("r1", "turn off", "VERB", "He turned off the lamp.", _synsets()),
            ("r2", "turn off", "VERB", "The joke turned him off.", _synsets()),
            ("r3", "turn off", "VERB", "She turned off the highway.", _synsets()),
        ]

    def test_batch_prompt_returns_one_decision_per_request(self):
        task = _task("S5-arbitrate", batch=True, size=3)
        reply = {"decisions": [
            {"request_id": "r1", **_reply("turn_off.v.01")},
            {"request_id": "r2", **_reply("turn_off.v.02")},
            {"request_id": "r3", **_reply("aucun_sens_adapte")},
        ]}
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            got = senses.arbitrate_batch(self._requests())

        self.assertEqual([got[r]["selected_sense"] for r in ("r1", "r2", "r3")],
                         ["turn_off.v.01", "turn_off.v.02", "aucun_sens_adapte"])
        sent_user = call.call_args.kwargs["messages"][0][1]["content"]
        self.assertIn('"decisions"', sent_user)
        self.assertIn("exactement une décision par request_id", sent_user)
        # les 3 requêtes tiennent dans UNE seule tranche (batch_size=3)
        self.assertEqual(len(call.call_args.kwargs["messages"]), 1)

    def test_batch_detects_missing_and_duplicate_ids_as_error_results(self):
        task = _task("S5-arbitrate", batch=True, size=3)
        reply = {"decisions": [
            {"request_id": "r1", **_reply()},
            {"request_id": "r1", **_reply()},
        ]}
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]):
            got = senses.arbitrate_batch(self._requests())

        self.assertTrue(all(item["selected_sense"] is None for item in got.values()))
        self.assertTrue(all("error" in item for item in got.values()))

    def test_batch_rejects_more_requests_than_configured_batch_size(self):
        task = _task("S5-arbitrate", batch=True, size=2)
        with patch.object(senses, "task_config", return_value=task):
            with self.assertRaises(ValueError):
                senses.arbitrate_batch(self._requests())

    def test_llm_failure_marks_every_request_as_error(self):
        task = _task("S5-arbitrate", batch=True, size=3)
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[RuntimeError("down")]):
            got = senses.arbitrate_batch(self._requests())

        self.assertEqual(set(got), {"r1", "r2", "r3"})
        self.assertTrue(all(item["selected_sense"] is None for item in got.values()))

    def test_changing_batch_size_across_runs_triggers_zero_new_calls(self):
        """Le cœur du plan de décorrélation : les mêmes 3 requêtes, une
        seconde fois avec une taille de lot différente, ne doivent
        déclencher aucun nouvel appel — le magasin unitaire ne connaît pas
        batch_size."""
        task3 = _task("S5-arbitrate", batch=True, size=3)
        reply = {"decisions": [
            {"request_id": "r1", **_reply("turn_off.v.01")},
            {"request_id": "r2", **_reply("turn_off.v.02")},
            {"request_id": "r3", **_reply("aucun_sens_adapte")},
        ]}
        with patch.object(senses, "task_config", return_value=task3), \
             patch.object(senses.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            first = senses.arbitrate_batch(self._requests())
        call.assert_called_once()

        task2 = _task("S5-arbitrate", batch=True, size=2)
        requests_reordered = self._requests()[:2]  # 2 des 3, batch_size différent
        with patch.object(senses, "task_config", return_value=task2), \
             patch.object(senses.llm_client.litellm, "batch_completion") as call2:
            second = senses.arbitrate_batch(requests_reordered)
        call2.assert_not_called()
        self.assertEqual(second["r1"], first["r1"])
        self.assertEqual(second["r2"], first["r2"])


if __name__ == "__main__":
    unittest.main()
