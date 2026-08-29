"""Contrats hors réseau du lot M5 (prompts S5-arbitrate unitaire et lot)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline import senses
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


class S5ArbitrateUnitTests(unittest.TestCase):
    def test_unit_prompt_uses_task_slot_and_scalar_schema(self):
        task = _task("S5-arbitrate", batch=False, size=1, model="catgpt/arbiter-a")
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client, "call", return_value=_reply()) as call:
            got = senses.arbitrate("turn off", "VERB", "He turned off the lamp.", _synsets())

        self.assertEqual(got["selected_sense"], "turn_off.v.01")
        self.assertEqual(call.call_args.kwargs["model"], "catgpt/arbiter-a")
        self.assertFalse(call.call_args.kwargs["cache_key_fields"]["extra"]["mode_batch"])
        self.assertNotIn('"decisions"', call.call_args.kwargs["prompt"])

    def test_llm_failure_returns_none_selection_like_before(self):
        task = _task("S5-arbitrate", batch=False, size=1)
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client, "call", side_effect=senses.llm_client.LLMError("down")):
            got = senses.arbitrate("turn off", "VERB", "He turned off the lamp.", _synsets())

        self.assertIsNone(got["selected_sense"])
        self.assertEqual(got["confidence"], 0.0)
        self.assertIn("down", got["error"])


class S5ArbitrateBatchTests(unittest.TestCase):
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
             patch.object(senses.llm_client, "call", return_value=reply) as call:
            got = senses.arbitrate_batch(self._requests())

        self.assertEqual([got[r]["selected_sense"] for r in ("r1", "r2", "r3")],
                         ["turn_off.v.01", "turn_off.v.02", "aucun_sens_adapte"])
        prompt = call.call_args.kwargs["prompt"]
        self.assertIn('"decisions"', prompt)
        self.assertIn("exactement une décision par request_id", prompt)
        self.assertTrue(call.call_args.kwargs["cache_key_fields"]["extra"]["mode_batch"])
        self.assertEqual(call.call_args.kwargs["cache_key_fields"]["extra"]["batch_size"], 3)

    def test_batch_detects_missing_and_duplicate_ids_as_error_results(self):
        task = _task("S5-arbitrate", batch=True, size=3)
        reply = {"decisions": [
            {"request_id": "r1", **_reply()},
            {"request_id": "r1", **_reply()},
        ]}
        with patch.object(senses, "task_config", return_value=task), \
             patch.object(senses.llm_client, "call", return_value=reply):
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
             patch.object(senses.llm_client, "call", side_effect=senses.llm_client.LLMError("down")):
            got = senses.arbitrate_batch(self._requests())

        self.assertEqual(set(got), {"r1", "r2", "r3"})
        self.assertTrue(all(item["selected_sense"] is None for item in got.values()))


if __name__ == "__main__":
    unittest.main()
