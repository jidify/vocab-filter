"""Tests offline (litellm mocké) du client LLM unique — pipeline/llm_client.py.

Couvre le contrat du Lot U1 du plan d'unification (fix_pipeline/multi_models/
report_multi_models.md §4bis) : routage des 3 providers, api_base ollama,
response_format dict vs Pydantic, hit/miss de cache, conversion des
exceptions en LLMError. Aucun appel réseau réel."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel, ValidationError

from pipeline import config, llm_client


class _Choice:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})()


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Decision(BaseModel):
    label: str
    confidence: float


class LlmClientCacheIsolatedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(config, "CACHE_DIR", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)


class CallDictModeTests(LlmClientCacheIsolatedTests):
    def test_cache_miss_calls_completion_and_writes_cache(self):
        with patch.object(llm_client.litellm, "completion",
                          return_value=_Response('{"ok": true}')) as mocked:
            result = llm_client.call(
                model="ollama/mistral-small:24b", system="s", prompt="p",
                cache_key_fields={"task_id": "t", "model": "ollama/mistral-small:24b"},
            )
        self.assertEqual(result, {"ok": True})
        mocked.assert_called_once()
        # deuxième appel : cache hit, pas de deuxième appel réseau
        with patch.object(llm_client.litellm, "completion") as mocked2:
            result2 = llm_client.call(
                model="ollama/mistral-small:24b", system="s", prompt="p",
                cache_key_fields={"task_id": "t", "model": "ollama/mistral-small:24b"},
            )
        mocked2.assert_not_called()
        self.assertEqual(result2, {"ok": True})

    def test_ollama_gets_api_base_and_temperature(self):
        with patch.object(llm_client.litellm, "completion",
                          return_value=_Response('{"ok": true}')) as mocked:
            llm_client.call(
                model="ollama/mistral-small:24b", system="s", prompt="p",
                cache_key_fields={"k": "ollama"},
            )
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["api_base"], config.OLLAMA_URL)
        self.assertEqual(kwargs["temperature"], config.LLM_TEMPERATURE)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_openai_skips_temperature(self):
        with patch.object(llm_client.litellm, "completion",
                          return_value=_Response('{"ok": true}')) as mocked:
            llm_client.call(
                model="openai/gpt-5-mini", system="s", prompt="p",
                cache_key_fields={"k": "openai"},
            )
        self.assertNotIn("temperature", mocked.call_args.kwargs)
        self.assertNotIn("api_base", mocked.call_args.kwargs)

    def test_catgpt_gets_registered_and_allowed_reasoning_effort(self):
        with patch.object(llm_client.litellm, "completion",
                          return_value=_Response('{"ok": true}')) as mocked:
            llm_client.call(
                model="catgpt/browser", system="s", prompt="p",
                cache_key_fields={"k": "catgpt"},
                reasoning_effort="low",
            )
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertIn("reasoning_effort", kwargs["allowed_openai_params"])
        self.assertTrue(
            any(p.get("provider") == "catgpt" for p in llm_client.litellm.custom_provider_map)
        )

    def test_completion_exception_becomes_llm_error(self):
        with patch.object(llm_client.litellm, "completion", side_effect=RuntimeError("boom")):
            with self.assertRaises(llm_client.LLMError):
                llm_client.call(
                    model="ollama/x", system="s", prompt="p", cache_key_fields={"k": 1},
                )

    def test_invalid_json_becomes_llm_error(self):
        with patch.object(llm_client.litellm, "completion", return_value=_Response("not json")):
            with self.assertRaises(llm_client.LLMError):
                llm_client.call(
                    model="ollama/x", system="s", prompt="p", cache_key_fields={"k": 2},
                )

    def test_return_cost_reports_real_cost_on_miss_and_zero_on_hit(self):
        key = {"k": "cost"}
        with patch.object(llm_client.litellm, "completion", return_value=_Response('{"ok": true}')), \
             patch.object(llm_client.litellm, "completion_cost", return_value=0.0042):
            result, cost = llm_client.call(
                model="ollama/x", system="s", prompt="p", cache_key_fields=key, return_cost=True,
            )
        self.assertEqual(result, {"ok": True})
        self.assertAlmostEqual(cost, 0.0042)
        with patch.object(llm_client.litellm, "completion") as mocked:
            result2, cost2 = llm_client.call(
                model="ollama/x", system="s", prompt="p", cache_key_fields=key, return_cost=True,
            )
        mocked.assert_not_called()
        self.assertEqual(cost2, 0.0)


class CallPydanticModeTests(LlmClientCacheIsolatedTests):
    def test_pydantic_round_trip_through_cache(self):
        payload = json.dumps({"label": "idiome", "confidence": 0.9})
        with patch.object(llm_client.litellm, "completion", return_value=_Response(payload)):
            result = llm_client.call(
                model="ollama/x", system="s", prompt="p",
                response_model=_Decision, cache_key_fields={"k": 3},
            )
        self.assertIsInstance(result, _Decision)
        self.assertEqual(result.label, "idiome")
        with patch.object(llm_client.litellm, "completion") as mocked:
            cached = llm_client.call(
                model="ollama/x", system="s", prompt="p",
                response_model=_Decision, cache_key_fields={"k": 3},
            )
        mocked.assert_not_called()
        self.assertEqual(cached.confidence, 0.9)

    def test_validation_error_becomes_llm_error(self):
        with patch.object(llm_client.litellm, "completion",
                          return_value=_Response('{"label": "x"}')):  # confidence manquant
            with self.assertRaises(llm_client.LLMError):
                llm_client.call(
                    model="ollama/x", system="s", prompt="p",
                    response_model=_Decision, cache_key_fields={"k": 4},
                )


class CallBatchCompletionTests(LlmClientCacheIsolatedTests):
    def test_mixes_cache_hit_and_miss_and_aggregates_cost(self):
        hit_key = {"task_id": "t", "id": "a"}
        cache_file = llm_client.cache_path_for(hit_key, prefix="x_")
        cache_file.write_text(_Decision(label="cached", confidence=0.5).model_dump_json(), encoding="utf-8")

        items = [
            llm_client.BatchItem(system="s", user="a", cache_key_fields=hit_key, cache_prefix="x_"),
            llm_client.BatchItem(system="s", user="b", cache_key_fields={"task_id": "t", "id": "b"}, cache_prefix="x_"),
        ]
        payload = json.dumps({"label": "fresh", "confidence": 0.8})
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[_Response(payload)]) as mocked, \
             patch.object(llm_client.litellm, "completion_cost", return_value=0.001):
            results, cost = llm_client.call_batch_completion(
                items, model="openai/gpt-5-mini", response_model=_Decision, max_workers=2,
            )
        self.assertEqual(mocked.call_args.kwargs["max_workers"], 2)
        self.assertEqual(len(mocked.call_args.kwargs["messages"]), 1)  # un seul miss envoyé
        self.assertEqual(results[0].label, "cached")
        self.assertEqual(results[1].label, "fresh")
        self.assertAlmostEqual(cost, 0.001)

    def test_per_item_exception_reported_and_others_still_processed(self):
        items = [
            llm_client.BatchItem(system="s", user="a", cache_key_fields={"id": "a"}, cache_prefix="y_"),
            llm_client.BatchItem(system="s", user="b", cache_key_fields={"id": "b"}, cache_prefix="y_"),
        ]
        payload = json.dumps({"label": "ok", "confidence": 0.5})
        errors = []
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[RuntimeError("dead"), _Response(payload)]), \
             patch.object(llm_client.litellm, "completion_cost", return_value=0.0):
            results, _ = llm_client.call_batch_completion(
                items, model="openai/gpt-5-mini", response_model=_Decision,
                on_error=lambda i, item, exc: errors.append((i, exc)),
            )
        self.assertIsNone(results[0])
        self.assertEqual(results[1].label, "ok")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], 0)

    def test_all_cached_never_calls_batch_completion(self):
        key = {"id": "only"}
        cache_file = llm_client.cache_path_for(key, prefix="z_")
        cache_file.write_text(_Decision(label="cached", confidence=0.1).model_dump_json(), encoding="utf-8")
        items = [llm_client.BatchItem(system="s", user="a", cache_key_fields=key, cache_prefix="z_")]
        with patch.object(llm_client.litellm, "batch_completion") as mocked:
            results, cost = llm_client.call_batch_completion(
                items, model="openai/gpt-5-mini", response_model=_Decision,
            )
        mocked.assert_not_called()
        self.assertEqual(cost, 0.0)
        self.assertEqual(results[0].label, "cached")


class IsAvailableTests(unittest.TestCase):
    def test_ollama_ping_uses_configured_url(self):
        with patch.object(config, "OLLAMA_URL", "http://ollama-host"), \
             patch("pipeline.llm_client.urllib.request.urlopen") as opened:
            opened.return_value.__enter__.return_value.status = 200
            self.assertTrue(llm_client.is_available(backend="ollama"))
            request = opened.call_args.args[0]
            self.assertEqual(request.full_url, "http://ollama-host/api/tags")

    def test_unreachable_returns_false(self):
        with patch("pipeline.llm_client.urllib.request.urlopen", side_effect=OSError("down")):
            self.assertFalse(llm_client.is_available(backend="ollama"))


if __name__ == "__main__":
    unittest.main()
