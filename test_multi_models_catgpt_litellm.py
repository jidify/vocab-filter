"""Adaptateur LiteLLM -> CatGPT-Gateway (pipeline/llm_litellm_catgpt.py) :
ferme l'écart documenté par report_multi_models.md §2.1 — `catgpt/*`
passait la validation de config mais échouait à l'appel réel pour les 4
tâches S6 routées par LiteLLM. Trois volets couverts ici :

1. `call_kwargs()` n'a d'effet que pour un modèle `catgpt/...` — invariant
   explicitement demandé : les modèles `json_schema` natifs (openai/*)
   restent inchangés (test de non-régression).
2. Bout en bout mocké au niveau HTTP (`urllib.request.urlopen`) : le
   schéma JSON demandé via `response_format=<PydanticModel>` est bien
   injecté dans le message système envoyé au gateway, en `json_object`.
3. Une erreur réseau ne fait pas planter tout le lot — même chemin de
   dégradation que pour ollama/openai (`isinstance(response, Exception)`).
"""

from __future__ import annotations

import json as _json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import litellm

from pipeline import config, llm_litellm_catgpt
from pipeline import sense_fr_adjudicate as adjudicate
from pipeline import sense_fr_frontier as frontier


def _target(key="x"):
    return {"key": key, "kind": "synset", "pos": "n", "lemmas_en": ["x"], "definition_en": "a thing"}


class Response:
    """Même doublure que test_multi_models_m8_gate.py::Response."""

    def __init__(self, payload):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": _json.dumps(payload)})()})()]


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = _json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class _ProviderMapIsolation(unittest.TestCase):
    """`litellm.custom_provider_map` est un état module-level partagé —
    restauré après chaque test pour ne pas influencer les autres fichiers
    de la suite (déjà exécutée en `discover`, ordre non garanti)."""

    def setUp(self):
        self._orig_map = list(litellm.custom_provider_map)
        self.addCleanup(lambda: litellm.custom_provider_map.__setitem__(
            slice(None), self._orig_map))


class CallKwargsTests(_ProviderMapIsolation):
    def test_catgpt_model_registers_provider_and_allows_reasoning_effort(self):
        kwargs = llm_litellm_catgpt.call_kwargs("catgpt/catgpt-browser")
        self.assertEqual(kwargs, {"allowed_openai_params": ["reasoning_effort"]})
        self.assertTrue(any(item["provider"] == "catgpt" for item in litellm.custom_provider_map))

    def test_call_kwargs_is_idempotent(self):
        llm_litellm_catgpt.call_kwargs("catgpt/catgpt-browser")
        before = len(litellm.custom_provider_map)
        llm_litellm_catgpt.call_kwargs("catgpt/catgpt-browser")
        self.assertEqual(len(litellm.custom_provider_map), before)

    def test_non_catgpt_model_is_untouched(self):
        before = list(litellm.custom_provider_map)
        for model in ("openai/gpt-5-mini", "ollama/mistral-small:24b"):
            with self.subTest(model=model):
                self.assertEqual(llm_litellm_catgpt.call_kwargs(model), {})
        self.assertEqual(litellm.custom_provider_map, before)


class NoRegressionOnJsonSchemaModelsTests(_ProviderMapIsolation):
    """Invariant explicite du plan : les providers qui savent faire du
    json_schema natif (openai/*) doivent continuer à le recevoir
    exactement comme avant ce lot — aucun kwarg supplémentaire, aucun
    schéma dupliqué dans le prompt."""

    def setUp(self):
        super().setUp()
        # frontier._translate_batches / adjudicate._backtranslate_batch
        # écrivent un vrai cache disque indexé sur {model, system, user} —
        # isolé pour ne pas dépendre (ni interférer avec) un fichier déjà
        # écrit par un autre test de la suite avec le même modèle/prompt
        # (piège documenté dans test_multi_models_m8_gate.py).
        self._tmp_cache = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_cache.cleanup)
        patcher = patch.object(config, "CACHE_DIR", Path(self._tmp_cache.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_frontier_openai_model_keeps_pydantic_response_format(self):
        item = (_target(), [], [])
        payload = {"sense_id": "x", "fr": ["battre"], "translation_type": "equivalence_directe",
                   "sense_fit": "ok", "sense_fit_note": "", "source": "reecrit", "confidence": "high"}
        with patch.object(frontier.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(payload)]) as mocked:
            frontier._translate_batches(
                [[item]], "openai/gpt-5-mini", mode_batch=False, batch_size=1,
            )
        kwargs = mocked.call_args.kwargs
        self.assertIs(kwargs["response_format"], frontier.UnitTranslation)
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertNotIn("allowed_openai_params", kwargs)

    def test_adjudicate_backtranslate_openai_model_keeps_pydantic_response_format(self):
        with patch.object(litellm, "completion",
                          return_value=Response({"key": "x", "en": "thing"})) as mocked:
            adjudicate._backtranslate_batch(
                [{"key": "x", "fr": "mot", "definition_en": "thing"}],
                "openai/gpt-5-mini", mode_batch=False,
            )
        kwargs = mocked.call_args.kwargs
        self.assertIs(kwargs["response_format"], adjudicate._Guess)
        self.assertNotIn("allowed_openai_params", kwargs)


class CatgptEndToEndTests(_ProviderMapIsolation):
    """Chemin réel (litellm.batch_completion/completion non mockés) —
    seul `urllib.request.urlopen` est mocké, même principe que
    test_llm_client.py pour pipeline/llm_client.py."""

    def setUp(self):
        super().setUp()
        self._tmp_cache = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_cache.cleanup)
        patcher = patch.object(config, "CACHE_DIR", Path(self._tmp_cache.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_frontier_catgpt_injects_schema_and_posts_json_object(self):
        items = [(_target("x"), [], []), (_target("y"), [], [])]
        payload = {"translations": [
            {"sense_id": "x", "fr": ["battre"], "translation_type": "equivalence_directe",
             "sense_fit": "ok", "sense_fit_note": "", "source": "reecrit", "confidence": "high"},
            {"sense_id": "y", "fr": ["frapper"], "translation_type": "equivalence_directe",
             "sense_fit": "ok", "sense_fit_note": "", "source": "reecrit", "confidence": "high"},
        ]}
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            captured["body"] = _json.loads(request.data.decode("utf-8"))
            return _FakeHTTPResponse({"choices": [{"message": {"content": _json.dumps(payload)}}]})

        with patch("pipeline.llm_litellm_catgpt.urllib.request.urlopen", side_effect=fake_urlopen):
            got, _ = frontier._translate_batches(
                [items], "catgpt/catgpt-browser", mode_batch=True, batch_size=2,
            )

        self.assertEqual(captured["url"], f"{config.CATGPT_BASE_URL}/chat/completions")
        self.assertEqual(captured["headers"].get("authorization"), f"Bearer {config.CATGPT_API_TOKEN}")
        self.assertEqual(captured["body"]["response_format"], {"type": "json_object"})
        system_msg = next(m["content"] for m in captured["body"]["messages"] if m["role"] == "system")
        self.assertIn('"translations"', system_msg)
        self.assertIn('"translation_type"', system_msg)
        self.assertEqual(got[0]["x"].fr, ["battre"])
        self.assertEqual(got[0]["y"].fr, ["frapper"])

    def test_frontier_catgpt_network_error_drops_batch_without_crashing(self):
        item = (_target(), [], [])
        with patch("pipeline.llm_litellm_catgpt.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            got, cost = frontier._translate_batches(
                [[item]], "catgpt/catgpt-browser", mode_batch=False, batch_size=1,
            )
        self.assertEqual(got, [{}])
        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
