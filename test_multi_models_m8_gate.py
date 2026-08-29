"""Lot M8 — gate final (plan §5 Lot M8) : matrice manuelle minimale, « au
moins un run mock par provider ». pipeline/llm.py (S3, S5, S6-*-local) est
déjà couvert pour ollama/catgpt/openai par test_llm_backends.py (3 tests, un
par provider). Côté LiteLLM (S6-translate-frontier, S6-backtranslate,
S6-judge-dossier, S6-reassign), test_multi_models_m2_s6.py exerçait déjà
openai/catgpt (build_entry/apply_decision, provenance du modèle) mais jamais
ollama à travers un appel litellm réellement mocké — ce fichier ferme cet
écart. Complète aussi le premier item de la checklist : « chaque tâche
production a un modèle résolu »."""

from __future__ import annotations

import json as _json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import litellm

from pipeline import config
from pipeline import sense_fr_adjudicate as adjudicate
from pipeline import sense_fr_frontier as frontier
from pipeline import sense_fr_reassign as reassign
from pipeline.llm_tasks import ALLOWED_PROVIDERS, TASK_REGISTRY, task_config


def _target(key="x"):
    return {"key": key, "kind": "synset", "pos": "n", "lemmas_en": ["x"], "definition_en": "a thing"}


class Response:
    def __init__(self, payload):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": _json.dumps(payload)})()})()]


class LiteLlmProviderMatrixTests(unittest.TestCase):
    """openai et catgpt sont déjà exercés côté LiteLLM ailleurs dans la
    suite ; il manquait ollama, alors que pipeline.llm_tasks.ALLOWED_PROVIDERS
    l'autorise pour ces 4 tâches au même titre que pour les 5 autres.

    Ces 4 chemins écrivent un cache disque réel (frontier/reassign via
    _cache_path ; adjudicate inline) sous config.CACHE_DIR — voir la même
    précaution dans test_multi_models_m2_s6.py::test_frontier_translate_batches_
    batch_size_one_sends_unit_prompt. Répertoire temporaire dédié par test pour
    ne jamais polluer pipeline_out/cache/ ni dépendre d'un run précédent qui
    aurait déjà écrit ce cache (ce qui court-circuiterait silencieusement le
    mock au run suivant, comme observé lors de l'écriture de ce lot)."""

    def setUp(self):
        self._tmp_cache = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_cache.cleanup)
        patcher = patch.object(config, "CACHE_DIR", Path(self._tmp_cache.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_frontier_batch_completion_accepts_ollama_model(self):
        item = (_target(), [], [])
        payload = {"sense_id": "x", "fr": ["battre"], "translation_type": "equivalence_directe",
                   "sense_fit": "ok", "sense_fit_note": "", "source": "reecrit", "confidence": "high"}
        with patch.object(frontier.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(payload)]) as mocked:
            got, _ = frontier._translate_batches(
                [[item]], "ollama/mistral-small:24b", mode_batch=False, batch_size=1,
            )
        self.assertEqual(mocked.call_args.kwargs["model"], "ollama/mistral-small:24b")
        self.assertEqual(got[0]["x"].fr, ["battre"])

    def test_reassign_batch_completion_accepts_ollama_model(self):
        item = (_target(), [], [])
        decision = {"key": "x", "pos": "n", "sense_id": "x.n.01", "fr": ["mot"],
                    "translation_type": "equivalence_directe", "sense_fit": "ok",
                    "sense_fit_note": "", "confidence": "high", "reason": "ok"}
        with patch.object(reassign.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(decision)]) as mocked:
            got, _ = reassign._translate_batches(
                [[item]], "ollama/mistral-small:24b", mode_batch=False, batch_size=1,
            )
        self.assertEqual(mocked.call_args.kwargs["model"], "ollama/mistral-small:24b")
        self.assertEqual(got[0]["x"].sense_id, "x.n.01")

    def test_adjudication_backtranslate_completion_accepts_ollama_model(self):
        one = {"key": "x", "en": "thing"}
        with patch.object(litellm, "completion", return_value=Response(one)) as mocked:
            got = adjudicate._backtranslate_batch(
                [{"key": "x", "fr": "mot", "definition_en": "thing"}],
                "ollama/mistral-small:24b", mode_batch=False,
            )
        self.assertEqual(mocked.call_args.kwargs["model"], "ollama/mistral-small:24b")
        self.assertEqual(got, {"x": "thing"})

    def test_adjudication_judge_completion_accepts_ollama_model(self):
        verdict = {"key": "x", "fr": "mot", "fr_alt": [], "confidence": "high",
                   "reason": "ok", "no_equivalent": False}
        with patch.object(litellm, "completion", return_value=Response({"verdicts": [verdict]})) as mocked:
            got = adjudicate._judge_batch(
                {}, [{"key": "x", "fr": "mot", "definition_en": "thing"}], {},
                "ollama/mistral-small:24b", {}, mode_batch=True,
            )
        self.assertEqual(mocked.call_args.kwargs["model"], "ollama/mistral-small:24b")
        self.assertEqual(got["x"]["fr"], "mot")


class EveryProductionTaskResolvesAModelTests(unittest.TestCase):
    """Checklist M8, item 2 : « chaque tâche production a un modèle
    résolu ». Balaie tout le registre (pas seulement les tâches déjà
    nommément testées ailleurs) sous l'environnement réel du process — si
    task_config() lève pour une tâche quelconque, ce test le signale."""

    def test_every_registered_task_resolves_without_raising(self):
        for task_id in TASK_REGISTRY:
            with self.subTest(task_id=task_id):
                resolved = task_config(task_id)
                self.assertEqual(resolved.task_id, task_id)
                self.assertIn(resolved.provider, ALLOWED_PROVIDERS)
                self.assertTrue(resolved.bare_model)
                self.assertGreaterEqual(resolved.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
