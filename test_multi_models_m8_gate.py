"""Lot M8 — gate final (plan §5 Lot M8) : matrice manuelle minimale, « au
moins un run mock par provider ». Historique : à l'écriture de ce fichier,
S3/S5/S6-*-local passaient encore par l'ancien pipeline/llm.py (couvert
alors par test_llm_backends.py) et seules les 4 tâches S6 batchées
passaient par LiteLLM — les deux clients ont depuis fusionné dans
pipeline/llm_client.py (Lot U du plan d'unification, voir
fix_pipeline/multi_models/report_multi_models.md §4bis) ; test_llm_client.py
couvre désormais les 3 providers pour tout le registre. Ce fichier reste
utile pour son deuxième volet : ollama n'était jamais exercé à travers un
appel litellm réellement mocké côté S6 batché — ce fichier ferme cet écart.
Complète aussi le premier item de la checklist : « chaque tâche production a
un modèle résolu »."""

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

    Ces 4 chemins stockent désormais en unitaire via pipeline/llm_store.py
    (plan de décorrélation lot/stockage — llm_client.run_units), pas un cache
    disque par prompt de lot entier. Base isolée par test pour ne jamais
    polluer data/llm_results.sqlite3 ni dépendre d'un run précédent."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            config, "LLM_RESULTS_DB_PATH", Path(self._tmp.name) / "llm_results.sqlite3",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_frontier_batch_completion_accepts_ollama_model(self):
        item = (_target(), [], [])
        payload = {"sense_id": "x", "fr": ["battre"], "translation_type": "equivalence_directe",
                   "sense_fit": "ok", "sense_fit_note": "",
                   "definition_fr_fit": "ok", "definition_fr_fit_note": "",
                   "source": "reecrit", "confidence": "high"}
        with patch.object(frontier.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(payload)]) as mocked:
            got, _ = frontier._translate_units(
                [item], "ollama/mistral-small:24b", mode_batch=False, batch_size=1,
            )
        self.assertEqual(mocked.call_args.kwargs["model"], "ollama/mistral-small:24b")
        self.assertEqual(got["x"].fr, ["battre"])

    def test_reassign_batch_completion_accepts_ollama_model(self):
        item = (_target(), [], [])
        decision = {"key": "x", "pos": "n", "sense_id": "x.n.01", "fr": ["mot"],
                    "translation_type": "equivalence_directe", "sense_fit": "ok",
                    "sense_fit_note": "",
                    "definition_fr_fit": "ok", "definition_fr_fit_note": "",
                    "confidence": "high", "reason": "ok"}
        with patch.object(reassign.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(decision)]) as mocked:
            got, _ = reassign._translate_units(
                [item], "ollama/mistral-small:24b", mode_batch=False, batch_size=1,
            )
        self.assertEqual(mocked.call_args.kwargs["model"], "ollama/mistral-small:24b")
        self.assertEqual(got["x"].sense_id, "x.n.01")

    def test_adjudication_backtranslate_completion_accepts_ollama_model(self):
        one = {"key": "x", "en": "thing"}
        with patch.object(adjudicate.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(one)]) as mocked:
            got = adjudicate._backtranslate_units(
                [{"key": "x", "fr": "mot", "definition_en": "thing"}],
                "ollama/mistral-small:24b", mode_batch=False, batch_size=1,
            )
        self.assertEqual(mocked.call_args.kwargs["model"], "ollama/mistral-small:24b")
        self.assertEqual(got, {"x": "thing"})

    def test_adjudication_judge_completion_accepts_ollama_model(self):
        verdict = {"key": "x", "fr": "mot", "fr_alt": [], "confidence": "high",
                   "reason": "ok", "no_equivalent": False}
        with patch.object(adjudicate.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response({"verdicts": [verdict]})]) as mocked:
            got = adjudicate._judge_units(
                [{"key": "x", "fr": "mot", "definition_en": "thing"}], {}, {},
                "ollama/mistral-small:24b", batch_size=20, mode_batch=True,
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
