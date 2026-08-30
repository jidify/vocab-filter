"""Remplace l'ancien contenu de ce fichier (4 digests figés sur le cache
disque `pipeline_out/cache/{frontier,reassign,backtranslate,judge}_*.json`,
byte-figé avant le plan de décorrélation lot/stockage — Lot U3 du plan
d'unification, fix_pipeline/multi_models/report_multi_models.md §4bis). Ce
cache ne stocke plus rien pour ces 4 tâches : elles passent désormais par
`llm_client.run_units` (stockage unitaire, pipeline/llm_store.py), donc
`frontier._cache_path`/`reassign._cache_path`/les préfixes `backtranslate_`/
`judge_` n'existent plus — plus rien à figer là.

Ce qui remplace la garantie que ces tests protégeaient (« un cache stable,
qui ne se réinvalide pas pour rien ») : le test d'acceptation central du
plan — les mêmes unités, appelées deux fois avec des `batch_size`
différents, ne déclenchent qu'UNE seule série d'appels LLM. Voir aussi
test_llm_client_run_units.py::BatchSizeIndependenceTests (le même invariant
au niveau de `run_units` lui-même) et test_multi_models_m5_s5.py (S5-arbitrate)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import config
from pipeline import sense_fr_adjudicate as adjudicate
from pipeline import sense_fr_frontier as frontier
from pipeline import sense_fr_reassign as reassign


def _target(key="x"):
    return {"key": key, "kind": "synset", "pos": "n", "lemmas_en": ["x"], "definition_en": "a thing"}


class Response:
    def __init__(self, payload):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": json.dumps(payload)})()})()]


class BatchSizeDecorrelationTests(unittest.TestCase):
    """Base isolée de la vraie data/llm_results.sqlite3, comme test_llm_store.py."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            config, "LLM_RESULTS_DB_PATH", Path(self._tmp.name) / "llm_results.sqlite3",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_frontier_batch_size_change_triggers_zero_new_calls(self):
        items = [(_target(str(i)), [], []) for i in range(4)]
        payload = {"translations": [
            {"sense_id": str(i), "fr": ["mot"], "translation_type": "equivalence_directe",
             "sense_fit": "ok", "sense_fit_note": "",
             "definition_fr_fit": "ok", "definition_fr_fit_note": "",
             "source": "reecrit", "confidence": "high"}
            for i in range(4)
        ]}
        with patch.object(frontier.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(payload) for _ in kw["messages"]]) as mocked:
            first, _ = frontier._translate_units(items, "openai/m", batch_size=4, mode_batch=True)
        mocked.assert_called_once()
        with patch.object(frontier.llm_client.litellm, "batch_completion") as mocked2:
            second, _ = frontier._translate_units(items, "openai/m", batch_size=2, mode_batch=True)
        mocked2.assert_not_called()
        self.assertEqual({k: v.fr for k, v in second.items()}, {k: v.fr for k, v in first.items()})

    def test_reassign_batch_size_change_triggers_zero_new_calls(self):
        items = [(_target(str(i)), [], []) for i in range(4)]
        decisions = [
            {"key": str(i), "pos": "n", "sense_id": f"{i}.n.01", "fr": ["mot"],
             "translation_type": "equivalence_directe", "sense_fit": "ok", "sense_fit_note": "",
             "definition_fr_fit": "ok", "definition_fr_fit_note": "",
             "confidence": "high", "reason": "ok"}
            for i in range(4)
        ]
        payload = {"decisions": decisions}
        with patch.object(reassign.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(payload) for _ in kw["messages"]]) as mocked:
            first, _ = reassign._translate_units(items, "openai/m", batch_size=4, mode_batch=True)
        mocked.assert_called_once()
        with patch.object(reassign.llm_client.litellm, "batch_completion") as mocked2:
            second, _ = reassign._translate_units(items, "openai/m", batch_size=2, mode_batch=True)
        mocked2.assert_not_called()
        self.assertEqual({k: v.sense_id for k, v in second.items()}, {k: v.sense_id for k, v in first.items()})

    def test_backtranslate_batch_size_change_triggers_zero_new_calls(self):
        entries = [{"key": str(i), "fr": "mot", "definition_en": "thing"} for i in range(4)]
        payload = {"guesses": [{"key": str(i), "en": "thing"} for i in range(4)]}
        with patch.object(adjudicate.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(payload) for _ in kw["messages"]]) as mocked:
            first = adjudicate._backtranslate_units(entries, "openai/m", batch_size=4, mode_batch=True)
        mocked.assert_called_once()
        with patch.object(adjudicate.llm_client.litellm, "batch_completion") as mocked2:
            second = adjudicate._backtranslate_units(entries, "openai/m", batch_size=2, mode_batch=True)
        mocked2.assert_not_called()
        self.assertEqual(second, first)

    def test_judge_dossier_batch_size_change_triggers_zero_new_calls(self):
        entries = [{"key": str(i), "fr": "mot", "definition_en": "thing"} for i in range(4)]
        payload = {"verdicts": [
            {"key": str(i), "fr": "mot", "fr_alt": [], "confidence": "high", "reason": "ok", "no_equivalent": False}
            for i in range(4)
        ]}
        with patch.object(adjudicate.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(payload) for _ in kw["messages"]]) as mocked:
            first = adjudicate._judge_units(entries, {}, {}, "openai/m", batch_size=4, mode_batch=True)
        mocked.assert_called_once()
        with patch.object(adjudicate.llm_client.litellm, "batch_completion") as mocked2:
            second = adjudicate._judge_units(entries, {}, {}, "openai/m", batch_size=2, mode_batch=True)
        mocked2.assert_not_called()
        self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()
