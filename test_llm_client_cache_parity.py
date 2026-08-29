"""Lot U3 du plan d'unification (fix_pipeline/multi_models/report_multi_models.md
§4bis) : les 4 tâches S6 historiquement routées par LiteLLM correspondent à des
appels OpenAI déjà payés — leur cache disque ne doit PAS être invalidé par le
passage à pipeline/llm_client.py. Les digests attendus ci-dessous sont calculés
avec la formule de clé EXACTE d'avant migration (voir fix_pipeline/multi_models/
report_m2_s6.md et sense_fr_frontier.py::_cache_path avant ce lot) — un digest
différent signale une régression de cache, pas une simple différence de style."""

import unittest

from pipeline import sense_fr_frontier as frontier
from pipeline import sense_fr_reassign as reassign


class CacheKeyByteParityTests(unittest.TestCase):
    def test_frontier_cache_path_matches_pre_migration_digest(self):
        path = frontier._cache_path("openai/gpt-5-mini", "s", "u", mode_batch=True, batch_size=40)
        self.assertEqual(
            path.name,
            "frontier_f21c395236150896552eb0359cfde514a9e15c1c43cc8f173fda591f85ddc860.json",
        )

    def test_reassign_cache_path_matches_pre_migration_digest(self):
        path = reassign._cache_path("openai/gpt-5-mini", "s", "u", mode_batch=False, batch_size=1)
        self.assertEqual(
            path.name,
            "reassign_8feea2a03b3cc9cf24395aca8f96cffae348c5db0c3751630d0ffb6799799f22.json",
        )

    def test_backtranslate_cache_path_matches_pre_migration_digest(self):
        cache_key_fields = {"task_id": "S6-backtranslate", "model": "openai/gpt-5-mini",
                            "mode_batch": True, "batch_size": 40, "system": "s", "user": "u"}
        from pipeline import llm_client
        path = llm_client.cache_path_for(cache_key_fields, prefix="backtranslate_")
        self.assertEqual(
            path.name,
            "backtranslate_d65c0e307c760cf3edb6378f1371989b71985a84cc484347e690f860a32c01f1.json",
        )

    def test_judge_dossier_cache_path_matches_pre_migration_digest(self):
        cache_key_fields = {"task_id": "S6-judge-dossier", "model": "openai/gpt-5-mini",
                            "mode_batch": True, "batch_size": 20, "system": "s", "user": "u"}
        from pipeline import llm_client
        path = llm_client.cache_path_for(cache_key_fields, prefix="judge_")
        self.assertEqual(
            path.name,
            "judge_f172e9a0b2d0ca1378598011d4656addbd5576d8401ac673e190ea4e83200d48.json",
        )


if __name__ == "__main__":
    unittest.main()
