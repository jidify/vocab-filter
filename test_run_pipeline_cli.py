"""Lot M6 : le CLI (--llm-backend/--llm-model) et configure_llm() ne doivent
plus laisser croire qu'ils couvrent les 4 tâches S6 routées par LiteLLM
(S6-translate-frontier/S6-backtranslate/S6-judge-dossier/S6-reassign) —
voir fix_pipeline/multi_models/plan_multi_models.md §3.3-3.4, Lot M6."""

from __future__ import annotations

import subprocess
import sys
import unittest

from pipeline import config


class RunPipelineCliHelpDocumentsScopeTests(unittest.TestCase):
    def test_help_clarifies_llm_backend_and_llm_model_scope(self):
        result = subprocess.run(
            [sys.executable, "run_pipeline.py", "--help"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # --llm-backend et --llm-model doivent chacun documenter qu'ils ne
        # couvrent pas les tâches S6 frontière/adjudication/reassign.
        # argparse enveloppe les lignes du --help (parfois au milieu d'un mot
        # composé, ex. "S6-translate-\nfrontier") : on compare donc sur la
        # sortie débarrassée de tout espace/retour à la ligne, jamais telle quelle.
        flattened = "".join(result.stdout.split())
        self.assertIn("VOCAB_LLM_S6", flattened)
        self.assertIn("S6-translate-frontier", flattened)


class ConfigureLlmNoRegressionWhenEnvEmptyTests(unittest.TestCase):
    """Sans option CLI (tous les paramètres None, comme quand aucune
    variable d'environnement dédiée n'est posée), configure_llm() doit
    rester un no-op — comportement actuel du pipeline, inchangé par M6."""

    def test_configure_llm_is_a_noop_with_all_none(self):
        before = (config.LLM_BACKEND, config.OLLAMA_URL, config.OLLAMA_MODEL,
                  config.CATGPT_BASE_URL, config.CATGPT_API_TOKEN,
                  config.CATGPT_MODEL, config.CATGPT_TIMEOUT)
        config.configure_llm(backend=None, base_url=None, api_token=None,
                             model=None, timeout=None)
        after = (config.LLM_BACKEND, config.OLLAMA_URL, config.OLLAMA_MODEL,
                 config.CATGPT_BASE_URL, config.CATGPT_API_TOKEN,
                 config.CATGPT_MODEL, config.CATGPT_TIMEOUT)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
