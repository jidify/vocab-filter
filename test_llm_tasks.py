import os
import unittest
from unittest.mock import patch

from pipeline.llm_tasks import (
    TASK_REGISTRY,
    TaskConfigError,
    effective_batch_size,
    task_config,
    use_batch_prompt,
)


TASK_ENV_KEYS = {
    "VOCAB_LLM_BACKEND",
    "OLLAMA_MODEL",
    "CATGPT_MODEL",
    "PROVIDER",
    *(f"VOCAB_LLM_{task_id.replace('-', '_')}" for task_id in TASK_REGISTRY),
}


class LlmTaskRegistryTests(unittest.TestCase):
    def env(self, **values):
        clean = {key: value for key, value in os.environ.items() if key not in TASK_ENV_KEYS}
        clean.update(values)
        return patch.dict(os.environ, clean, clear=True)

    def test_registry_contains_m0_tasks_and_defaults(self):
        expected = {
            "S3-judge-type": (False, "ollama/mistral-small:24b", False, 1),
            "S3-judge-occurrence": (True, "ollama/mistral-small:24b", False, 1),
            "S3-definition-cluster": (True, "ollama/mistral-small:24b", False, 1),
            "S5-arbitrate": (True, "ollama/mistral-small:24b", False, 1),
            "S6-translate-frontier": (True, "openai/gpt-5-mini", True, 40),
            "S6-backtranslate": (True, "openai/gpt-5-mini", True, 40),
            "S6-judge-dossier": (True, "openai/gpt-5-mini", True, 20),
            "S6-reassign": (True, "openai/gpt-5-mini", True, 10),
            "S6-translate-local": (False, "ollama/mistral-small:24b", False, 1),
            "S6-backtranslate-local": (False, "ollama/mistral-small:24b", False, 1),
        }
        with self.env():
            actual = {
                task_id: (
                    descriptor.batch_allowed,
                    task_config(task_id).model,
                    task_config(task_id).mode_batch,
                    task_config(task_id).batch_size,
                )
                for task_id, descriptor in TASK_REGISTRY.items()
            }
        self.assertEqual(expected, actual)

    def test_parses_documented_override(self):
        with self.env(VOCAB_LLM_S6_TRANSLATE_FRONTIER="openai/gpt-5-mini;batch=true;batch_size=40"):
            cfg = task_config("S6-translate-frontier")
        self.assertEqual(("openai", "gpt-5-mini"), (cfg.provider, cfg.bare_model))
        self.assertTrue(cfg.mode_batch)
        self.assertEqual(40, cfg.batch_size)
        self.assertEqual(40, effective_batch_size(cfg))

    def test_batchable_tasks_support_unit_and_batch_config_paths(self):
        for task_id, descriptor in TASK_REGISTRY.items():
            if not descriptor.batch_allowed:
                continue
            env_key = f"VOCAB_LLM_{task_id.replace('-', '_')}"
            with self.subTest(task_id=task_id, mode="unit"), self.env(**{env_key: f"{descriptor.default_model};batch=false"}):
                self.assertEqual(1, effective_batch_size(task_config(task_id)))
            with self.subTest(task_id=task_id, mode="batch"), self.env(**{env_key: f"{descriptor.default_model};batch=true;batch_size=2"}):
                self.assertEqual(2, effective_batch_size(task_config(task_id)))

    def test_global_backend_and_models_are_fallback_for_s3_s5(self):
        with self.env(VOCAB_LLM_BACKEND="catgpt", CATGPT_MODEL="custom-browser"):
            for task_id in ("S3-judge-occurrence", "S3-definition-cluster", "S5-arbitrate"):
                self.assertEqual("catgpt/custom-browser", task_config(task_id).model)
        with self.env(VOCAB_LLM_BACKEND="ollama", OLLAMA_MODEL="gemma3:27b"):
            self.assertEqual("ollama/gemma3:27b", task_config("S5-arbitrate").model)

    def test_provider_chatgpt_alias_selects_catgpt_global_fallback(self):
        with self.env(PROVIDER="chatgpt", CATGPT_MODEL="gateway-model"):
            cfg = task_config("S3-judge-occurrence")
        self.assertEqual(("catgpt", "gateway-model"), (cfg.provider, cfg.bare_model))

    def test_dedicated_override_wins_over_global_fallback(self):
        with self.env(
            VOCAB_LLM_BACKEND="ollama",
            OLLAMA_MODEL="global",
            VOCAB_LLM_S5_ARBITRATE="catgpt/dedicated;batch=false",
        ):
            self.assertEqual("catgpt/dedicated", task_config("S5-arbitrate").model)

    def test_rejects_batch_for_non_batchable_task(self):
        with self.env(VOCAB_LLM_S6_TRANSLATE_LOCAL="ollama/model;batch=true;batch_size=2"):
            with self.assertRaisesRegex(TaskConfigError, "batch_allowed=false"):
                task_config("S6-translate-local")

    def test_rejects_batch_override_without_valid_size(self):
        for value in ("openai/model;batch=true", "openai/model;batch=true;batch_size=0", "openai/model;batch=true;batch_size=nope"):
            with self.subTest(value=value), self.env(VOCAB_LLM_S6_REASSIGN=value):
                with self.assertRaises(TaskConfigError):
                    task_config("S6-reassign")

    def test_use_batch_prompt_false_when_effective_batch_size_is_one(self):
        # Régression du défaut n°3 (revue M2) : batch=true;batch_size=1 doit
        # être traité comme le chemin unitaire par les quatre tâches S6 déjà
        # batchées, pas comme un lot d'un seul item.
        for task_id in ("S6-translate-frontier", "S6-backtranslate", "S6-judge-dossier", "S6-reassign"):
            env_key = f"VOCAB_LLM_{task_id.replace('-', '_')}"
            with self.subTest(task_id=task_id), self.env(**{env_key: "openai/m;batch=true;batch_size=1"}):
                self.assertFalse(use_batch_prompt(task_config(task_id)))
                self.assertFalse(use_batch_prompt(task_id))

    def test_use_batch_prompt_true_at_default_batch_size(self):
        for task_id in ("S6-translate-frontier", "S6-backtranslate", "S6-judge-dossier", "S6-reassign"):
            with self.subTest(task_id=task_id), self.env():
                self.assertTrue(use_batch_prompt(task_id))

    def test_use_batch_prompt_honours_explicit_size_override(self):
        with self.env():
            task = task_config("S6-reassign")
            self.assertTrue(use_batch_prompt(task))
            self.assertFalse(use_batch_prompt(task, batch_size=1))

    def test_prompt_option_resolves_named_variant(self):
        with self.env(VOCAB_LLM_S3_JUDGE_OCCURRENCE="catgpt/x;prompt=s3-occurrence-tags"):
            cfg = task_config("S3-judge-occurrence")
        self.assertIsNotNone(cfg.custom_prompt)
        self.assertEqual(cfg.custom_prompt.schema_variant, "tags")

    def test_unknown_prompt_variant_raises(self):
        with self.env(VOCAB_LLM_S3_JUDGE_OCCURRENCE="catgpt/x;prompt=does-not-exist"):
            with self.assertRaises(TaskConfigError):
                task_config("S3-judge-occurrence")

    def test_no_prompt_option_leaves_custom_prompt_none(self):
        with self.env():
            cfg = task_config("S3-judge-occurrence")
        self.assertIsNone(cfg.custom_prompt)

    def test_rejects_unknown_task_provider_option_and_duplicate(self):
        with self.env():
            with self.assertRaisesRegex(TaskConfigError, "tâche LLM inconnue"):
                task_config("S9-nope")
        invalid = (
            "anthropic/model;batch=false",
            "openai/model;wat=true",
            "openai/model;batch=false;batch=false",
            ";batch=false",
        )
        for value in invalid:
            with self.subTest(value=value), self.env(VOCAB_LLM_S5_ARBITRATE=value):
                with self.assertRaises(TaskConfigError):
                    task_config("S5-arbitrate")


if __name__ == "__main__":
    unittest.main()
