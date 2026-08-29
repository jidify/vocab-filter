from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline import sense_fr
from pipeline.llm_tasks import TaskLlmConfig


def _task(task_id: str, model: str = "ollama/mistral-small:24b") -> TaskLlmConfig:
    provider, bare_model = model.split("/", 1)
    return TaskLlmConfig(
        task_id=task_id, batch_allowed=False, model=model, provider=provider,
        bare_model=bare_model, mode_batch=False, batch_size=1,
    )


class CollectTargetsMweIdentityTests(unittest.TestCase):
    def test_mwe_target_carries_pos_through(self):
        """S6-1 : identité complète — le POS calculé par S3/S4
        (mwe_judge.py/select.py, ex. "VERB") se perdait entre
        score.build_mwe_units() et le target consommé par S6 (frontier/
        reassign ne recevaient jamais que pos=None pour toute MWE)."""
        fake_mwe_unit = {
            "canonical_form": "give out", "pos": "VERB", "sense_id": "phrasal_verb",
            "unit_key": "mwe:give out:phrasal_verb", "occurrences": 1,
            "definition_en": "To break down.", "label": "phrasal_verb",
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[]), \
             patch("pipeline.score.build_mwe_units", return_value=[fake_mwe_unit]):
            targets = sense_fr.collect_targets()

        self.assertEqual(targets["mwe:give out:phrasal_verb"]["pos"], "VERB")

    def test_falls_back_to_none_when_unit_has_no_pos(self):
        fake_mwe_unit = {
            "canonical_form": "wing it", "pos": None, "sense_id": "idiome",
            "unit_key": "mwe:wing it:idiome", "occurrences": 1,
            "definition_en": None, "label": "idiome",
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[]), \
             patch("pipeline.score.build_mwe_units", return_value=[fake_mwe_unit]):
            targets = sense_fr.collect_targets()

        self.assertIsNone(targets["mwe:wing it:idiome"]["pos"])


class ClassifyMweKeyPosTests(unittest.TestCase):
    def test_uses_target_pos_instead_of_hardcoded_none(self):
        target = {
            "key": "mwe:give out:phrasal_verb", "lemmas_en": ["give out"],
            "occurrences": 1, "definition_en": "To break down.", "pos": "VERB",
        }
        with patch("pipeline.sense_fr.llm_is_available", return_value=False):
            entry = sense_fr.classify_mwe_key(target)
        self.assertEqual(entry["pos"], "VERB")


class LlmTranslateVotesTaskSlotTests(unittest.TestCase):
    """Lot M6 : S6-translate-local doit résoudre son modèle/provider via
    task_config("S6-translate-local"), pas via le backend global implicite
    de llm.call_json — même si, par défaut (registre M1, global_model_fallback),
    la valeur résolue reste identique au comportement actuel."""

    def test_uses_dedicated_task_slot_model_and_provider(self):
        task = _task("S6-translate-local", model="catgpt/dict-translator")
        with patch("pipeline.sense_fr.llm_is_available", return_value=True), \
             patch.object(sense_fr, "task_config", return_value=task) as task_config_mock, \
             patch.object(sense_fr.llm_client, "call",
                          return_value={"fr": "battre", "fr_alt": []}) as call:
            sense_fr.llm_translate_votes(["beat"], "v", "to hit repeatedly", [])

        task_config_mock.assert_called_with("S6-translate-local")
        self.assertEqual(call.call_args.kwargs["model"], "catgpt/dict-translator")

    def test_cache_metadata_carries_task_id_and_unit_mode(self):
        task = _task("S6-translate-local")
        with patch("pipeline.sense_fr.llm_is_available", return_value=True), \
             patch.object(sense_fr, "task_config", return_value=task), \
             patch.object(sense_fr.llm_client, "call",
                          return_value={"fr": "battre", "fr_alt": []}) as call:
            sense_fr.llm_translate_votes(["beat"], "v", "to hit repeatedly", [])

        meta = call.call_args.kwargs["cache_key_fields"]["extra"]
        self.assertEqual(meta["task_id"], "S6-translate-local")
        self.assertFalse(meta["mode_batch"])
        self.assertEqual(meta["batch_size"], 1)


class LlmBacktranslateTaskSlotTests(unittest.TestCase):
    def test_uses_dedicated_task_slot_model_and_provider(self):
        task = _task("S6-backtranslate-local", model="catgpt/dict-backtranslator")
        with patch("pipeline.sense_fr.llm_is_available", return_value=True), \
             patch.object(sense_fr, "task_config", return_value=task) as task_config_mock, \
             patch.object(sense_fr.llm_client, "call", return_value={"en": "to beat"}) as call:
            sense_fr.llm_backtranslate("battre", "to hit repeatedly")

        task_config_mock.assert_called_with("S6-backtranslate-local")
        self.assertEqual(call.call_args.kwargs["model"], "catgpt/dict-backtranslator")

    def test_cache_metadata_carries_task_id_and_unit_mode(self):
        task = _task("S6-backtranslate-local")
        with patch("pipeline.sense_fr.llm_is_available", return_value=True), \
             patch.object(sense_fr, "task_config", return_value=task), \
             patch.object(sense_fr.llm_client, "call", return_value={"en": "to beat"}) as call:
            sense_fr.llm_backtranslate("battre", "to hit repeatedly")

        meta = call.call_args.kwargs["cache_key_fields"]["extra"]
        self.assertEqual(meta["task_id"], "S6-backtranslate-local")
        self.assertFalse(meta["mode_batch"])
        self.assertEqual(meta["batch_size"], 1)


class LlmIsAvailablePingsResolvedTaskBackendTests(unittest.TestCase):
    """S6-translate-local et S6-backtranslate-local partagent
    global_model_fallback=True, qui honore l'alias PROVIDER=chatgpt->catgpt
    (plan §3.3). Le ping de disponibilité mémoïsé (une seule fois par
    process, voir la docstring de _llm_available) doit interroger le
    backend RÉSOLU par task_config, pas config.LLM_BACKEND brut — sinon
    il peut pinger Ollama alors que les appels réels partent vers CatGPT
    (ou l'inverse), un vrai .env avec PROVIDER=chatgpt suffit à le révéler."""

    def setUp(self):
        sense_fr._llm_available = None
        self.addCleanup(setattr, sense_fr, "_llm_available", None)

    def test_pings_the_provider_resolved_by_task_config(self):
        task = _task("S6-translate-local", model="catgpt/gateway-model")
        with patch.object(sense_fr, "task_config", return_value=task), \
             patch.object(sense_fr.llm_client, "is_available", return_value=True) as ping:
            self.assertTrue(sense_fr.llm_is_available())
        ping.assert_called_once_with(backend="catgpt")


class LocalTaskSlotsNoRegressionWhenEnvEmptyTests(unittest.TestCase):
    """§0/§3.4 du plan : sans AUCUNE variable dédiée ni globale (env
    vraiment vide, pas seulement les deux slots "-local"), le modèle résolu
    pour les chemins locaux doit rester le défaut de registre actuel —
    identique au comportement du pipeline avant ce lot. On ne compare pas à
    `config.LLM_BACKEND`/`config.llm_model()` : ces globals n'honorent PAS
    l'alias PROVIDER=chatgpt->catgpt (plan §3.3), contrairement à
    task_config() — un vrai `.env` de ce dépôt posant PROVIDER=chatgpt fait
    diverger les deux résolutions si on ne vide pas aussi cet alias ici."""

    _ENV_KEYS_TO_CLEAR = (
        "VOCAB_LLM_S6_TRANSLATE_LOCAL", "VOCAB_LLM_S6_BACKTRANSLATE_LOCAL",
        "VOCAB_LLM_BACKEND", "PROVIDER", "OLLAMA_MODEL", "CATGPT_MODEL",
    )

    def _clean_env(self):
        import os
        clean = {k: v for k, v in os.environ.items() if k not in self._ENV_KEYS_TO_CLEAR}
        return patch.dict("os.environ", clean, clear=True)

    def test_translate_local_falls_back_to_default_registry_model_when_env_empty(self):
        from pipeline.llm_tasks import task_config as real_task_config

        with self._clean_env():
            task = real_task_config("S6-translate-local")
        self.assertEqual(task.model, "ollama/mistral-small:24b")
        self.assertFalse(task.mode_batch)
        self.assertEqual(task.batch_size, 1)

    def test_backtranslate_local_falls_back_to_default_registry_model_when_env_empty(self):
        from pipeline.llm_tasks import task_config as real_task_config

        with self._clean_env():
            task = real_task_config("S6-backtranslate-local")
        self.assertEqual(task.model, "ollama/mistral-small:24b")
        self.assertFalse(task.mode_batch)
        self.assertEqual(task.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
