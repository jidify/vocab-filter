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

    def test_mwe_target_carries_definition_needs_review_through(self):
        fake_mwe_unit = {
            "canonical_form": "get a grip", "pos": "OTHER", "sense_id": "mwe-custom-v1:x",
            "unit_key": "mwe:get a grip:other:mwe-custom-v1:x", "occurrences": 1,
            "definition_en": "To come to one's senses.", "label": "idiome",
            "definition_needs_review": True,
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[]), \
             patch("pipeline.score.build_mwe_units", return_value=[fake_mwe_unit]):
            targets = sense_fr.collect_targets()

        self.assertTrue(targets["mwe:get a grip:other:mwe-custom-v1:x"]["definition_needs_review"])

    def test_defaults_definition_needs_review_to_false_when_absent(self):
        fake_mwe_unit = {
            "canonical_form": "give out", "pos": "VERB", "sense_id": "fail.v.04",
            "unit_key": "mwe:give out:verb:fail.v.04", "occurrences": 1,
            "definition_en": "To stop operating.", "label": "phrasal_verb",
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[]), \
             patch("pipeline.score.build_mwe_units", return_value=[fake_mwe_unit]):
            targets = sense_fr.collect_targets()

        self.assertFalse(targets["mwe:give out:verb:fail.v.04"]["definition_needs_review"])


class CollectTargetsManualCorrectionMweKindTests(unittest.TestCase):
    """S6-1 : une clé "mwe:..." atteinte via une correction manuelle
    (score.py::load_manual_corrections, ex. "mwe:pig smash:semi_fige", un
    jeu inventé par l'auteur sans aucune entrée WordNet) doit être classée
    kind="mwe", jamais kind="synset" par défaut — sinon
    collect_frontier_targets() tente nwn.synset("mwe:pig smash:semi_fige"),
    échoue, et la route silencieusement vers "sense_id_non_resolu" pour
    toujours, sans jamais appeler le modèle (cas réel mesuré : "beat" et
    "pig smash" bloqués ainsi dans data/sense_fr.jsonl avant ce correctif)."""

    def test_mwe_style_sense_id_from_manual_correction_is_kind_mwe(self):
        fake_word_unit = {
            "canonical_form": "pig smash", "sense_id": "mwe:pig smash:semi_fige",
            "occurrences": 1, "pos": None, "definition_en": "A game name invented by the author.",
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[fake_word_unit]), \
             patch("pipeline.score.build_mwe_units", return_value=[]):
            targets = sense_fr.collect_targets()

        target = targets["mwe:pig smash:semi_fige"]
        self.assertEqual(target["kind"], "mwe")
        self.assertEqual(target["definition_en"], "A game name invented by the author.")

    def test_real_wordnet_sense_id_stays_kind_synset(self):
        fake_word_unit = {
            "canonical_form": "attend", "sense_id": "attend.v.02",
            "occurrences": 1, "pos": "v", "definition_en": "take charge of or deal with",
        }
        with patch("pipeline.score.build_records", return_value=[]), \
             patch("pipeline.score.aggregate_and_score", return_value=[fake_word_unit]), \
             patch("pipeline.score.build_mwe_units", return_value=[]):
            targets = sense_fr.collect_targets()

        self.assertEqual(targets["attend.v.02"]["kind"], "synset")


class BlocksAutoLockExtendedAxesTests(unittest.TestCase):
    def test_definition_fr_fit_contradiction_blocks_even_with_ok_sense_fit(self):
        self.assertEqual(
            sense_fr.blocks_auto_lock("ok", "equivalence_directe", definition_fr_fit="contradiction"),
            "definition_fr_contradiction",
        )

    def test_definition_needs_review_blocks_even_with_ok_everything(self):
        self.assertEqual(
            sense_fr.blocks_auto_lock(
                "ok", "equivalence_directe", definition_fr_fit="ok", definition_needs_review=True,
            ),
            "definition_non_validee",
        )

    def test_all_ok_and_reviewed_never_blocks(self):
        self.assertIsNone(sense_fr.blocks_auto_lock(
            "ok", "equivalence_directe", definition_fr_fit="ok", definition_needs_review=False,
        ))


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
