import json as _json
import os
import random
import unittest
from unittest.mock import patch

import litellm

from pipeline import config
from pipeline import sense_fr_adjudicate as adjudicate
from pipeline import sense_fr_frontier as frontier
from pipeline import sense_fr_reassign as reassign
from pipeline.llm_tasks import task_config, use_batch_prompt


def _target(key="x"):
    return {"key": key, "kind": "synset", "pos": "n", "lemmas_en": ["x"], "definition_en": "a thing"}


class Response:
    def __init__(self, payload):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": _json.dumps(payload)})()})()]


class S6M2PromptTests(unittest.TestCase):
    def test_frontier_has_distinct_unit_and_batch_contracts(self):
        item = (_target(), [], [])
        unit = frontier.build_unit_user_prompt(item)
        batch = frontier.build_user_prompt([item, (_target("y"), [], [])])
        self.assertIn("1", unit)
        self.assertNotIn("decisions", unit.lower())
        self.assertIn("2", batch)
        self.assertNotEqual(unit, batch)
        self.assertEqual(frontier.UnitTranslation.__name__, "UnitTranslation")

    def test_reassign_has_distinct_unit_and_batch_contracts(self):
        item = (_target(), [], [])
        unit = reassign.build_unit_user_prompt(item)
        batch = reassign.build_user_prompt([item, (_target("y"), [], [])])
        self.assertIn("1", unit)
        self.assertIn("JSON", unit)
        self.assertIn("2", batch)
        self.assertNotEqual(unit, batch)
        self.assertEqual(reassign.UnitReassignedDecision.__name__, "UnitReassignedDecision")

    def test_adjudication_unit_and_batch_are_selected_by_config(self):
        entry = {"key": "x", "fr": "mot", "definition_en": "a thing"}
        with patch.dict(os.environ, {"VOCAB_LLM_S6_BACKTRANSLATE": "openai/m;batch=false"}, clear=False):
            self.assertFalse(task_config("S6-backtranslate").mode_batch)
        self.assertNotEqual(
            adjudicate.build_backtranslate_unit_prompt([entry]),
            adjudicate.build_backtranslate_batch_prompt([entry, {**entry, "key": "y"}]),
        )

    def test_judge_unit_prompt_carries_same_dossier_as_batch_for_one_item(self):
        # Régression du défaut n°1 (revue M2) : le chemin unitaire du juge
        # Stage C perdait les phrases réelles du livre, le POS, les lemmes
        # et le mélange des candidats (rng.shuffle) présents dans le
        # dossier du chemin lot — voir la docstring de _judge_batch.
        entry = {
            "key": "x", "pos": "n", "lemmas_en": ["beat"],
            "fr": "battement", "fr_alt": ["coup"], "definition_en": "a rhythmic pulsation",
        }
        occurrences_by_sense = {"x": [
            {"segment_idx": 0, "context": "The steady beat of the drum.", "target_surface": "beat"},
        ]}
        audits = {"x": {"dbnary_fr": "pulsation"}}
        unit = adjudicate.build_judge_unit_prompt([entry], audits, occurrences_by_sense, random.Random(42))
        batch = adjudicate.build_judge_batch_prompt([entry], audits, occurrences_by_sense, random.Random(42))
        for fragment in ("beat", "The steady beat of the drum", "battement", "coup", "pulsation", "rhythmic pulsation"):
            self.assertIn(fragment, unit, f"{fragment!r} absent du prompt unitaire")
        # même ligne de dossier qu'un lot d'un seul élément, hors enveloppe/consigne finale
        self.assertEqual(unit.splitlines()[1], batch.splitlines()[1])

    def test_judge_unit_prompt_rejects_more_than_one_entry(self):
        entries = [
            {"key": "x", "pos": "n", "lemmas_en": [], "fr": "a", "fr_alt": [], "definition_en": "?"},
            {"key": "y", "pos": "n", "lemmas_en": [], "fr": "b", "fr_alt": [], "definition_en": "?"},
        ]
        with self.assertRaises(ValueError):
            adjudicate.build_judge_unit_prompt(entries, {}, {}, random.Random(42))


class S6M2ModelProvenanceTests(unittest.TestCase):
    def test_frontier_entry_records_resolved_model_not_global_default(self):
        # Régression du défaut n°2 : evidence.frontier_model doit refléter
        # le modèle réellement résolu par task_config, pas le modèle global
        # historique, dès qu'un override VOCAB_LLM_S6_TRANSLATE_FRONTIER
        # est posé.
        # MWE : évite la résolution omw-fr/WoNeF (nécessite target["_offset"],
        # non pertinent ici — voir frontier._resources_for).
        target = {
            "key": "mwe:give out:phrasal_verb", "kind": "mwe", "pos": "v",
            "lemmas_en": ["give out"], "occurrences": 1,
            "definition_en": "To break down, get out of order, fail.",
        }
        translation = frontier.SenseTranslation(
            sense_id="x", fr=["mot"], translation_type="equivalence_directe",
            sense_fit="ok", sense_fit_note="", source="reecrit", confidence="high",
        )
        entry = frontier.build_entry(target, translation, model="catgpt/dedicated")
        self.assertEqual(entry["evidence"]["frontier_model"], "catgpt/dedicated")
        self.assertNotEqual(entry["evidence"]["frontier_model"], config.SENSE_FR_FRONTIER_MODEL)

    def test_reassign_entry_records_resolved_model_not_global_default(self):
        entry = {"key": "hang.v.01", "kind": "synset", "pos": "v", "lemmas_en": ["hang"], "occurrences": 1}
        decision = reassign.ReassignedDecision(
            key="hang.v.01", pos="v", sense_id="cling.v.03", fr=["s'accrocher"],
            translation_type="equivalence_directe", sense_fit="ok", sense_fit_note="",
            confidence="high", reason="r",
        )
        inventory = [{"sense_id": "cling.v.03", "pos": "v", "definition": "..."}]
        store = {"hang.v.01": entry}
        group, _ = reassign.apply_decision(entry, decision, inventory, "ctx", store, model="catgpt/dedicated")
        self.assertEqual(group, "reassigne")
        self.assertEqual(store["cling.v.03"]["evidence"]["frontier_model"], "catgpt/dedicated")
        self.assertNotEqual(store["cling.v.03"]["evidence"]["frontier_model"], config.SENSE_FR_FRONTIER_MODEL)


class S6M2BatchSizeOneSelectsUnitPathTests(unittest.TestCase):
    def test_frontier_translate_batches_batch_size_one_sends_unit_prompt(self):
        # Régression du défaut n°3 : batch=true;batch_size=1 doit basculer
        # sur le chemin unitaire, pas envoyer un prompt lot à un seul item.
        # Modèle/clé dédiés à ce test : le cache disque est réel (voir
        # frontier._cache_path) et partagé entre exécutions de la suite.
        item = (_target("batch-size-one-probe"), [], [])
        with patch.dict(os.environ, {"VOCAB_LLM_S6_TRANSLATE_FRONTIER": "openai/m2-probe;batch=true;batch_size=1"}, clear=False):
            task = task_config("S6-translate-frontier")
            mode_batch = use_batch_prompt(task)
            batch_size = frontier.effective_batch_size(task)
        self.assertFalse(mode_batch)
        self.assertEqual(batch_size, 1)
        cache_file = frontier._cache_path(
            "openai/m2-probe", frontier.SYSTEM_PROMPT, frontier.build_unit_user_prompt(item),
            mode_batch=mode_batch, batch_size=batch_size,
        )
        cache_file.unlink(missing_ok=True)
        with patch.object(frontier.litellm, "batch_completion", side_effect=lambda **kw: [Response({
            "sense_id": "batch-size-one-probe", "fr": ["mot"], "translation_type": "equivalence_directe",
            "sense_fit": "ok", "sense_fit_note": "", "source": "reecrit", "confidence": "high",
        })]) as mocked:
            frontier._translate_batches([[item]], "openai/m2-probe", mode_batch=mode_batch, batch_size=batch_size)
        cache_file.unlink(missing_ok=True)
        self.assertIsNotNone(mocked.call_args, "cache disque périmé : le mock n'a pas été appelé")
        self.assertIs(mocked.call_args.kwargs["response_format"], frontier.UnitTranslation)

    def test_stage_b_batch_size_one_uses_unit_path(self):
        with patch.object(adjudicate, "_backtranslate_batch", return_value={}) as mocked:
            adjudicate.run_stage_b({}, [{"key": "x", "fr": "mot", "definition_en": "thing"}],
                                    "openai/m", batch_size=1)
        self.assertFalse(mocked.call_args.kwargs["mode_batch"])

    def test_stage_c_batch_size_one_uses_unit_path(self):
        calls = []

        def fake_judge(store, batch, audits, model, occurrences, **kwargs):
            calls.append(kwargs["mode_batch"])
            return {}

        with patch.object(adjudicate, "_judge_batch", side_effect=fake_judge):
            adjudicate.run_stage_c({}, [{"key": "x", "fr": "mot", "definition_en": "thing"}], {}, "openai/m", {},
                                    batch_size=1)
        self.assertEqual(calls, [False])


class S6M2UnitGuardTests(unittest.TestCase):
    def test_backtranslate_unit_path_rejects_multiple_entries(self):
        entries = [{"key": "x", "fr": "a", "definition_en": "?"}, {"key": "y", "fr": "b", "definition_en": "?"}]
        with self.assertRaises(ValueError):
            adjudicate._backtranslate_batch(entries, "openai/m", mode_batch=False, batch_size=1)

    def test_judge_unit_path_rejects_multiple_entries(self):
        entries = [{"key": "x", "fr": "a", "definition_en": "?"}, {"key": "y", "fr": "b", "definition_en": "?"}]
        with self.assertRaises(ValueError):
            adjudicate._judge_batch({}, entries, {}, "openai/m", {}, mode_batch=False, batch_size=1)

    def test_frontier_translate_batches_unit_path_rejects_multiple_items(self):
        batch = [(_target(), [], []), (_target("y"), [], [])]
        with self.assertRaises(ValueError):
            frontier._translate_batches([batch], "openai/m", mode_batch=False, batch_size=1)

    def test_reassign_translate_batches_unit_path_rejects_multiple_items(self):
        batch = [(_target(), [], []), (_target("y"), [], [])]
        with self.assertRaises(ValueError):
            reassign._translate_batches([batch], "openai/m", mode_batch=False, batch_size=1)


class S6M2CacheTests(unittest.TestCase):
    def test_cache_keys_include_task_mode_and_effective_batch_size(self):
        f_unit = frontier._cache_path("openai/m", "s", "u", mode_batch=False, batch_size=1)
        f_batch = frontier._cache_path("openai/m", "s", "u", mode_batch=True, batch_size=2)
        r_unit = reassign._cache_path("openai/m", "s", "u", mode_batch=False, batch_size=1)
        r_batch = reassign._cache_path("openai/m", "s", "u", mode_batch=True, batch_size=2)
        self.assertNotEqual(f_unit, f_batch)
        self.assertNotEqual(r_unit, r_batch)

    def test_cache_keys_distinguish_batch_size_at_same_mode(self):
        f_20 = frontier._cache_path("openai/m", "s", "u", mode_batch=True, batch_size=20)
        f_40 = frontier._cache_path("openai/m", "s", "u", mode_batch=True, batch_size=40)
        r_20 = reassign._cache_path("openai/m", "s", "u", mode_batch=True, batch_size=20)
        r_40 = reassign._cache_path("openai/m", "s", "u", mode_batch=True, batch_size=40)
        self.assertNotEqual(f_20, f_40)
        self.assertNotEqual(r_20, r_40)

    def test_frontier_mock_calls_use_unit_and_batch_response_formats(self):
        item = (_target(), [], [])
        unit_payload = {"sense_id": "x", "fr": ["mot"], "translation_type": "equivalence_directe", "sense_fit": "ok", "sense_fit_note": "", "source": "reecrit", "confidence": "high"}
        batch_payload = {"translations": [unit_payload, {**unit_payload, "sense_id": "y"}]}
        with patch.object(frontier.litellm, "batch_completion", side_effect=lambda **kw: [Response(unit_payload)]):
            got, _ = frontier._translate_batches([[item]], "openai/m", mode_batch=False, batch_size=1)
        self.assertEqual(got[0]["x"].fr, ["mot"])
        with patch.object(frontier.litellm, "batch_completion", side_effect=lambda **kw: [Response(batch_payload)]):
            got, _ = frontier._translate_batches([[item, (_target("y"), [], [])]], "openai/m", mode_batch=True, batch_size=2)
        self.assertEqual(set(got[0]), {"x", "y"})

    def test_adjudication_mock_paths_parse_scalar_and_envelope(self):
        one = {"key": "x", "en": "thing"}
        with patch.object(litellm, "completion", return_value=Response(one)):
            self.assertEqual(adjudicate._backtranslate_batch([{"key": "x", "fr": "mot", "definition_en": "thing"}], "openai/m", mode_batch=False), {"x": "thing"})
        verdict = {"key": "x", "fr": "mot", "fr_alt": [], "confidence": "high", "reason": "ok", "no_equivalent": False}
        with patch.object(litellm, "completion", return_value=Response({"verdicts": [verdict]})):
            got = adjudicate._judge_batch({}, [{"key": "x", "fr": "mot", "definition_en": "thing"}], {}, "openai/m", {}, mode_batch=True)
        self.assertEqual(got["x"]["fr"], "mot")

    def test_stage_c_uses_real_configured_chunks(self):
        targets = [{"key": str(i), "fr": "mot", "definition_en": "thing"} for i in range(5)]
        calls = []

        def fake_judge(store, batch, audits, model, occurrences, **kwargs):
            calls.append((len(batch), kwargs["mode_batch"], kwargs["batch_size"]))
            return {}

        with patch.dict(os.environ, {"VOCAB_LLM_S6_JUDGE_DOSSIER": "openai/m;batch=true;batch_size=2"}, clear=False), patch.object(adjudicate, "_judge_batch", side_effect=fake_judge):
            adjudicate.run_stage_c({}, targets, {}, None, {})
        self.assertEqual(calls, [(2, True, 2), (2, True, 2), (1, True, 2)])

    def test_reassign_mock_paths_parse_scalar_and_envelope(self):
        decision = {"key": "x", "pos": "n", "sense_id": "x.n.01", "fr": ["mot"], "translation_type": "equivalence_directe", "sense_fit": "ok", "sense_fit_note": "", "confidence": "high", "reason": "ok"}
        item = (_target(), [], [])
        with patch.object(reassign.litellm, "batch_completion", side_effect=lambda **kw: [Response(decision)]):
            got, _ = reassign._translate_batches([[item]], "openai/m", mode_batch=False, batch_size=1)
        self.assertEqual(got[0]["x"].sense_id, "x.n.01")
        with patch.object(reassign.litellm, "batch_completion", side_effect=lambda **kw: [Response({"decisions": [decision, {**decision, "key": "y"}]})]):
            got, _ = reassign._translate_batches([[item, (_target("y"), [], [])]], "openai/m", mode_batch=True, batch_size=2)
        self.assertEqual(set(got[0]), {"x", "y"})


if __name__ == "__main__":
    unittest.main()
