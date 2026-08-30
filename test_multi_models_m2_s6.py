"""Contrats hors réseau du lot M2 (prompts S6 unitaires et lots : frontier,
reassign, backtranslate, judge-dossier).

Réécrit pour le plan de décorrélation lot/stockage : les 4 tâches S6
n'appellent plus `llm_client.call`/`call_batch_completion` (cache disque par
prompt de lot entier, `_cache_path`/`_translate_batches`/`_judge_batch`/
`_backtranslate_batch`) mais `llm_client.run_units` (cache unitaire, voir
pipeline/llm_store.py, via `_translate_units`/`_judge_units`/
`_backtranslate_units`) — les mocks ciblent donc `litellm.batch_completion`,
et les anciennes clés de cache disque (`_cache_path`, préfixes de fichier)
n'existent plus."""

import json as _json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class LlmStoreIsolatedTests(unittest.TestCase):
    """run_units (pipeline/llm_client.py) stocke chaque décision dans
    pipeline/llm_store.py — isolé de la vraie data/llm_results.sqlite3 comme
    test_llm_store.py."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            config, "LLM_RESULTS_DB_PATH", Path(self._tmp.name) / "llm_results.sqlite3",
        )
        patcher.start()
        self.addCleanup(patcher.stop)


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
        # et le mélange des candidats présents dans le dossier du chemin
        # lot — voir la docstring de _judge_units.
        entry = {
            "key": "x", "pos": "n", "lemmas_en": ["beat"],
            "fr": "battement", "fr_alt": ["coup"], "definition_en": "a rhythmic pulsation",
        }
        occurrences_by_sense = {"x": [
            {"segment_idx": 0, "context": "The steady beat of the drum.", "target_surface": "beat"},
        ]}
        audits = {"x": {"dbnary_fr": "pulsation"}}
        unit = adjudicate.build_judge_unit_prompt([entry], audits, occurrences_by_sense)
        batch = adjudicate.build_judge_batch_prompt([entry], audits, occurrences_by_sense)
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
            adjudicate.build_judge_unit_prompt(entries, {}, {})


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
            sense_fit="ok", sense_fit_note="",
            definition_fr_fit="ok", definition_fr_fit_note="",
            source="reecrit", confidence="high",
        )
        entry = frontier.build_entry(target, translation, model="catgpt/dedicated")
        self.assertEqual(entry["evidence"]["frontier_model"], "catgpt/dedicated")
        self.assertNotEqual(entry["evidence"]["frontier_model"], config.SENSE_FR_FRONTIER_MODEL)

    def test_reassign_entry_records_resolved_model_not_global_default(self):
        entry = {"key": "hang.v.01", "kind": "synset", "pos": "v", "lemmas_en": ["hang"], "occurrences": 1}
        decision = reassign.ReassignedDecision(
            key="hang.v.01", pos="v", sense_id="cling.v.03", fr=["s'accrocher"],
            translation_type="equivalence_directe", sense_fit="ok", sense_fit_note="",
            definition_fr_fit="ok", definition_fr_fit_note="",
            confidence="high", reason="r",
        )
        inventory = [{"sense_id": "cling.v.03", "pos": "v", "definition": "..."}]
        store = {"hang.v.01": entry}
        group, _ = reassign.apply_decision(entry, decision, inventory, "ctx", store, model="catgpt/dedicated")
        self.assertEqual(group, "reassigne")
        self.assertEqual(store["cling.v.03"]["evidence"]["frontier_model"], "catgpt/dedicated")
        self.assertNotEqual(store["cling.v.03"]["evidence"]["frontier_model"], config.SENSE_FR_FRONTIER_MODEL)


class S6M2BatchSizeOneSelectsUnitPathTests(LlmStoreIsolatedTests):
    def test_frontier_translate_units_batch_size_one_sends_unit_prompt(self):
        # Régression du défaut n°3 : batch=true;batch_size=1 doit basculer
        # sur le chemin unitaire, pas envoyer un prompt lot à un seul item.
        item = (_target("batch-size-one-probe"), [], [])
        with patch.dict(os.environ, {"VOCAB_LLM_S6_TRANSLATE_FRONTIER": "openai/m2-probe;batch=true;batch_size=1"}, clear=False):
            task = task_config("S6-translate-frontier")
            mode_batch = use_batch_prompt(task)
            batch_size = frontier.effective_batch_size(task)
        self.assertFalse(mode_batch)
        self.assertEqual(batch_size, 1)
        with patch.object(frontier.llm_client.litellm, "batch_completion", side_effect=lambda **kw: [Response({
            "sense_id": "batch-size-one-probe", "fr": ["mot"], "translation_type": "equivalence_directe",
            "sense_fit": "ok", "sense_fit_note": "",
            "definition_fr_fit": "ok", "definition_fr_fit_note": "",
            "source": "reecrit", "confidence": "high",
        })]) as mocked:
            frontier._translate_units([item], "openai/m2-probe", mode_batch=mode_batch, batch_size=batch_size)
        self.assertIsNotNone(mocked.call_args)
        self.assertIs(mocked.call_args.kwargs["response_format"], frontier.UnitTranslation)

    def test_stage_b_batch_size_one_uses_unit_path(self):
        with patch.object(adjudicate, "_backtranslate_units", return_value={}) as mocked:
            adjudicate.run_stage_b({}, [{"key": "x", "fr": "mot", "definition_en": "thing"}],
                                    "openai/m", batch_size=1)
        self.assertFalse(mocked.call_args.kwargs["mode_batch"])

    def test_stage_c_batch_size_one_uses_unit_path(self):
        calls = []

        def fake_judge(targets, audits, occurrences_by_sense, model, **kwargs):
            calls.append(kwargs["mode_batch"])
            return {}

        with patch.object(adjudicate, "_judge_units", side_effect=fake_judge):
            adjudicate.run_stage_c({}, [{"key": "x", "fr": "mot", "definition_en": "thing"}], {}, "openai/m", {},
                                    batch_size=1)
        self.assertEqual(calls, [False])


class S6M2UnitGuardTests(unittest.TestCase):
    """Les gardes de taille (ValueError sur un chemin unitaire à N>1
    entrées) restent portées par les fonctions de RENDU (build_*_unit_prompt),
    pas par _translate_units/_judge_units/_backtranslate_units — ces
    dernières délèguent tout découpage à llm_client.run_units."""

    def test_judge_unit_prompt_rejects_multiple_entries(self):
        entries = [{"key": "x", "fr": "a", "definition_en": "?"}, {"key": "y", "fr": "b", "definition_en": "?"}]
        with self.assertRaises(ValueError):
            adjudicate.build_judge_unit_prompt(entries, {}, {})

    def test_backtranslate_unit_prompt_uses_first_entry_only(self):
        # build_backtranslate_unit_prompt (contrairement à build_judge_unit_prompt)
        # a toujours pris entries[0] sans garde explicite — comportement
        # inchangé par ce lot, juste documenté ici pour mémoire.
        entries = [{"key": "x", "fr": "a", "definition_en": "?"}, {"key": "y", "fr": "b", "definition_en": "?"}]
        prompt = adjudicate.build_backtranslate_unit_prompt(entries)
        self.assertIn("x", prompt)
        self.assertNotIn("- y |", prompt)


class S6M2CacheTests(LlmStoreIsolatedTests):
    def test_frontier_mock_calls_use_unit_and_batch_response_formats(self):
        item = (_target(), [], [])
        unit_payload = {
            "sense_id": "x", "fr": ["mot"], "translation_type": "equivalence_directe",
            "sense_fit": "ok", "sense_fit_note": "",
            "definition_fr_fit": "ok", "definition_fr_fit_note": "",
            "source": "reecrit", "confidence": "high",
        }
        batch_payload = {"translations": [unit_payload, {**unit_payload, "sense_id": "y"}]}
        with patch.object(frontier.llm_client.litellm, "batch_completion", side_effect=lambda **kw: [Response(unit_payload)]):
            got, _ = frontier._translate_units([item], "openai/m", mode_batch=False, batch_size=1)
        self.assertEqual(got["x"].fr, ["mot"])
        with patch.object(frontier.llm_client.litellm, "batch_completion", side_effect=lambda **kw: [Response(batch_payload)]):
            got, _ = frontier._translate_units([item, (_target("y"), [], [])], "openai/m", mode_batch=True, batch_size=2)
        self.assertEqual(set(got), {"x", "y"})

    def test_adjudication_mock_paths_parse_scalar_and_envelope(self):
        one = {"key": "x", "en": "thing"}
        with patch.object(adjudicate.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response(one)]):
            self.assertEqual(
                adjudicate._backtranslate_units(
                    [{"key": "x", "fr": "mot", "definition_en": "thing"}], "openai/m",
                    mode_batch=False, batch_size=1,
                ),
                {"x": "thing"},
            )
        verdict = {"key": "x", "fr": "mot", "fr_alt": [], "confidence": "high", "reason": "ok", "no_equivalent": False}
        with patch.object(adjudicate.llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [Response({"verdicts": [verdict]})]):
            got = adjudicate._judge_units(
                [{"key": "x", "fr": "mot", "definition_en": "thing"}], {}, {}, "openai/m",
                batch_size=20, mode_batch=True,
            )
        self.assertEqual(got["x"]["fr"], "mot")

    def test_stage_c_uses_real_configured_chunks(self):
        targets = [{"key": str(i), "fr": "mot", "definition_en": "thing"} for i in range(5)]
        calls = []

        def fake_judge(targets, audits, occurrences_by_sense, model, **kwargs):
            calls.append((len(targets), kwargs["mode_batch"], kwargs["batch_size"]))
            return {}

        with patch.dict(os.environ, {"VOCAB_LLM_S6_JUDGE_DOSSIER": "openai/m;batch=true;batch_size=2"}, clear=False), \
             patch.object(adjudicate, "_judge_units", side_effect=fake_judge):
            adjudicate.run_stage_c({}, targets, {}, None, {})
        # Lot de décorrélation lot/stockage : un seul appel à _judge_units
        # avec TOUTES les cibles — le découpage en tranches de 2 est
        # désormais interne à llm_client.run_units, plus une responsabilité
        # de run_stage_c.
        self.assertEqual(calls, [(5, True, 2)])

    def test_reassign_mock_paths_parse_scalar_and_envelope(self):
        decision = {
            "key": "x", "pos": "n", "sense_id": "x.n.01", "fr": ["mot"],
            "translation_type": "equivalence_directe", "sense_fit": "ok", "sense_fit_note": "",
            "definition_fr_fit": "ok", "definition_fr_fit_note": "",
            "confidence": "high", "reason": "ok",
        }
        item = (_target(), [], [])
        with patch.object(reassign.llm_client.litellm, "batch_completion", side_effect=lambda **kw: [Response(decision)]):
            got, _ = reassign._translate_units([item], "openai/m", mode_batch=False, batch_size=1)
        self.assertEqual(got["x"].sense_id, "x.n.01")
        with patch.object(reassign.llm_client.litellm, "batch_completion", side_effect=lambda **kw: [Response({"decisions": [decision, {**decision, "key": "y"}]})]):
            got, _ = reassign._translate_units([item, (_target("y"), [], [])], "openai/m", mode_batch=True, batch_size=2)
        self.assertEqual(set(got), {"x", "y"})


if __name__ == "__main__":
    unittest.main()
