"""Lot U4 du plan d'unification (fix_pipeline/multi_models/report_multi_models.md
§4bis) : champ ``custom_prompt`` du registre de tâches + variante
``s3-occurrence-tags`` récupérée de fix_pipeline/evaluate_s3_judges.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import config, mwe_judge
from pipeline.llm_tasks import TaskConfigError, TaskLlmConfig
from pipeline.prompt_variants import PROMPT_VARIANTS, PromptOverride, PromptVariantError, render


def _task(task_id: str, *, custom_prompt=None, batch: bool = False, size: int = 1,
         model: str = "catgpt/test-model"):
    provider, bare_model = model.split("/", 1)
    return TaskLlmConfig(
        task_id=task_id, batch_allowed=True, model=model, provider=provider,
        bare_model=bare_model, mode_batch=batch, batch_size=size, custom_prompt=custom_prompt,
    )


def _occurrence(index: int = 1) -> dict:
    return {
        "occurrence_id": f"m:{index}:0:7", "segment_idx": index,
        "surface": "turn off", "source": "fixture", "vpc_decision_reason": "fixture",
    }


def _completion_response(payload: dict):
    """Réponse litellm.completion factice — S3-judge-occurrence n'utilise pas
    de response_model Pydantic (JSON libre), donc un dict brut sérialisé."""
    content = json.dumps(payload)
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


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


class RenderTests(unittest.TestCase):
    def test_render_substitutes_known_placeholders(self):
        self.assertEqual(render("hello {name}", {"name": "world"}), "hello world")

    def test_render_raises_prompt_variant_error_on_unknown_placeholder(self):
        with self.assertRaises(PromptVariantError):
            render("hello {missing}", {"name": "world"})


class S3OccurrenceTagsCatalogTests(unittest.TestCase):
    def test_variant_is_registered_with_tags_schema(self):
        variant = PROMPT_VARIANTS["s3-occurrence-tags"]
        self.assertEqual(variant.schema_variant, "tags")
        self.assertIn("{canonical_form}", variant.user_template)
        self.assertIn("{count}", variant.batch_template)
        self.assertNotIn('"reason"', variant.user_template)


class JudgeOccurrenceCustomPromptTests(LlmStoreIsolatedTests):
    def test_unit_path_uses_custom_system_and_template(self):
        custom = PROMPT_VARIANTS["s3-occurrence-tags"]
        task = _task("S3-judge-occurrence", custom_prompt=custom)
        reply = {"label": "phrasal_verb", "canonical_form": "turn off", "pos": "VERB",
                 "contextual_paraphrase": "switch off", "confidence": 0.9,
                 "evidence": ["sens_specialise"], "wordnet_sense_id": None}
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            got = mwe_judge.judge_occurrence("turn off", _occurrence(), {})

        self.assertEqual(got["label"], "phrasal_verb")
        sent_system, sent_user = call.call_args.kwargs["messages"][0]
        self.assertEqual(sent_system["content"], custom.system)
        self.assertIn('"turn off"', sent_user["content"])
        self.assertNotIn("indice linguistique observable", sent_user["content"])

    def test_default_path_unaffected_when_no_custom_prompt(self):
        task = _task("S3-judge-occurrence", custom_prompt=None)
        reply = {"label": "phrasal_verb", "canonical_form": "turn off", "pos": "VERB",
                 "contextual_paraphrase": "switch off", "confidence": 0.9,
                 "evidence": ["a free-text clue"], "wordnet_sense_id": None}
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            got = mwe_judge.judge_occurrence("turn off", _occurrence(), {})

        sent_system, _sent_user = call.call_args.kwargs["messages"][0]
        self.assertEqual(sent_system["content"], mwe_judge.OCC_SYSTEM_PROMPT)
        # texte libre non filtré hors variante "tags"
        self.assertEqual(got["evidence"], ["a free-text clue"])

    def test_batch_path_uses_custom_batch_system_and_template(self):
        custom = PROMPT_VARIANTS["s3-occurrence-tags"]
        task = _task("S3-judge-occurrence", custom_prompt=custom, batch=True, size=2)
        batch = [("turn off", _occurrence(1)), ("turn off", _occurrence(2))]
        reply = {"decisions": [
            {"occurrence_id": "m:1:0:7", "label": "phrasal_verb", "canonical_form": "turn off",
             "pos": "VERB", "contextual_paraphrase": "switch off", "confidence": 0.9,
             "evidence": ["sens_specialise"], "wordnet_sense_id": None},
            {"occurrence_id": "m:2:0:7", "label": "littéral", "canonical_form": "turn off",
             "pos": "VERB", "contextual_paraphrase": "rotate away", "confidence": 0.8,
             "evidence": ["sens_specialise"], "wordnet_sense_id": None},
        ]}
        with patch.object(mwe_judge, "task_config", return_value=task), \
             patch.object(mwe_judge.llm_client.litellm, "batch_completion",
                          return_value=[_completion_response(reply)]) as call:
            got = mwe_judge.judge_occurrences_batch(batch, {})

        self.assertEqual(got["m:1:0:7"]["label"], "phrasal_verb")
        self.assertEqual(got["m:2:0:7"]["label"], "littéral")
        sent_system, _sent_user = call.call_args.kwargs["messages"][0]
        self.assertEqual(sent_system["content"], custom.batch_system)


class TagsEvidenceValidationTests(unittest.TestCase):
    def test_unknown_tag_is_dropped_and_treated_as_no_evidence(self):
        result = {"label": "idiome", "canonical_form": "let go", "pos": "VERB",
                  "contextual_paraphrase": "stop worrying", "confidence": 0.95,
                  "evidence": ["not_a_real_tag"], "wordnet_sense_id": None}
        normalized = mwe_judge._normalize_occurrence_result("let go", result, [], schema_variant="tags")
        self.assertEqual(normalized["evidence"], [])
        self.assertLess(normalized["confidence"], mwe_judge.MIN_CONFIDENCE)

    def test_known_tag_counts_as_observable_evidence(self):
        result = {"label": "idiome", "canonical_form": "let go", "pos": "VERB",
                  "contextual_paraphrase": "stop worrying", "confidence": 0.95,
                  "evidence": ["sens_specialise"], "wordnet_sense_id": None}
        normalized = mwe_judge._normalize_occurrence_result("let go", result, [], schema_variant="tags")
        self.assertEqual(normalized["evidence"], ["sens_specialise"])
        self.assertIn("observable_evidence_present", normalized["confidence_features"])
        self.assertGreaterEqual(normalized["confidence"], mwe_judge.MIN_CONFIDENCE)

    def test_default_variant_keeps_free_text_evidence_unfiltered(self):
        result = {"label": "idiome", "canonical_form": "let go", "pos": "VERB",
                  "contextual_paraphrase": "stop worrying", "confidence": 0.95,
                  "evidence": ["substitution impossible ici"], "wordnet_sense_id": None}
        normalized = mwe_judge._normalize_occurrence_result("let go", result, [])
        self.assertEqual(normalized["evidence"], ["substitution impossible ici"])


class CompleteDropsReasonForBothVariantsTests(unittest.TestCase):
    """reason ne compte plus dans `complete`, pour toute variante (Lot U4) —
    n'était relu par aucune logique du pipeline, contrairement à canonical_form/
    pos/contextual_paraphrase (sense_id, clustering)."""

    def test_default_variant_reaches_full_score_without_reason(self):
        result = {"confidence": 1.0, "canonical_form": "x", "pos": "VERB",
                  "contextual_paraphrase": "y", "evidence": ["clue"], "reason": ""}
        score, features = mwe_judge._calibrate_occurrence(result, "idiome")
        self.assertEqual(score, 1.0)
        self.assertIn("required_fields_complete", features)

    def test_tags_variant_reaches_full_score_without_reason(self):
        result = {"confidence": 1.0, "canonical_form": "x", "pos": "VERB",
                  "contextual_paraphrase": "y", "evidence": ["sens_specialise"], "reason": ""}
        score, features = mwe_judge._calibrate_occurrence(result, "idiome", schema_variant="tags")
        self.assertEqual(score, 1.0)
        self.assertIn("required_fields_complete", features)


class UnknownPromptOptionRaisesTests(unittest.TestCase):
    def test_render_error_becomes_task_config_error_in_occurrence_prompt(self):
        broken = PromptOverride(user_template="{does_not_exist}")
        with self.assertRaises(TaskConfigError):
            mwe_judge._occurrence_prompt("turn off", _occurrence(), {}, custom_prompt=broken)


if __name__ == "__main__":
    unittest.main()
