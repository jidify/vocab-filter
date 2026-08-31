"""Correction S6-3 (plan §6, `fix_pipeline/plan_action_fix_pipeline.md`) —
`fix_pipeline/evaluate_s6_3_translation_leakfree.py` : mesure de la
traduction sur le benchmark sans fuite. Ces tests tournent hors réseau — le
juge LLM est toujours remplacé par une fonction figée (voir Q0-2, plan §0 :
"les tests savent tourner hors réseau avec réponses LLM figées")."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fix_pipeline.evaluate_s6_3_translation_leakfree import (
    AUDIT_SAMPLE_SIZE,
    JUDGE_BATCH_SIZE,
    JUDGE_MODEL,
    JUDGE_TIMEOUT_SECONDS,
    build_audit_sample,
    evaluate,
    judge_pairs,
    run,
    translation_model,
    translation_source,
    write_audit_sample,
)

FIELDS = [
    "canonical_form", "surface_forms", "unit_type", "pos", "sense_id",
    "meaning_fr_official", "meaning_fr_alt", "definition_en", "needs_review", "fr_status", "contexte_en",
]


def row(canon, sense, definition, fr, fr_alt="", *, pos="n", unit_type="word",
        fr_status="", surface=None, review="False", context=""):
    surface = canon if surface is None else surface
    return dict(zip(FIELDS, [canon, surface, unit_type, pos, sense, fr, fr_alt, definition, review, fr_status, context]))


def _no_call_judge(pairs):
    raise AssertionError(f"the judge must not be called on a deterministic pair, got {pairs!r}")


class NoBenchmarkLeakTests(unittest.TestCase):
    """Vérification explicite du plan : "aucune lecture du benchmark dans
    les modules S6 de production"."""

    def test_production_pipeline_never_references_the_benchmark_file(self):
        root = Path(__file__).parent / "pipeline"
        offenders = [p for p in root.rglob("*.py") if "vocab_corrige" in p.read_text(encoding="utf-8")]
        self.assertEqual(offenders, [])

    def test_production_pipeline_never_imports_this_evaluator(self):
        root = Path(__file__).parent / "pipeline"
        offenders = [
            p for p in root.rglob("*.py")
            if "evaluate_s6_3_translation_leakfree" in p.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])


class DeterministicShortCircuitTests(unittest.TestCase):
    """L'égalité de chaîne souple de Q0-1 doit trancher sans jamais payer
    d'appel LLM — le juge, ici, lève si on l'appelle."""

    def test_identical_translation_is_equivalent_without_llm_call(self):
        actual = [row("dog", "dog.n.01", "a domesticated animal", "chien")]
        expected = [row("dog", "dog.n.01", "a domesticated animal", "chien")]
        result = evaluate(actual, expected, {}, judge=_no_call_judge)
        self.assertEqual(result["counts"]["judged_by_llm"], 0)
        self.assertEqual(result["counts"]["judged_deterministically"], 1)
        self.assertEqual(result["rows"][0]["verdict"], "equivalent")

    def test_match_against_a_benchmark_alternative_is_also_deterministic(self):
        actual = [row("plow", "plow.v.01", "to till", "labourer")]
        expected = [row("plow", "plow.v.01", "to till", "cultiver", fr_alt="labourer/défricher")]
        result = evaluate(actual, expected, {}, judge=_no_call_judge)
        self.assertEqual(result["counts"]["judged_by_llm"], 0)
        self.assertEqual(result["rows"][0]["verdict"], "equivalent")


class LlmJudgeTests(unittest.TestCase):
    def test_llm_distinguishes_synonym_from_contresens(self):
        actual = [
            row("touch", "touch.v.01", "to make physical contact", "contacter"),
            row("haggard", "haggard.s.01", "gaunt and exhausted", "hagard"),
        ]
        expected = [
            row("touch", "touch.v.01", "to make physical contact", "toucher"),
            row("haggard", "haggard.s.01", "gaunt and exhausted", "émacié"),
        ]

        verdict_by_canon = {"touch": "synonyme_acceptable", "haggard": "contresens"}

        def fake_judge(pairs):
            return {p["id"]: {"verdict": verdict_by_canon[p["canonical_form"]], "reason": "test"} for p in pairs}

        result = evaluate(actual, expected, {}, judge=fake_judge)
        self.assertEqual(result["counts"]["judged_by_llm"], 2)
        self.assertEqual(result["metrics"]["semantic_fidelity_rate"]["numerator"], 1)
        self.assertEqual(result["metrics"]["semantic_fidelity_rate"]["denominator"], 2)
        self.assertEqual(len(result["contresens"]), 1)
        self.assertEqual(result["contresens"][0]["canonical_form"], "haggard")

    def test_missing_verdict_from_judge_defaults_to_incertain(self):
        actual = [row("spa", "spa.n.01", "a mineral spring", "source")]
        expected = [row("spa", "spa.n.01", "a mineral spring", "station thermale")]
        result = evaluate(actual, expected, {}, judge=lambda pairs: {})
        self.assertEqual(result["rows"][0]["verdict"], "incertain")

    def test_empty_official_translation_is_excluded_from_fidelity(self):
        actual = [row("gizmo", "gizmo.n.01", "a gadget", "")]
        expected = [row("gizmo", "gizmo.n.01", "a gadget", "bidule")]
        result = evaluate(actual, expected, {}, judge=_no_call_judge)
        self.assertEqual(result["counts"]["total_pairs_with_both_translations"], 0)


class ContextPropagationTests(unittest.TestCase):
    """Sans `contexte_en`, ni le juge ni un relecteur humain ne peuvent
    trancher un cas ambigu (transitif/réfléchi, polysémie) — voir la
    correction qui a suivi le premier run réel de S6-3."""

    def test_actual_context_is_propagated_to_the_row(self):
        actual = [row("calm", "calm.v.01", "make calm", "calmer", context="She calmed the baby down.")]
        expected = [row("calm", "calm.v.01", "make calm", "apaiser")]
        captured = {}

        def capturing_judge(pairs):
            captured["pairs"] = pairs
            return {p["id"]: {"verdict": "synonyme_acceptable", "reason": "ok"} for p in pairs}

        evaluate(actual, expected, {}, judge=capturing_judge)
        self.assertEqual(captured["pairs"][0]["contexte_en"], "She calmed the baby down.")

    def test_falls_back_to_benchmark_context_when_actual_has_none(self):
        actual = [row("calm", "calm.v.01", "make calm", "calmer")]
        expected = [row("calm", "calm.v.01", "make calm", "apaiser", context="She calmed the baby down.")]
        result = evaluate(actual, expected, {}, judge=lambda pairs: {
            p["id"]: {"verdict": "synonyme_acceptable", "reason": "ok"} for p in pairs
        })
        self.assertEqual(result["rows"][0]["contexte_en"], "She calmed the baby down.")

    def test_judge_prompt_carries_the_book_context(self):
        from fix_pipeline.evaluate_s6_3_translation_leakfree import _judge_prompt

        pair = {
            "id": "pair-0", "lemmas_en": "calm", "definition_en": "make calm",
            "contexte_en": "She calmed the baby down.",
            "actual_fr": "calmer", "actual_fr_alt": "", "benchmark_fr": "apaiser", "benchmark_fr_alt": "",
        }
        self.assertIn("She calmed the baby down.", _judge_prompt([pair]))

    def test_judge_prompt_shows_placeholder_when_context_is_missing(self):
        from fix_pipeline.evaluate_s6_3_translation_leakfree import _judge_prompt

        pair = {
            "id": "pair-0", "lemmas_en": "calm", "definition_en": "make calm", "contexte_en": "",
            "actual_fr": "calmer", "actual_fr_alt": "", "benchmark_fr": "apaiser", "benchmark_fr_alt": "",
        }
        self.assertIn("(aucun)", _judge_prompt([pair]))

    def test_audit_sample_carries_context_for_human_review(self):
        row_with_context = {
            "id": "c-0", "judged_by": "llm", "verdict": "contresens", "reason": "faux sens",
            "canonical_form": "calm", "unit_type": "word", "sense_id": "calm.v.01",
            "definition_en": "make calm", "contexte_en": "She calmed the baby down.",
            "actual_fr": "calmer", "actual_fr_alt": "", "benchmark_fr": "apaiser", "benchmark_fr_alt": "",
            "status": "auto_strong", "source": "frontier_concordant", "model": "openai/gpt-5-mini",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            write_audit_sample([row_with_context], path)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                out_rows = list(csv.DictReader(handle))
        self.assertEqual(out_rows[0]["contexte_en"], "She calmed the baby down.")


class BatchingTests(unittest.TestCase):
    def test_judge_batch_size_timeout_and_model_are_as_specified(self):
        self.assertEqual(JUDGE_BATCH_SIZE, 50)
        self.assertEqual(JUDGE_TIMEOUT_SECONDS, 1200.0)
        self.assertTrue(JUDGE_MODEL.startswith("catgpt/"))

    def test_judge_pairs_splits_into_batches_of_at_most_judge_batch_size(self):
        calls = []

        def fake_judge_batch(pairs):
            calls.append(len(pairs))
            return {p["id"]: {"verdict": "incertain", "reason": ""} for p in pairs}

        import fix_pipeline.evaluate_s6_3_translation_leakfree as mod
        original = mod._judge_batch
        mod._judge_batch = fake_judge_batch
        try:
            pairs = [{"id": f"pair-{i}"} for i in range(JUDGE_BATCH_SIZE + 1)]
            results = judge_pairs(pairs)
        finally:
            mod._judge_batch = original
        self.assertEqual(calls, [JUDGE_BATCH_SIZE, 1])
        self.assertEqual(len(results), JUDGE_BATCH_SIZE + 1)


class SourceModelAttributionTests(unittest.TestCase):
    def test_missing_store_entry_is_reported_as_absent(self):
        self.assertEqual(translation_source(None), "absent_du_magasin")
        self.assertEqual(translation_model(None, local_translate_model="ollama/mistral-small:24b"),
                         "absent_du_magasin")

    def test_validated_entry_is_attributed_to_a_human_regardless_of_evidence(self):
        entry = {"status": "validated", "agreement": "source_unique",
                 "evidence": {"frontier_model": "openai/gpt-5-mini"}}
        self.assertEqual(translation_model(entry, local_translate_model="ollama/mistral-small:24b"), "humain")

    def test_frontier_model_is_read_from_evidence(self):
        entry = {"status": "auto_strong", "agreement": "frontier_concordant",
                 "evidence": {"frontier_model": "openai/gpt-5-mini"}}
        self.assertEqual(translation_source(entry), "frontier_concordant")
        self.assertEqual(translation_model(entry, local_translate_model="ollama/mistral-small:24b"),
                         "openai/gpt-5-mini")

    def test_dictionary_only_entry_has_no_model(self):
        entry = {"status": "auto_strong", "agreement": "concordantes", "evidence": {"llm_votes": {}}}
        self.assertEqual(translation_model(entry, local_translate_model="ollama/mistral-small:24b"),
                         "dictionnaire_seul")

    def test_local_consensus_entry_uses_the_local_task_model(self):
        entry = {"status": "auto_strong", "agreement": "source_unique",
                 "evidence": {"llm_votes": {"chien": 3}}}
        self.assertEqual(translation_model(entry, local_translate_model="ollama/mistral-small:24b"),
                         "ollama/mistral-small:24b")


class GroupingTests(unittest.TestCase):
    def test_report_breaks_down_by_status_source_and_model(self):
        actual = [
            row("touch", "touch.v.01", "to make physical contact", "contacter", fr_status="auto_llm"),
            row("haggard", "haggard.s.01", "gaunt and exhausted", "hagard", fr_status="auto_strong"),
        ]
        expected = [
            row("touch", "touch.v.01", "to make physical contact", "toucher"),
            row("haggard", "haggard.s.01", "gaunt and exhausted", "émacié"),
        ]
        store = {
            "touch.v.01": {"status": "auto_llm", "agreement": "source_unique",
                          "evidence": {"llm_votes": {"contacter": 3}}},
            "haggard.s.01": {"status": "auto_strong", "agreement": "concordantes", "evidence": {}},
        }

        verdict_by_canon = {"touch": "synonyme_acceptable", "haggard": "contresens"}

        def fake_judge(pairs):
            return {p["id"]: {"verdict": verdict_by_canon[p["canonical_form"]], "reason": "test"} for p in pairs}

        result = evaluate(actual, expected, store, judge=fake_judge)
        self.assertIn("auto_llm", result["by_status"])
        self.assertIn("auto_strong", result["by_status"])
        self.assertEqual(result["by_status"]["auto_llm"]["acceptable"], 1)
        self.assertEqual(result["by_status"]["auto_strong"]["contresens"], 1)
        self.assertIn("source_unique", result["by_source"])
        self.assertIn("concordantes", result["by_source"])


class AuditSampleTests(unittest.TestCase):
    def _judged_rows(self, n_contresens, n_other):
        rows = []
        for i in range(n_contresens):
            rows.append({"id": f"c-{i}", "judged_by": "llm", "verdict": "contresens"})
        for i in range(n_other):
            rows.append({"id": f"o-{i}", "judged_by": "llm", "verdict": "synonyme_acceptable"})
        return rows

    def test_all_contresens_are_kept_when_within_the_sample_size(self):
        rows = self._judged_rows(3, 10)
        sample = build_audit_sample(rows, size=AUDIT_SAMPLE_SIZE)
        contresens_ids = {r["id"] for r in rows if r["verdict"] == "contresens"}
        sample_ids = {r["id"] for r in sample}
        self.assertTrue(contresens_ids.issubset(sample_ids))
        self.assertLessEqual(len(sample), AUDIT_SAMPLE_SIZE)

    def test_sample_is_reproducible_across_calls(self):
        rows = self._judged_rows(2, 40)
        first = [r["id"] for r in build_audit_sample(rows)]
        second = [r["id"] for r in build_audit_sample(rows)]
        self.assertEqual(first, second)

    def test_deterministic_pairs_never_enter_the_sample(self):
        rows = self._judged_rows(1, 1) + [{"id": "d-0", "judged_by": "deterministic", "verdict": "equivalent"}]
        sample = build_audit_sample(rows, size=AUDIT_SAMPLE_SIZE)
        self.assertNotIn("d-0", [r["id"] for r in sample])

    def test_write_audit_sample_leaves_human_columns_blank(self):
        rows = [{
            "id": "c-0", "judged_by": "llm", "verdict": "contresens", "reason": "faux sens",
            "canonical_form": "haggard", "unit_type": "word", "sense_id": "haggard.s.01",
            "definition_en": "gaunt", "actual_fr": "hagard", "actual_fr_alt": "",
            "benchmark_fr": "émacié", "benchmark_fr_alt": "", "status": "auto_strong",
            "source": "concordantes", "model": "dictionnaire_seul",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            n = write_audit_sample(rows, path)
            self.assertEqual(n, 1)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                out_rows = list(csv.DictReader(handle))
        self.assertEqual(out_rows[0]["human_verdict"], "")
        self.assertEqual(out_rows[0]["human_note"], "")
        self.assertEqual(out_rows[0]["verdict"], "contresens")


class RunReadOnlyTests(unittest.TestCase):
    """`run()` sur les vrais artefacts, juge figé (pas d'appel réseau) —
    le benchmark ne doit jamais être modifié."""

    def test_run_never_writes_to_the_benchmark(self):
        import hashlib

        benchmark_path = Path("pipeline_out/vocab_corrige.csv")
        actual_path = Path("pipeline_out/vocab.csv")
        store_path = Path("data/sense_fr.jsonl")
        before = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()

        def fixed_judge(pairs):
            return {p["id"]: {"verdict": "incertain", "reason": "test hors réseau"} for p in pairs}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run(
                actual_path, benchmark_path, store_path,
                tmp_path / "metrics.json", tmp_path / "report.md", tmp_path / "audit.csv",
                judge=fixed_judge, limit=5,
            )
            self.assertTrue((tmp_path / "metrics.json").exists())
            self.assertTrue((tmp_path / "report.md").exists())
            self.assertTrue((tmp_path / "audit.csv").exists())
        self.assertLessEqual(result["counts"]["judged_by_llm"], 5)
        self.assertEqual(hashlib.sha256(benchmark_path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
