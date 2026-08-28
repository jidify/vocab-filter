import hashlib
import json
import unittest
from pathlib import Path

from fix_pipeline.evaluate_fix_quality import evaluate, match_rows, normalize, read_csv

FIELDS = ["canonical_form", "surface_forms", "unit_type", "pos", "sense_id", "meaning_fr_official", "meaning_fr_alt", "definition_en", "needs_review"]


def row(canon, surface, sense, definition, fr="fr", pos="n", unit_type="word", review="False"):
    return dict(zip(FIELDS, [canon, surface, unit_type, pos, sense, fr, "", definition, review]))


class FixQualityEvaluatorTests(unittest.TestCase):
    def test_normalization_unicode_case_apostrophes(self):
        self.assertEqual(normalize("  L’ÉTÉ\u00a0"), normalize("l'été"))

    def test_homonyms_are_not_overwritten_and_match_by_surface(self):
        actual = [row("burn out", "burns out", "wrong.1", "bulb"), row("burn out", "burn out", "right.2", "person")]
        expected = [row("BURN OUT", "burn out", "right.2", "person"), row("burn out", "burns out", "right.1", "bulb")]
        pairs, only_a, only_e = match_rows(actual, expected)
        self.assertFalse(only_a or only_e)
        self.assertEqual(len(pairs), 2)
        self.assertEqual({(actual[p.actual_index]["surface_forms"], expected[p.expected_index]["surface_forms"]) for p in pairs}, {("burns out", "burns out"), ("burn out", "burn out")})
        self.assertEqual(evaluate(actual, expected)["metrics"]["sense_identity_accuracy"]["numerator"], 1)

    def test_multiset_inventory_counts_duplicate_excess(self):
        actual = [row("lead", "lead", "lead.n.1", "metal"), row("lead", "lead", "lead.v.1", "guide")]
        expected = [row("lead", "lead", "lead.n.1", "metal")]
        result = evaluate(actual, expected)
        self.assertEqual(result["counts"]["matched_rows"], 1)
        self.assertEqual(result["counts"]["actual_only_rows"], 1)
        self.assertEqual(result["metrics"]["unit_precision"]["value"], 1.0)
        self.assertEqual(result["counts"]["out_of_scope_needs_review"], 1)

    def test_absence_from_benchmark_is_not_automatically_false_positive(self):
        result = evaluate([row("latch", "latch", "latch.n.01", "a fastening")], [])
        self.assertEqual(result["actual_only"][0]["classification"], "needs_review")
        self.assertEqual(result["counts"]["true_false_positives"], 0)

    def test_audited_out_of_scope_categories_are_separate(self):
        actual = [row("extra", "extra", "extra.n.01", "extra"), row("noise", "noise", "noise.n.01", "noise")]
        audit = {
            ("extra", "word", "extra.n.01"): {"status": "validated_improvement", "reason": "independent evidence"},
            ("noise", "word", "noise.n.01"): {"status": "false_positive", "reason": "wrong span"},
        }
        result = evaluate(actual, [], audit)
        self.assertEqual(result["counts"]["validated_out_of_scope_improvements"], 1)
        self.assertEqual(result["counts"]["true_false_positives"], 1)
        self.assertEqual(result["metrics"]["audited_out_of_scope_precision"]["value"], 0.5)

    def test_run_is_reproducible_and_benchmark_read_only(self):
        benchmark_path = Path("pipeline_out/vocab_corrige.csv")
        actual_path = Path("pipeline_out/vocab.csv")
        before = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
        first = json.dumps(evaluate(read_csv(actual_path), read_csv(benchmark_path)), ensure_ascii=False, sort_keys=True)
        second = json.dumps(evaluate(read_csv(actual_path), read_csv(benchmark_path)), ensure_ascii=False, sort_keys=True)
        self.assertEqual(first, second)
        self.assertEqual(hashlib.sha256(benchmark_path.read_bytes()).hexdigest(), before)

    def test_production_modules_do_not_import_evaluator(self):
        root = Path(__file__).parent
        offenders = []
        for path in (root / "pipeline").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "evaluate_fix_quality" in text or "vocab_corrige" in text:
                offenders.append(path)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
