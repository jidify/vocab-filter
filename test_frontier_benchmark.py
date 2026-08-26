from __future__ import annotations

import unittest

from pipeline.eval_frontier_ablation import (
    AnswerEvaluation,
    PairJudgment,
    _inventory,
    _judge_prompt,
    _normalize_judgment,
    build_report,
)
from pipeline.sense_fr_frontier import is_protected


class FrontierBenchmarkTests(unittest.TestCase):
    def test_inventory_spans_pos_and_contains_beat_noun_and_verb(self):
        inventory = _inventory("beat")
        keys = {row["sense_id"] for row in inventory}
        self.assertIn("beat.n.08", keys)
        self.assertIn("beat.v.04", keys)
        self.assertIn("beat.v.08", keys)

    def test_normalization_removes_blind_order(self):
        good = AnswerEvaluation(
            pos_correct=True, sense_correct=True, fr_acceptable=True,
            overall=True, reason="ok",
        )
        bad = AnswerEvaluation(
            pos_correct=False, sense_correct=False, fr_acceptable=False,
            overall=False, reason="non",
        )
        judgment = PairJudgment(
            case_id="c", x=good, y=bad, preferred="x",
        )
        normalized = _normalize_judgment(judgment, first_joint=True)
        self.assertTrue(normalized["joint"]["overall"])
        self.assertFalse(normalized["current"]["overall"])
        self.assertEqual("joint", normalized["preferred"])

        normalized = _normalize_judgment(judgment, first_joint=False)
        self.assertTrue(normalized["current"]["overall"])
        self.assertFalse(normalized["joint"]["overall"])
        self.assertEqual("current", normalized["preferred"])

    def test_second_judge_pass_inverts_every_pair(self):
        case = {
            "case_id": "c", "target_surface": "beat", "context": "Small beat.",
            "inventory": [],
            "current": {"pos": "v", "sense_id": "beat.v.04", "meaning_fr": "pause"},
        }
        joint = {"c": {"pos": "n", "sense_id": None, "meaning_fr": "pause"}}
        prompt_1, order_1 = _judge_prompt([case], joint, 42, 1)
        prompt_2, order_2 = _judge_prompt([case], joint, 42, 2)
        self.assertNotEqual(order_1["c"], order_2["c"])
        self.assertNotEqual(prompt_1, prompt_2)

    def test_report_exposes_dimensions_strata_and_gates(self):
        evaluation = {
            "pos_correct": True, "sense_correct": True,
            "fr_acceptable": True, "overall": True, "reason": "ok",
        }
        cases = [{"case_id": "c", "stratum": "pending_structurel"}]
        judgments = [{
            "case_id": "c", "stratum": "pending_structurel",
            "source_status": "pending", "stable": True,
            "pass_1": {"current": evaluation, "joint": evaluation, "preferred": "tie"},
            "pass_2": {"current": evaluation, "joint": evaluation, "preferred": "tie"},
        }]
        report = build_report(cases, judgments, "candidate", "judge")
        self.assertIn("### Par dimension", report)
        self.assertIn("### Par strate", report)
        self.assertIn("Jugements instables <10 %", report)


class ProtectedStatusTests(unittest.TestCase):
    def test_validated_and_auto_joint_are_protected_from_rerun(self):
        self.assertTrue(is_protected({"status": "validated"}))
        self.assertTrue(is_protected({"status": "auto_joint"}))

    def test_pending_and_auto_strong_are_not_protected(self):
        self.assertFalse(is_protected({"status": "pending"}))
        self.assertFalse(is_protected({"status": "auto_strong"}))
        self.assertFalse(is_protected(None))


if __name__ == "__main__":
    unittest.main()
