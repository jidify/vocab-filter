"""Evaluation Q0-2 réelle et non déterministe, explicitement opt-in.

Lancer avec ``RUN_REAL_Q0_2=1 python -m unittest -v test_q0_2_real_eval``.
Cette suite peut contacter le LLM configuré ; elle ne fait pas partie des tests
ordinaires hors réseau et ne modifie aucune attente du corpus.
"""

import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_REAL_Q0_2") == "1", "évaluation LLM réelle désactivée (opt-in)")
class Q02RealLlmEvaluation(unittest.TestCase):
    def test_real_mwe_judgment_returns_a_valid_verdict(self):
        from pipeline import mwe_judge
        occurrences = [{"segment_idx": 1, "surface": "worked up"}]
        verdict = mwe_judge.judge_type("work up", occurrences, {}, wordnet_candidates=[])
        self.assertIn(verdict["label"], mwe_judge.VALID_LABELS)
        self.assertFalse(mwe_judge.is_llm_failure(verdict), verdict)


if __name__ == "__main__":
    unittest.main()
