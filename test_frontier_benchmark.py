from __future__ import annotations

import unittest

from pipeline.sense_fr_frontier import is_protected


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
