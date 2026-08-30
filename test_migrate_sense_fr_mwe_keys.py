"""tools/migrate_sense_fr_mwe_keys.py — élagage des clés MWE héritées
(`mwe:<canon>:<catégorie>`, plan §6, S6-1) de data/sense_fr.jsonl.

Fixtures minimales isolées (pas le vrai magasin du dépôt) : deux clés
héritées, l'une régénérable (canon présent dans pipeline_out/selected_mwe.jsonl
au format actuel), l'autre orpheline (aucune correspondance actuelle) — pour
vérifier que seule la première est élidée."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import config, sense_fr, verify_fr_lock
from tools import migrate_sense_fr_mwe_keys as migrate


class MigrateSenseFrMweKeysTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        self._store_patch = patch.object(config, "SENSE_FR_STORE_PATH", root / "sense_fr.jsonl")
        self._store_patch.start()
        self.addCleanup(self._store_patch.stop)

        self._lock_patch = patch.object(config, "SENSE_FR_LOCK_PATH", root / "sense_fr.lock.json")
        self._lock_patch.start()
        self.addCleanup(self._lock_patch.stop)

        self._selected_mwe_patch = patch.object(config, "SELECTED_MWE_PATH", root / "selected_mwe.jsonl")
        self._selected_mwe_patch.start()
        self.addCleanup(self._selected_mwe_patch.stop)

        self._data_dir_patch = patch.object(config, "DATA_DIR", root)
        self._data_dir_patch.start()
        self.addCleanup(self._data_dir_patch.stop)

        # Régénérable : le canon "give out" a un unit_key ACTUEL.
        config.SELECTED_MWE_PATH.write_text(
            json.dumps({"canonical_form": "give out", "unit_key": "mwe:give out:verb:fail.v.04"}) + "\n",
            encoding="utf-8",
        )

        store = {
            "mwe:give out:phrasal_verb": {
                "key": "mwe:give out:phrasal_verb", "kind": "mwe", "status": "validated",
                "lemmas_en": ["give out"], "fr": "tomber en panne", "fr_alt": [],
            },
            "mwe:pig smash:semi_fige": {
                "key": "mwe:pig smash:semi_fige", "kind": "mwe", "status": "validated",
                "lemmas_en": ["pig smash"], "fr": "pig smash", "fr_alt": [],
            },
            "attend.v.02": {  # pas une clé mwe: -> jamais touchée
                "key": "attend.v.02", "kind": "synset", "status": "validated",
                "lemmas_en": ["attend"], "fr": "s'occuper de", "fr_alt": [],
            },
        }
        sense_fr.write_store(store)
        verify_fr_lock.write_lock({
            "mwe:give out:phrasal_verb": "tomber en panne",
            "mwe:pig smash:semi_fige": "pig smash",
            "attend.v.02": "s'occuper de",
        })

    def test_is_legacy_mwe_key(self):
        self.assertTrue(migrate.is_legacy_mwe_key("mwe:give out:phrasal_verb"))
        self.assertTrue(migrate.is_legacy_mwe_key("mwe:lame:semi_fige"))
        self.assertFalse(migrate.is_legacy_mwe_key("mwe:give out:verb:fail.v.04"))  # format actuel
        self.assertFalse(migrate.is_legacy_mwe_key("attend.v.02"))

    def test_plan_migration_splits_regenerable_from_orphan(self):
        store = sense_fr.load_store()
        elide, keep = migrate.plan_migration(store)
        self.assertEqual(elide, ["mwe:give out:phrasal_verb"])
        self.assertEqual(keep, ["mwe:pig smash:semi_fige"])

    def test_dry_run_writes_nothing(self):
        code = migrate.run(dry_run=True)
        self.assertEqual(code, 0)
        store = sense_fr.load_store()
        self.assertIn("mwe:give out:phrasal_verb", store)
        self.assertIn("mwe:pig smash:semi_fige", store)

    def test_real_run_elides_regenerable_and_keeps_orphan(self):
        migrate.run(dry_run=False)

        store = sense_fr.load_store()
        self.assertNotIn("mwe:give out:phrasal_verb", store)  # élidée
        self.assertIn("mwe:pig smash:semi_fige", store)       # conservée intacte
        self.assertEqual(store["mwe:pig smash:semi_fige"]["fr"], "pig smash")
        self.assertIn("attend.v.02", store)                   # jamais touchée

    def test_real_run_removes_only_elided_key_from_lock(self):
        migrate.run(dry_run=False)

        lock = verify_fr_lock.load_lock()
        self.assertNotIn("mwe:give out:phrasal_verb", lock)
        self.assertEqual(lock["mwe:pig smash:semi_fige"], "pig smash")  # préservé tel quel
        self.assertEqual(lock["attend.v.02"], "s'occuper de")           # préservé tel quel


if __name__ == "__main__":
    unittest.main()
