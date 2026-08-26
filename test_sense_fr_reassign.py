from __future__ import annotations

import unittest

from pipeline.sense_fr_reassign import (
    ReassignedDecision,
    STRUCTURAL_AGREEMENTS,
    apply_decision,
    classify_decision,
    open_inventory,
    select_targets,
)


def _decision(**overrides) -> ReassignedDecision:
    base = dict(
        key="k", pos="n", sense_id=None, fr=["mot"],
        translation_type="equivalence_directe", confidence="high", reason="r",
    )
    base.update(overrides)
    return ReassignedDecision(**base)


class OpenInventoryTests(unittest.TestCase):
    def test_spans_pos_and_contains_beat_noun_and_verb(self):
        keys = {row["sense_id"] for row in open_inventory("beat")}
        self.assertIn("beat.n.08", keys)
        self.assertIn("beat.v.04", keys)
        self.assertIn("beat.v.08", keys)


class SelectTargetsTests(unittest.TestCase):
    def test_excludes_frontier_sans_ressource_and_auto_statuses(self):
        store = {
            "a": {"key": "a", "status": "pending", "agreement": "sense_id_suspect"},
            "b": {"key": "b", "status": "pending", "agreement": "frontier_sans_ressource"},
            "c": {"key": "c", "status": "auto_strong", "agreement": "frontier_concordant"},
            "d": {"key": "d", "status": "pending", "agreement": "frontier_desaccord"},
        }
        keys = {e["key"] for e in select_targets(store)}
        self.assertEqual(keys, {"a"})
        self.assertTrue(STRUCTURAL_AGREEMENTS.isdisjoint({"frontier_sans_ressource", "frontier_desaccord"}))


class ClassifyDecisionTests(unittest.TestCase):
    def test_mwe_never_rekeyed_even_with_a_sense_id_proposed(self):
        entry = {"key": "mwe:give out:phrasal_verb", "kind": "mwe", "pos": None}
        group, new_key = classify_decision(entry, "fail.v.04")
        self.assertEqual(group, "promu")
        self.assertEqual(new_key, entry["key"])

    def test_same_sense_id_confirmed_promotes_in_place(self):
        entry = {"key": "beat.n.08", "kind": "synset", "pos": "n"}
        group, new_key = classify_decision(entry, "beat.n.08")
        self.assertEqual((group, new_key), ("promu", "beat.n.08"))

    def test_different_sense_same_pos_reassigns(self):
        entry = {"key": "hang.v.01", "kind": "synset", "pos": "v"}
        group, new_key = classify_decision(entry, "cling.v.03")
        self.assertEqual((group, new_key), ("reassigne", "cling.v.03"))

    def test_satellite_and_relational_adjective_treated_as_same_pos(self):
        entry = {"key": "subatomic.s.01", "kind": "synset", "pos": "s"}
        group, new_key = classify_decision(entry, "subatomic.a.01")
        self.assertEqual((group, new_key), ("reassigne", "subatomic.a.01"))

    def test_pos_change_goes_to_audit_not_rekey(self):
        entry = {"key": "beat.v.04", "kind": "synset", "pos": "v"}
        group, new_key = classify_decision(entry, "beat.n.08")
        self.assertEqual((group, new_key), ("audit", None))

    def test_sense_id_none_goes_to_audit(self):
        entry = {"key": "beat.v.04", "kind": "synset", "pos": "v"}
        group, new_key = classify_decision(entry, None)
        self.assertEqual((group, new_key), ("audit", None))


class ApplyDecisionTests(unittest.TestCase):
    def test_invented_sense_id_outside_inventory_falls_back_to_audit(self):
        entry = {
            "key": "beat.v.04", "kind": "synset", "pos": "v", "lemmas_en": ["beat"],
            "occurrences": 1,
        }
        inventory = [{"sense_id": "beat.v.08", "pos": "v", "definition": "..."}]
        decision = _decision(key="beat.v.04", pos="n", sense_id="beat.n.99")  # n'existe pas dans l'inventaire
        store = {"beat.v.04": entry}
        group, row = apply_decision(entry, decision, inventory, "ctx", store)
        self.assertEqual(group, "audit")
        self.assertEqual(row["sense_id_propose"], "")
        # le magasin n'est pas modifié pour un cas "audit"
        self.assertEqual(store["beat.v.04"], entry)

    def test_locked_target_is_never_overwritten(self):
        entry = {
            "key": "hang.v.01", "kind": "synset", "pos": "v", "lemmas_en": ["hang"],
            "occurrences": 2, "agreement": "sense_id_douteux", "status": "pending",
        }
        locked_target = {
            "key": "cling.v.03", "status": "validated", "fr": "s'accrocher",
            "occurrences": 5, "lemmas_en": ["cling"],
        }
        store = {"hang.v.01": entry, "cling.v.03": locked_target}
        inventory = [{"sense_id": "cling.v.03", "pos": "v", "definition": "..."}]
        decision = _decision(key="hang.v.01", pos="v", sense_id="cling.v.03", fr=["se cramponner"])

        group, row = apply_decision(entry, decision, inventory, "ctx", store)

        self.assertEqual(group, "bloque")
        self.assertIn("verrouillée", row["note"])
        # ni la cible verrouillée ni l'entrée d'origine ne sont modifiées
        self.assertEqual(store["cling.v.03"], locked_target)
        self.assertEqual(store["hang.v.01"], entry)

    def test_unlocked_reassignment_writes_new_key_and_marks_origin(self):
        entry = {
            "key": "hang.v.01", "kind": "synset", "pos": "v", "lemmas_en": ["hang"],
            "occurrences": 2, "agreement": "sense_id_douteux", "status": "pending",
        }
        store = {"hang.v.01": entry}
        inventory = [{"sense_id": "cling.v.03", "pos": "v", "definition": "hold on tightly"}]
        decision = _decision(key="hang.v.01", pos="v", sense_id="cling.v.03", fr=["se cramponner"])

        group, row = apply_decision(entry, decision, inventory, "ctx", store)

        self.assertEqual(group, "reassigne")
        self.assertIsNone(row)
        self.assertEqual(store["cling.v.03"]["status"], "auto_joint")
        self.assertEqual(store["cling.v.03"]["fr"], "se cramponner")
        self.assertEqual(store["hang.v.01"]["agreement"], "reassigne_vers:cling.v.03")
        # l'entrée d'origine reste `pending` : ce module ne change jamais son statut
        self.assertEqual(store["hang.v.01"]["status"], "pending")

    def test_confirmed_synset_promotes_in_place(self):
        entry = {
            "key": "beat.n.08", "kind": "synset", "pos": "n", "lemmas_en": ["beat"],
            "occurrences": 30, "agreement": "sense_id_suspect", "status": "pending",
            "fr": "petite pause",
        }
        store = {"beat.n.08": entry}
        inventory = [{"sense_id": "beat.n.08", "pos": "n", "definition": "..."}]
        decision = _decision(key="beat.n.08", pos="n", sense_id="beat.n.08", fr=["petite pause"])

        group, row = apply_decision(entry, decision, inventory, "ctx", store)

        self.assertEqual(group, "promu")
        self.assertIsNone(row)
        self.assertEqual(store["beat.n.08"]["status"], "auto_joint")
        self.assertEqual(store["beat.n.08"]["agreement"], "auto_joint_confirme")

    def test_mwe_translation_update_never_touches_key(self):
        entry = {
            "key": "mwe:give out:phrasal_verb", "kind": "mwe", "pos": None,
            "lemmas_en": ["give out"], "occurrences": 1,
            "agreement": "sense_id_suspect", "status": "pending",
        }
        store = {entry["key"]: entry}
        decision = _decision(
            key=entry["key"], pos="mwe", sense_id="fail.v.04",  # le modèle propose un synset : doit être ignoré
            fr=["tomber en panne"],
        )

        group, row = apply_decision(entry, decision, [], "ctx", store)

        self.assertEqual(group, "promu")
        self.assertIn(entry["key"], store)
        self.assertEqual(store[entry["key"]]["fr"], "tomber en panne")
        self.assertEqual(store[entry["key"]]["kind"], "mwe")


if __name__ == "__main__":
    unittest.main()
