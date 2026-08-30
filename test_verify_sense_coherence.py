from __future__ import annotations

import unittest

from pipeline import verify_fr_lock
from pipeline.sense_fr import blocks_auto_lock
from pipeline.verify_sense_coherence import find_violations, revert_unverified_locks


def _entry(**overrides) -> dict:
    base = dict(
        key="give_out.v.01", kind="synset", status="auto_strong",
        sense_fit="ok", translation_type="equivalence_directe",
        fr="tomber en panne", definition_en="To break down.",
    )
    base.update(overrides)
    return base


class BlocksAutoLockTests(unittest.TestCase):
    def test_ok_and_literal_never_blocks(self):
        self.assertIsNone(blocks_auto_lock("ok", "equivalence_directe"))

    def test_definition_fr_contradiction_blocks(self):
        self.assertEqual(
            blocks_auto_lock("ok", "equivalence_directe", definition_fr_fit="contradiction"),
            "definition_fr_contradiction",
        )

    def test_definition_needs_review_blocks_regardless_of_other_axes(self):
        self.assertEqual(
            blocks_auto_lock("ok", "equivalence_directe", definition_fr_fit="ok",
                              definition_needs_review=True),
            "definition_non_validee",
        )

    def test_mismatch_blocks(self):
        self.assertEqual(blocks_auto_lock("mismatch", "equivalence_directe"), "sense_id_suspect")

    def test_doubtful_blocks(self):
        self.assertEqual(blocks_auto_lock("doubtful", "equivalence_directe"), "sense_id_douteux")

    def test_non_literal_translation_type_blocks_even_with_ok_sense_fit(self):
        self.assertEqual(blocks_auto_lock("ok", "reformulation"), "frontier_reformulation")

    def test_none_sense_fit_and_none_translation_type_never_blocks(self):
        # Chemin ollama historique (pipeline/sense_fr.py) : n'exprime jamais
        # cet avis, ce n'est pas en soi une preuve d'incohérence.
        self.assertIsNone(blocks_auto_lock(None, None))


class FindViolationsTests(unittest.TestCase):
    def test_reproduces_give_out_turn_off_historical_bug(self):
        """Cas réel mesuré (avant correctif) : sense_fr_reassign.py verrouillait
        une expression figée en `auto_joint` sans jamais renseigner sense_fit,
        alors que sa definition_en contredisait ouvertement sa traduction."""
        store = {
            "mwe:give out:phrasal_verb": _entry(
                key="mwe:give out:phrasal_verb", kind="mwe", status="auto_joint",
                sense_fit=None, definition_en="To utter, publish; to announce, proclaim, report.",
                fr="tomber en panne",
            ),
        }
        violations = find_violations(store)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["key"], "mwe:give out:phrasal_verb")

    def test_mismatch_sense_fit_on_auto_status_is_a_violation(self):
        store = {"a": _entry(key="a", status="auto_corroborated", sense_fit="mismatch")}
        violations = find_violations(store)
        self.assertEqual([v["key"] for v in violations], ["a"])

    def test_doubtful_sense_fit_on_auto_status_is_a_violation(self):
        store = {"a": _entry(key="a", status="auto_llm", sense_fit="doubtful")}
        violations = find_violations(store)
        self.assertEqual([v["key"] for v in violations], ["a"])

    def test_non_literal_translation_type_on_auto_status_is_a_violation(self):
        store = {"a": _entry(key="a", status="auto_joint", translation_type="reformulation")}
        violations = find_violations(store)
        self.assertEqual([v["key"] for v in violations], ["a"])

    def test_coherent_auto_entry_is_not_a_violation(self):
        # "work out" post-S6-1 : sense_fit="ok", definition_en cohérente
        # avec fr — aucune action attendue.
        store = {"work_out": _entry(
            key="work_out", status="auto_llm", sense_fit="ok",
            definition_en="To calculate.", fr="calculer",
        )}
        self.assertEqual(find_violations(store), [])

    def test_definition_fr_contradiction_on_auto_status_is_a_violation(self):
        # "bring up" post-correctif : sense_fit="ok" (usage compatible avec
        # UN sens de bring up) mais definition_fr_fit="contradiction" (fr
        # contredit la définition affichée) — doit être détecté même si
        # sense_fit seul ne le voit pas.
        store = {"a": _entry(key="a", status="auto_llm", sense_fit="ok",
                              definition_fr_fit="contradiction")}
        violations = find_violations(store)
        self.assertEqual([v["key"] for v in violations], ["a"])
        self.assertEqual(violations[0]["reason"], "definition_fr_contradiction")

    def test_definition_needs_review_on_auto_status_is_a_violation(self):
        store = {"a": _entry(key="a", status="auto_llm", sense_fit="ok",
                              definition_fr_fit="ok", definition_needs_review=True)}
        violations = find_violations(store)
        self.assertEqual([v["key"] for v in violations], ["a"])
        self.assertEqual(violations[0]["reason"], "definition_non_validee")

    def test_pre_existing_entry_without_definition_fr_fit_is_not_retroactively_flagged(self):
        # Champ nouveau, non rétroactif — y compris pour une entrée dont le
        # `decided_by` prouve qu'elle vient d'un module qui calcule ce champ
        # aujourd'hui (sense_fr_frontier.py) : un balayage par provenance
        # seul flanquerait la quasi-totalité du magasin déjà correctement
        # décidé (mesuré : 743 entrées), bien au-delà du plafond de 5% de
        # révision (plan §0), pour un gain nul sur les traductions déjà
        # correctes. Voir tools/migrate_sense_fr_mwe_keys.py pour le
        # traitement PONCTUEL des cas réels repérés par inspection directe.
        store = {"a": _entry(key="a", status="auto_llm", sense_fit="ok",
                              decided_by="auto_frontier")}
        self.assertNotIn("definition_fr_fit", store["a"])
        self.assertEqual(find_violations(store), [])

    def test_validated_status_is_never_flagged_even_with_stale_mismatch(self):
        # Une entrée relue par un humain (sense_fr_commit.py) ne réinitialise
        # jamais sense_fit : un reliquat "mismatch" d'AVANT relecture ne doit
        # plus être signalé une fois validée.
        store = {"a": _entry(key="a", status="validated", sense_fit="mismatch",
                              translation_type="reformulation")}
        self.assertEqual(find_violations(store), [])

    def test_legacy_path_without_sense_fit_field_is_out_of_scope(self):
        # pipeline/sense_fr.py::classify_synset_key n'écrit jamais sense_fit :
        # son acceptation automatique repose sur une preuve indépendante
        # (concordance omw-fr/WoNeF), hors périmètre de ce contrôle.
        store = {"a": _entry(key="a", status="auto_strong", sense_fit=None,
                              translation_type=None)}
        self.assertEqual(find_violations(store), [])


class RevertUnverifiedLocksTests(unittest.TestCase):
    def test_reverts_to_pending_and_preserves_fr_as_suggestion(self):
        store = {"mwe:give out:phrasal_verb": _entry(
            key="mwe:give out:phrasal_verb", kind="mwe", status="auto_joint",
            sense_fit=None, fr="tomber en panne", note="note d'origine",
        )}
        violations = find_violations(store)
        reverted = revert_unverified_locks(store, violations)

        self.assertEqual(reverted, ["mwe:give out:phrasal_verb"])
        entry = store["mwe:give out:phrasal_verb"]
        self.assertEqual(entry["status"], "pending")
        self.assertNotIn(entry["status"], verify_fr_lock.LOCKED_STATUSES)
        self.assertEqual(entry["fr"], "tomber en panne")  # jamais supprimé, reste une suggestion
        self.assertIn("note d'origine", entry["note"])
        self.assertIn("S6-1", entry["note"])
        self.assertIsNone(entry["decided_at"])
        self.assertIsNone(entry["decided_by"])

    def test_revert_makes_the_store_pass_a_second_check(self):
        store = {"a": _entry(key="a", status="auto_corroborated", sense_fit="mismatch")}
        reverted = revert_unverified_locks(store, find_violations(store))
        self.assertEqual(len(reverted), 1)
        self.assertEqual(find_violations(store), [])


if __name__ == "__main__":
    unittest.main()
