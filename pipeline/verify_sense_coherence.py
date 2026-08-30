"""Contrôle bloquant de cohérence sens-définition-FR (plan §6, Correction
S6-1) : aucune entrée VERROUILLÉE de `data/sense_fr.jsonl`
(pipeline.verify_fr_lock.LOCKED_STATUSES) ne doit porter un `sense_fit`
"mismatch"/"doubtful", un `definition_fr_fit` "contradiction" (la
traduction fr contredit sa PROPRE définition — axe DISTINCT de sense_fit,
voir sense_fr.blocks_auto_lock), un `definition_needs_review` vrai (la
définition elle-même n'a jamais été validée en amont par S3-3/S4), un
`translation_type` non littéral, ni — cas plus insidieux — un statut qui
ne peut être produit QUE par un module tenu de calculer ce signal
(`auto_joint`, pipeline/sense_fr_reassign.py) tout en laissant `sense_fit`
à `null` : cela signale une entrée verrouillée AVANT que cette porte
n'existe, jamais vérifiée (cas réel mesuré : `give out` et `turn off`
dans data/sense_fr.jsonl, dont la `definition_en` contredisait ouvertement
le `fr` verrouillé — voir sense_fr.blocks_auto_lock et le correctif
appliqué à ces deux clés).

`pipeline/sense_fr.py` (chemin ollama historique, classify_synset_key /
classify_mwe_key) ne renseigne jamais `sense_fit` : son acceptation
automatique repose sur une preuve différente et indépendante (concordance
omw-fr/WoNeF, ou consensus LLM + rétro-traduction sémantique), hors du
périmètre de ce contrôle par construction — voir sa docstring. Seules les
entrées produites par un module qui EXPRIME un avis sur `sense_fit`
(sense_fr_frontier.py, sense_fr_reassign.py, ou sense_fr_adjudicate.py/
sense_fr_commit.py quand ils promeuvent une entrée qui en porte déjà un)
sont concernées ici.

Usage :
    uv run python -m pipeline.verify_sense_coherence
"""

from __future__ import annotations

from pipeline import config, sense_fr, verify_fr_lock

# Statuts où l'ABSENCE de sense_fit est elle-même une anomalie : ce statut
# n'est produit QUE par un module qui calcule ce champ pour CHAQUE décision
# (sense_fr_reassign.py) — contrairement à auto_strong/auto_llm, qui
# peuvent aussi venir du chemin ollama historique (pipeline/sense_fr.py),
# lequel ne connaît pas ce champ par construction (voir la docstring du
# module ci-dessus).
STATUSES_REQUIRING_SENSE_FIT = {"auto_joint"}


def find_violations(store: dict[str, dict]) -> list[dict]:
    """Une ligne par entrée verrouillée qui ne prouve pas sa cohérence
    sens-définition-FR — liste vide si le magasin est propre.

    `validated` (relu par un HUMAIN, pipeline/sense_fr_commit.py) est
    volontairement exclu : c'est le statut vers lequel un mismatch/doubtful
    doit précisément être routé (voir sense_fr.blocks_auto_lock et la
    politique de relecture) — un `sense_fit` resté "mismatch"/"doubtful"
    sur une entrée `validated` est un reliquat d'AVANT la relecture, pas
    une incohérence encore active : sense_fr_commit.py ne le réinitialise
    jamais lui-même, un humain ayant déjà tranché n'a pas besoin de le
    faire. Seuls les statuts AUTOMATIQUES sont concernés ici ("Ne
    verrouille jamais automatiquement", plan §6 S6-1)."""
    violations = []
    for entry in store.values():
        if entry.get("status") not in verify_fr_lock.LOCKED_STATUSES:
            continue
        if entry.get("status") == "validated":
            continue

        sense_fit = entry.get("sense_fit")
        translation_type = entry.get("translation_type")
        definition_fr_fit = entry.get("definition_fr_fit")
        definition_needs_review = entry.get("definition_needs_review", False)

        if sense_fit is None:
            if entry.get("status") in STATUSES_REQUIRING_SENSE_FIT:
                violations.append({
                    "key": entry["key"], "status": entry["status"],
                    "sense_fit": sense_fit, "translation_type": translation_type,
                    "definition_fr_fit": definition_fr_fit,
                    "fr": entry.get("fr"), "definition_en": entry.get("definition_en"),
                    "reason": "jamais vérifié (sense_fit absent d'un statut qui doit le porter)",
                })
            continue  # hors périmètre (chemin sans sense_fit, voir la docstring du module)

        # `definition_fr_fit` absent (entrée écrite avant ce champ) n'est PAS
        # traité comme une violation ici, même via `decided_by` : un balayage
        # par provenance seul (`decided_by in {"auto_frontier", "auto_joint"}`)
        # flanque ~743 entrées sur ce magasin (mesuré) — la quasi-totalité des
        # sens de MOTS déjà correctement décidés, jamais réexaminés — ce qui
        # viole le plafond de 5% de la file de révision (plan §0) pour un
        # gain nul sur ces entrées déjà correctes. Un cas RÉEL repéré ainsi
        # (`mwe:check in:phrasal_verb`/`mwe:keep up:phrasal_verb`, fr
        # contredisant leur definition_en, non régénérables par le S1-S5
        # courant) a été routé en pending PONCTUELLEMENT — voir
        # tools/migrate_sense_fr_mwe_keys.py — plutôt que par une règle
        # générale ici.
        block_reason = sense_fr.blocks_auto_lock(
            sense_fit, translation_type,
            definition_fr_fit=definition_fr_fit,
            definition_needs_review=definition_needs_review,
        )
        if block_reason:
            violations.append({
                "key": entry["key"], "status": entry["status"],
                "sense_fit": sense_fit, "translation_type": translation_type,
                "definition_fr_fit": definition_fr_fit,
                "fr": entry.get("fr"), "definition_en": entry.get("definition_en"),
                "reason": block_reason,
            })

    return violations


def revert_unverified_locks(store: dict[str, dict], violations: list[dict]) -> list[str]:
    """Repasse chaque entrée en violation à `pending`, MUTE `store` en
    place, renvoie les clés touchées. Ne reçoit jamais un statut
    `validated` (voir find_violations) — seul un verrouillage AUTOMATIQUE
    jamais prouvé cohérent est concerné. `fr`/`fr_alt` restent en place
    comme SUGGESTION pour la relecture humaine (voir
    pipeline/sense_fr.py::build_review_row) : rien n'est supprimé (plan
    §7-3, aucune disparition silencieuse), seul le statut change."""
    reverted = []
    for v in violations:
        entry = store[v["key"]]
        previous_status = entry["status"]
        entry["status"] = "pending"
        entry["agreement"] = f"s6_1_coherence_revert:{v['reason']}"
        entry["note"] = (
            f"[S6-1] verrouillage automatique ({previous_status}) annulé, cohérence "
            f"sens-définition-FR jamais prouvée ({v['reason']}) — à relire. "
            + (entry.get("note") or "")
        ).strip()
        entry["decided_at"] = None
        entry["decided_by"] = None
        reverted.append(v["key"])
    return reverted


def run(fix: bool = False) -> int:
    store = sense_fr.load_store()
    violations = find_violations(store)

    if not violations:
        n_locked = sum(1 for e in store.values() if e.get("status") in verify_fr_lock.LOCKED_STATUSES)
        print(f"OK : {n_locked} entrée(s) verrouillée(s), aucune incohérence sens-définition-FR détectée.")
        return 0

    print(f"{len(violations)} entrée(s) verrouillée(s) sans cohérence sens-définition-FR vérifiée :")
    for v in violations:
        print(f"  - {v['key']} (status={v['status']}, sense_fit={v['sense_fit']!r}, "
              f"translation_type={v['translation_type']!r}, definition_fr_fit={v.get('definition_fr_fit')!r}, "
              f"raison={v['reason']}) -> fr={v['fr']!r} | definition_en={v['definition_en']!r}")

    if not fix:
        print("Relancer avec --fix pour repasser ces entrées en `pending` (relecture humaine "
              "requise) et les retirer du verrou (data/sense_fr.lock.json).")
        return 1

    reverted = revert_unverified_locks(store, violations)
    sense_fr.write_store(store)

    # Même geste que pipeline/purge_unit.py pour une clé qui disparaît du
    # verrou : sans ce retrait, verify_fr_lock échouerait au prochain run
    # sur une clé verrouillée qui n'est plus dans un statut LOCKED_STATUSES.
    lock = verify_fr_lock.load_lock() or {}
    n_lock_removed = 0
    for key in reverted:
        if key in lock:
            del lock[key]
            n_lock_removed += 1
    if n_lock_removed:
        verify_fr_lock.write_lock(lock)

    print(f"{len(reverted)} entrée(s) repassée(s) en `pending` (dont {n_lock_removed} retirée(s) "
          f"du verrou) -> à relire via {config.SENSE_FR_REVIEW_PATH} (régénéré par le prochain run "
          f"de sense_fr_frontier/sense_fr_adjudicate/sense_fr.write_review_csv).")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix", action="store_true",
        help="Repasse en `pending` (et retire du verrou) toute entrée verrouillée dont la "
             "cohérence sens-définition-FR n'est pas prouvée. Sans cette option, le script "
             "se contente de signaler (code de sortie 1) — contrôle bloquant en lecture seule.",
    )
    args = parser.parse_args()
    raise SystemExit(run(fix=args.fix))
