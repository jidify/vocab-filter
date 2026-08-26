"""Verrou de non-régression sur le magasin data/sense_fr.jsonl (voir
le plan, point 5) : empreinte des paires (clé, fr) dont le statut est
`validated` (décidé par un humain via sense_fr_commit.py), `auto_strong`
(décidé automatiquement par sense_fr.py ou sense_fr_frontier.py, sous
conditions strictes — voir leurs docstrings), `auto_llm` (modèle
frontière seul, sur un sens qu'aucune ressource lexicale ne couvre —
voir sense_fr_frontier.py), `auto_corroborated` (>=2 signaux
indépendants, dont au moins une source humaine hors-ligne — voir
sense_fr_adjudicate.py) ou `auto_judged` (juge sur dossier, confiance
haute, même module). Ces statuts ne sont produits QUE par ces scripts ;
ce verrou détecte toute modification qui ne passerait pas par eux
(édition manuelle du fichier, bug d'un futur run, etc.).

Élargit le verrou automatiquement quand de nouvelles clés apparaissent
(nouvelles validations légitimes) ; échoue seulement si une paire déjà
verrouillée change de valeur ou disparaît.

Usage :
    uv run python -m pipeline.verify_fr_lock
"""

from __future__ import annotations

import json

from pipeline import config, sense_fr

LOCKED_STATUSES = {"validated", "auto_strong", "auto_llm", "auto_corroborated", "auto_judged"}


def compute_lock(store: dict[str, dict]) -> dict[str, str]:
    return {
        entry["key"]: entry["fr"]
        for entry in store.values()
        if entry["status"] in LOCKED_STATUSES and entry.get("fr")
    }


def load_lock() -> dict[str, str] | None:
    if not config.SENSE_FR_LOCK_PATH.exists():
        return None
    return json.loads(config.SENSE_FR_LOCK_PATH.read_text(encoding="utf-8"))


def write_lock(lock: dict[str, str]) -> None:
    config.ensure_data_dir()
    config.SENSE_FR_LOCK_PATH.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def run() -> int:
    store = sense_fr.load_store()
    current = compute_lock(store)
    saved = load_lock()

    if saved is None:
        write_lock(current)
        print(f"Aucun verrou existant : création avec {len(current)} entrée(s).")
        return 0

    mismatches = []
    for key, fr in saved.items():
        if key not in current:
            mismatches.append((key, fr, None))
        elif current[key] != fr:
            mismatches.append((key, fr, current[key]))

    if mismatches:
        print(f"ÉCHEC : {len(mismatches)} traduction(s) verrouillée(s) modifiée(s) "
              f"hors de sense_fr_commit.py :")
        for key, old, new in mismatches:
            print(f"  - {key} : {old!r} -> {new!r}")
        return 1

    new_keys = sorted(set(current) - set(saved))
    if new_keys:
        write_lock(current)
        print(f"Verrou mis à jour : {len(new_keys)} nouvelle(s) entrée(s) "
              f"({len(current)} au total).")
    else:
        print(f"Verrou inchangé : {len(current)} entrée(s), aucune régression.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
