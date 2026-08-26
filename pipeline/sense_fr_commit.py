"""Réinjecte les décisions humaines prises dans pipeline_out/sense_fr_review.csv
(colonnes `fr_final`, `fr_alt_final`, `decision`, `note`, remplies à la
main) dans le magasin permanent data/sense_fr.jsonl.

C'est le SEUL chemin par lequel une entrée peut devenir `validated` —
voir pipeline/sense_fr.py et pipeline/verify_fr_lock.py, qui protège
ces valeurs contre toute modification qui ne passerait pas par ici.

`decision` attendu :
    ok    -> `fr_final` doit être rempli. status = validated.
    no    -> la suggestion pré-remplie était fausse, pas de remplacement
             fourni pour l'instant. status = rejected (reste hors export ;
             à retraiter manuellement plus tard si besoin).
    none  -> aucun équivalent français satisfaisant n'existe pour ce
             sens (cas légitime : expression trop culturellement située,
             etc.). status = no_equivalent.
    (vide) -> ligne pas encore relue, laissée telle quelle dans le
              magasin (`pending`) et régénérée dans le prochain CSV.

Usage :
    uv run python -m pipeline.sense_fr_commit
"""

from __future__ import annotations

import csv
from datetime import date

from pipeline import config, senses, sense_fr

VALID_DECISIONS = {"ok", "no", "none"}


def parse_alt(text: str) -> list[str]:
    return [a.strip() for a in text.split(";") if a.strip()]


def run(reviewer: str = "human") -> int:
    if not config.SENSE_FR_REVIEW_PATH.exists():
        print(f"Aucun fichier à relire : {config.SENSE_FR_REVIEW_PATH}")
        return 1

    store = sense_fr.load_store()

    with config.SENSE_FR_REVIEW_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    n_committed = {"ok": 0, "no": 0, "none": 0}
    n_skipped_blank = 0
    n_errors = 0
    today = date.today().isoformat()

    for row in rows:
        key = row["key"]
        decision = (row.get("decision") or "").strip().lower()

        if not decision:
            n_skipped_blank += 1
            continue
        if decision not in VALID_DECISIONS:
            print(f"  ! decision invalide ignorée pour {key!r} : {decision!r} "
                  f"(attendu : ok / no / none)")
            n_errors += 1
            continue
        if key not in store:
            print(f"  ! clé absente du magasin, ignorée : {key!r}")
            n_errors += 1
            continue

        entry = store[key]
        note = (row.get("note") or "").strip()

        if decision == "ok":
            fr_final = (row.get("fr_final") or "").strip()
            if not fr_final:
                print(f"  ! decision=ok mais fr_final vide pour {key!r} — ignoré")
                n_errors += 1
                continue
            entry["fr"] = fr_final
            entry["fr_alt"] = parse_alt(row.get("fr_alt_final") or "")
            entry["status"] = "validated"
        elif decision == "no":
            entry["fr"] = None
            entry["fr_alt"] = []
            entry["status"] = "rejected"
        else:  # "none"
            entry["fr"] = None
            entry["fr_alt"] = []
            entry["status"] = "no_equivalent"

        entry["decided_at"] = today
        entry["decided_by"] = reviewer
        entry["note"] = note
        store[key] = entry
        n_committed[decision] += 1

    sense_fr.write_store(store)
    # Livre courant uniquement (voir sense_fr.format_occurrences_en) —
    # jamais lu depuis le magasin.
    n_pending = sense_fr.write_review_csv(store, senses.load_occurrences_by_sense())

    print(f"Validées : {n_committed['ok']} | Rejetées : {n_committed['no']} | "
          f"Sans équivalent : {n_committed['none']} | "
          f"Non décidées (laissées en attente) : {n_skipped_blank}")
    if n_errors:
        print(f"{n_errors} ligne(s) ignorée(s) à cause d'une erreur (voir ci-dessus).")
    print(f"{n_pending} entrée(s) encore en attente -> {config.SENSE_FR_REVIEW_PATH}")

    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(run())
