"""Réinjecte les décisions humaines prises dans pipeline_out/sense_fr_review.csv
(colonnes `reassigner_vers`, `fr_final`, `fr_alt_final`, `decision`, `note`,
remplies à la main) dans le magasin permanent data/sense_fr.jsonl.

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

`reassigner_vers` (optionnel, `kind == "synset"` uniquement — voir
pipeline/sense_fr.py::REVIEW_FIELDS) : le relecteur a repéré, via
`contexte_en`, que les occurrences de `key` appartiennent en réalité à un
AUTRE sens (bug d'ingestion — mot composé coupé par la tokenisation,
expression absente de la base d'idiomes... — voir le plan du 2026-08-27
"Correction manuelle smart-ass / e-mail sans re-run complet", qui a
introduit ce mécanisme). Avec `decision=ok` et `fr_final` rempli, ÉCRIT :
- une entrée `validated` dans le magasin sous la NOUVELLE clé (comme un
  commit normal, mais visant `reassigner_vers` plutôt que `key`) ;
- une ligne dans data/manual_corrections.jsonl, consultée par
  pipeline/score.py::build_records() à CHAQUE export pour rediriger les
  occurrences de `key` vers la nouvelle clé — sans jamais rejouer S1-S5.
L'entrée d'origine (`key`) n'est PAS modifiée : elle continue d'exister
dans le magasin (potentiellement utile pour un autre livre), elle sort
juste naturellement de l'export de CE livre puisque ses occurrences sont
redirigées.

Usage :
    uv run python -m pipeline.sense_fr_commit
"""

from __future__ import annotations

import csv
import json
from datetime import date

from nltk.corpus import wordnet as nwn
from nltk.corpus.reader.wordnet import WordNetError

from pipeline import config, senses, sense_fr

VALID_DECISIONS = {"ok", "no", "none"}
VALID_MWE_LABELS = {"idiome", "phrasal_verb", "semi_fige"}


def parse_alt(text: str) -> list[str]:
    return [a.strip() for a in text.split(";") if a.strip()]


def derive_reassignment(
    new_key: str, definition_en_perso: str = "",
) -> tuple[str, str | None, str] | None:
    """Depuis une clé écrite à la main dans `reassigner_vers`, dérive
    (canonical_form, new_pos, definition_en) SANS jamais demander au
    relecteur de les taper séparément — tout se déduit mécaniquement de
    `new_key`, même convention que le reste du pipeline (voir
    pipeline/sense_fr_reassign.py::_build_reassigned_entry pour le sense_id
    WordNet, pipeline/score.py::build_mwe_units pour le format `mwe:`).
    Renvoie None si `new_key` n'est ni un sense_id WordNet valide ni une
    clé `mwe:<expression>:<label>` bien formée.

    `definition_en_perso` : glose tapée à la main (colonne CSV du même
    nom), utilisée UNIQUEMENT pour une clé `mwe:` — prioritaire sur la
    glose automatique de CUSTOM_IDIOMS quand fournie (typiquement une
    expression toute neuve, jamais ajoutée à pipeline/mwe.py)."""

    if new_key.startswith("mwe:"):
        parts = new_key.split(":")
        if len(parts) != 3 or not parts[1] or parts[2] not in VALID_MWE_LABELS:
            return None
        canonical_form = parts[1]
        if definition_en_perso.strip():
            definition_en = definition_en_perso.strip()
        else:
            try:
                from pipeline.mwe import get_idiom_definition
                definition_en = get_idiom_definition(canonical_form) or ""
            except Exception:
                definition_en = ""
        return canonical_form, None, definition_en

    try:
        synset = nwn.synset(new_key)
    except (WordNetError, ValueError):
        return None
    canonical_form = new_key.split(".")[0].replace("_", " ")
    new_pos = new_key.split(".")[-2]
    return canonical_form, new_pos, synset.definition()


def append_manual_correction(
    word: str, pos: str, wrong_sense_id: str, new_key: str,
    canonical_form: str, new_pos: str | None, definition_en: str, reason: str,
) -> None:
    """Ajoute (ou remplace, si déjà présente) l'entrée dans
    data/manual_corrections.jsonl — voir pipeline/score.py::
    load_manual_corrections. Clé = (word, pos, wrong_sense_id), jamais un
    segment_idx (S5 a déjà isolé tout homographe correctement désambiguïsé
    sous un AUTRE sense_id — voir la docstring du module)."""

    entries = []
    if config.MANUAL_CORRECTIONS_PATH.exists():
        with config.MANUAL_CORRECTIONS_PATH.open(encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]

    entries = [
        e for e in entries
        if not (e["word"] == word and e["pos"] == pos and e["wrong_sense_id"] == wrong_sense_id)
    ]
    entries.append({
        "word": word, "pos": pos, "wrong_sense_id": wrong_sense_id,
        "new_key": new_key, "canonical_form": canonical_form, "new_pos": new_pos,
        "definition_en": definition_en, "reason": reason,
    })

    config.ensure_data_dir()
    with config.MANUAL_CORRECTIONS_PATH.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def apply_decision(store: dict[str, dict], row: dict, reviewer: str = "human") -> dict:
    """Applique UNE ligne de décision (mêmes colonnes que REVIEW_FIELDS,
    remplies à la main ou par pipeline/review_ui.py) au magasin passé en
    argument — MUTE `store` en place, n'écrit rien sur disque (à
    l'appelant de faire `sense_fr.write_store(store)` ensuite).

    Extrait de `run()` ci-dessous pour que pipeline/review_ui.py (le petit
    serveur local, route POST /api/decision) applique une décision unitaire
    par le MÊME chemin que le commit par lot — c'est cette unicité de
    chemin que verify_fr_lock.py protège (voir sa docstring : "Ces statuts
    ne sont produits QUE par ces scripts"). Aucune logique de validation
    n'est changée par cette extraction, seul le point d'entrée est nouveau.

    Renvoie {"status": ..., "key": ..., "message": str|None} :
    - status "skipped" : ligne vide (ni decision ni reassigner_vers).
    - status "error" : rejetée, `message` explique pourquoi (même cas
      qu'un `!` dans la sortie CLI de `run()`).
    - status "ok"/"no"/"none" : validation normale sous `key`.
    - status "reassigned" : re-clée sous `reassigner_vers` ; `key` du
      résultat est alors la NOUVELLE clé. `message` porte l'avertissement
      "cible déjà verrouillée" quand applicable (pas une erreur)."""

    key = row["key"]
    decision = (row.get("decision") or "").strip().lower()
    reassign_to = (row.get("reassigner_vers") or "").strip()
    today = date.today().isoformat()

    if not decision and not reassign_to:
        return {"status": "skipped", "key": key, "message": None}
    if decision not in VALID_DECISIONS:
        return {"status": "error", "key": key,
                "message": f"decision invalide : {decision!r} (attendu : ok / no / none)"}
    if key not in store:
        return {"status": "error", "key": key, "message": "clé absente du magasin"}

    entry = store[key]
    note = (row.get("note") or "").strip()

    if reassign_to:
        if entry.get("kind") != "synset":
            return {"status": "error", "key": key,
                    "message": f"reassigner_vers ignoré : seules les entrées de type "
                               f"'synset' peuvent être re-clées (celle-ci est {entry.get('kind')!r})"}
        if decision != "ok":
            return {"status": "error", "key": key,
                    "message": "reassigner_vers rempli mais decision != 'ok'"}
        fr_final = (row.get("fr_final") or "").strip()
        if not fr_final:
            return {"status": "error", "key": key,
                    "message": "reassigner_vers rempli mais fr_final vide"}
        derived = derive_reassignment(reassign_to, row.get("definition_en_perso") or "")
        if derived is None:
            return {"status": "error", "key": key,
                    "message": f"reassigner_vers={reassign_to!r} invalide : ni sense_id "
                               f"WordNet connu, ni clé 'mwe:<expression>:"
                               f"{{{'/'.join(sorted(VALID_MWE_LABELS))}}}' bien formée"}
        canonical_form, new_pos, definition_en = derived

        word = (entry.get("lemmas_en") or [None])[0]
        pos = entry.get("pos")
        if not word or not pos:
            return {"status": "error", "key": key,
                    "message": "lemmas_en/pos manquants dans le magasin, "
                               "impossible de dériver (word, pos)"}

        new_entry = {
            "key": reassign_to, "kind": "mwe" if reassign_to.startswith("mwe:") else "synset",
            "lemmas_en": [canonical_form], "pos": new_pos or "mwe",
            "definition_en": definition_en,
            "occurrences": entry.get("occurrences", 0),
            "fr": fr_final, "fr_alt": parse_alt(row.get("fr_alt_final") or ""),
            "status": "validated", "agreement": f"reassigne_manuellement_depuis:{key}",
            "translation_type": "equivalence_directe", "sense_fit": "ok", "sense_fit_note": "",
            "source": None, "evidence": None,
            "decided_at": today, "decided_by": reviewer, "note": note,
        }
        message = None
        existing_target = store.get(reassign_to)
        if existing_target is not None and existing_target.get("status") in (
            "validated", "auto_strong", "auto_llm", "auto_corroborated", "auto_judged", "auto_joint",
        ) and existing_target.get("fr") not in (None, fr_final):
            message = (f"{reassign_to!r} déjà verrouillée avec une autre traduction "
                       f"({existing_target.get('fr')!r}) — écrasée par la décision humaine "
                       f"({fr_final!r}), comme le permet ce chemin de validation.")
        store[reassign_to] = new_entry

        append_manual_correction(
            word, pos, key, reassign_to, canonical_form, new_pos, definition_en,
            reason=note or f"reassigné manuellement depuis {key}",
        )
        return {"status": "reassigned", "key": reassign_to, "message": message}

    if decision == "ok":
        fr_final = (row.get("fr_final") or "").strip()
        if not fr_final:
            return {"status": "error", "key": key, "message": "decision=ok mais fr_final vide"}
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
    return {"status": decision, "key": key, "message": None}


def run(reviewer: str = "human") -> int:
    if not config.SENSE_FR_REVIEW_PATH.exists():
        print(f"Aucun fichier à relire : {config.SENSE_FR_REVIEW_PATH}")
        return 1

    store = sense_fr.load_store()

    with config.SENSE_FR_REVIEW_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    n_committed = {"ok": 0, "no": 0, "none": 0}
    n_reassigned = 0
    n_skipped_blank = 0
    n_errors = 0

    for row in rows:
        result = apply_decision(store, row, reviewer)
        status = result["status"]
        if status == "skipped":
            n_skipped_blank += 1
        elif status == "error":
            print(f"  ! {result['key']!r} : {result['message']}")
            n_errors += 1
        elif status == "reassigned":
            if result["message"]:
                print(f"  ! {result['message']}")
            n_reassigned += 1
        else:  # "ok" / "no" / "none"
            n_committed[status] += 1

    sense_fr.write_store(store)
    # Livre courant uniquement (voir sense_fr.format_occurrences_en) —
    # jamais lu depuis le magasin.
    n_pending = sense_fr.write_review_csv(store, senses.load_occurrences_by_sense())

    print(f"Validées : {n_committed['ok']} | Rejetées : {n_committed['no']} | "
          f"Sans équivalent : {n_committed['none']} | Re-clées : {n_reassigned} | "
          f"Non décidées (laissées en attente) : {n_skipped_blank}")
    if n_errors:
        print(f"{n_errors} ligne(s) ignorée(s) à cause d'une erreur (voir ci-dessus).")
    print(f"{n_pending} entrée(s) encore en attente -> {config.SENSE_FR_REVIEW_PATH}")
    if n_reassigned:
        print(f"{n_reassigned} correction(s) -> {config.MANUAL_CORRECTIONS_PATH} "
              f"(prise en compte au prochain `python -m pipeline.export`)")

    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(run())
