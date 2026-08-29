"""Lot 3 — Gel de l'inventaire lexical (plan Partie 2, point E).

`selected_types.jsonl` agrège par type et perd les offsets par occurrence :
il ne peut pas, à lui seul, prouver l'absence de chevauchement, ni servir de
base à une comparaison "cet artefact avale a-t-il été calculé contre LE MÊME
inventaire ?". `lexical_inventory.jsonl` (écrit par select.py::run(), une
ligne par occurrence retenue) comble ce trou ; `inventory.sha256` en est
l'empreinte : hash de la liste triée des (occurrence_id, unit_key).

Toute étape à partir de senses (senses.py, sense_fr_frontier.py,
sense_fr_adjudicate.py, export.py) doit pouvoir prouver qu'elle travaille
sur le même inventaire que celui actuellement figé par select.py — sinon
s'arrêter plutôt que mélanger silencieusement deux inventaires (ex. un
senses.jsonl calculé avant une correction de select.py, relu après). Le
mécanisme retenu ici : senses.py écrit un sidecar (SENSES_INVENTORY_HASH_PATH)
contenant le hash de l'inventaire contre lequel IL a tourné ; les étapes
suivantes comparent ce sidecar au hash COURANT de select.py
(verify_consumer). Ce n'est pas encore la fusion incrémentale par tranche
(Lot 6) — juste le garde-fou anti-mélange, posé maintenant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pipeline import atomic, config


UNRESOLVED_SENSE_ID = "unresolved"


def make_unit_key(canonical_form: str, pos: str, sense_id: str | None, *, kind: str) -> str:
    """Return the stable S4 semantic-unit key.

    The components also live as explicit columns in ``lexical_inventory``;
    this readable encoding is an identifier, not a format consumers should
    parse.  Word senses are deliberately unresolved until S5.
    """
    if kind not in {"word", "mwe"}:
        raise ValueError(f"unknown lexical unit kind: {kind!r}")
    canonical = " ".join(canonical_form.casefold().split())
    normalized_pos = pos.casefold()
    normalized_sense = sense_id or UNRESOLVED_SENSE_ID
    return f"{kind}:{canonical}:{normalized_pos}:{normalized_sense}"


def compute_hash(rows: list[dict]) -> str:
    """Hash déterministe de la liste triée des (occurrence_id, unit_key).
    Ignore volontairement tout autre champ (segment_idx, start_char,
    end_char, zone_id) : seule l'identité et l'affectation des occurrences
    retenues définissent "le même inventaire" — pas les détails de mise en
    page, qui peuvent changer sans que l'inventaire lui-même ait bougé."""

    # Le format historique sans analyse garde exactement son ancien digest.
    # Pour le nouveau schema, l'analyse canonique serialisee fait partie de
    # l'identite : changer les alternatives invalide correctement S5 et ses
    # consommateurs, meme si unit_key n'a pas encore change.
    import json
    pairs = sorted(
        (r["occurrence_id"], r["unit_key"],
         json.dumps({"surface": r.get("surface"),
                     "analysis": r.get("analysis"),
                     "multi_token_candidates": r.get("multi_token_candidates", [])},
                    ensure_ascii=False, sort_keys=True, separators=(",", ":"))
         if "surface" in r or "analysis" in r or "multi_token_candidates" in r else None)
        for r in rows
    )
    digest = hashlib.sha256()
    for occurrence_id, unit_key, analysis in pairs:
        line = f"{occurrence_id}\t{unit_key}"
        if analysis is not None:
            line += f"\t{analysis}"
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def write(rows: list[dict]) -> str:
    """Écrit lexical_inventory.jsonl + inventory.sha256, atomiquement.
    Retourne le hash écrit."""

    config.ensure_out_dir()
    atomic.atomic_write_jsonl(config.LEXICAL_INVENTORY_PATH, rows)
    digest = compute_hash(rows)
    atomic.atomic_write_text(config.INVENTORY_HASH_PATH, digest + "\n")
    return digest


def current_hash(step_name: str) -> str:
    """Hash actuellement figé par select.py. S'arrête clairement si absent
    (select.py pas encore lancé) plutôt que de laisser une étape avale
    tourner sans inventaire de référence."""

    if not config.INVENTORY_HASH_PATH.exists():
        raise SystemExit(
            f"{step_name} : {config.INVENTORY_HASH_PATH.name} absent — lance "
            f"`select` (S4) avant cette étape pour geler l'inventaire lexical "
            f"(plan Partie 2, point E)."
        )
    return config.INVENTORY_HASH_PATH.read_text(encoding="utf-8").strip()


def verify_consumer(consumer_hash_path: Path, step_name: str) -> str:
    """Vérifie qu'un artefact avale (repéré par son sidecar
    `consumer_hash_path`, écrit par la dernière exécution réussie de
    l'étape qui le produit) a bien été calculé contre l'inventaire COURANT.
    Retourne le hash courant. S'arrête avec un message explicite si le
    sidecar est absent (l'étape productrice n'a pas encore tourné) ou périmé
    (calculé contre un inventaire différent de l'actuel) — jamais de mélange
    silencieux de deux inventaires."""

    digest = current_hash(step_name)
    if not consumer_hash_path.exists():
        raise SystemExit(
            f"{step_name} : {consumer_hash_path.name} absent — lance `senses` "
            f"(S5) avant cette étape."
        )
    previous = consumer_hash_path.read_text(encoding="utf-8").strip()
    if previous != digest:
        raise SystemExit(
            f"{step_name} : inventaire périmé — {consumer_hash_path.name} a été "
            f"produit contre un inventory.sha256 différent de celui, courant, "
            f"écrit par `select`. Relance le pipeline depuis `select` (au moins "
            f"`senses`) pour régénérer cet artefact contre le même inventaire, "
            f"plutôt que de mélanger deux inventaires."
        )
    return digest


def mark_consumed(consumer_hash_path: Path, digest: str) -> None:
    """À appeler après qu'une étape avale a écrit son artefact avec succès :
    enregistre contre quel inventaire elle a tourné, pour que les étapes
    suivantes puissent le vérifier (verify_consumer)."""

    atomic.atomic_write_text(consumer_hash_path, digest + "\n")
