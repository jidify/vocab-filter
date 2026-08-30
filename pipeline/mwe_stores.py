"""Lot 3 — Magasin MWE (plan Partie 2, point C).

Une décision de TYPE (« "turn off" = phrasal_verb ») ne suffit pas pour
distinguer "turn off the light" (lexicalisé) de "turn off the road" (littéral,
même verbe+particule). Magasin permanent sous data/, sur le MÊME modèle que
`data/sense_fr.jsonl` (voir `pipeline/sense_fr.py::load_store`/`write_store`
et `pipeline/sense_fr_frontier.py::PROTECTED_STATUSES`) :

- `data/mwe_occurrence_decisions.jsonl` — clé = `occurrence_id|canon`. Toutes
  les sources passent par ce magasin ; le canon dans la clé empêche deux
  hypothèses lexicales concurrentes sur le même span de s'écraser.

(L'ancien magasin de décisions de TYPE global, `data/mwe_type_decisions.jsonl`
— une décision par idiome plutôt que par occurrence — a été supprimé : S3-1 l'a
rendu mort, plus aucune réservation de span ne le consultait ; voir
tools/migrate_sense_fr_mwe_keys.py pour la suppression du fichier.)

Les deux sont consultés avant tout appel LLM (si une clé existe déjà — quel
que soit son statut — on ne rejuge pas : c'est ce qui rend la reprise
gratuite d'un run à l'autre, voir la Partie 3 du plan, "mwe_judge : 322
appels LLM cachés en permanence via le magasin du point C"). Et jamais
écrasés en écriture pour une entrée `status: validated` (réservée à une
future relecture manuelle — pas encore construite dans ce lot, mais le
magasin est déjà compatible : une entrée avec `status: validated` écrite à
la main par un futur outil ne sera jamais recalculée ni reperdue)."""

from __future__ import annotations

import json
from datetime import date

from pipeline import atomic, config

PROTECTED_STATUSES = {"validated"}


def is_protected(entry: dict | None) -> bool:
    return entry is not None and entry.get("status") in PROTECTED_STATUSES


def _load(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    store: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            store[entry["key"]] = entry
    return store


def _write(path, store: dict[str, dict]) -> None:
    config.ensure_data_dir()
    atomic.atomic_write_jsonl(path, (store[key] for key in sorted(store)))


def load_occurrence_store() -> dict[str, dict]:
    return _load(config.MWE_OCCURRENCE_STORE_PATH)


def write_occurrence_store(store: dict[str, dict]) -> None:
    _write(config.MWE_OCCURRENCE_STORE_PATH, store)


def build_entry(key: str, decision: dict) -> dict:
    """Décision fraîchement calculée par le LLM (judge_type/judge_occurrence)
    -> entrée de magasin, statut `auto` (jamais `validated` — réservé à une
    relecture manuelle qui n'existe pas encore pour les MWE)."""

    entry = {
        "key": key,
        "label": decision["label"],
        "confidence": decision["confidence"],
        "reason": decision.get("reason", ""),
        "status": "auto",
        "decided_at": date.today().isoformat(),
    }
    if "wordnet_sense_id" in decision:
        entry["wordnet_sense_id"] = decision["wordnet_sense_id"]
    for field in (
        "verdict", "canonical_form", "pos", "contextual_paraphrase",
        "model_confidence", "confidence_features", "evidence",
    ):
        if field in decision:
            entry[field] = decision[field]
    return entry
