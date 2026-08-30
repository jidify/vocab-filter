"""Script one-shot — reprend `pipeline_out/cache/` (ancien cache disque par
prompt de lot entier, `pipeline/llm_client.py::cache_path_for`) vers
`data/llm_results.sqlite3` (magasin LLM unitaire, voir
`pipeline/llm_store.py`) — plan "décorréler l'appel en lot du stockage
unitaire". Après ce script, un run avec un `batch_size` différent de celui
utilisé au moment de ces appels ne repaie plus les occurrences/clusters déjà
décidés.

Portée — UNIQUEMENT les réponses de LOT (``{"decisions": [...]}``) des deux
tâches S3 :

- **S3-judge-occurrence** : chaque décision porte un ``occurrence_id`` déjà
  présent dans `pipeline_out/mwe_candidates.jsonl` — le payload sémantique
  (`pipeline/mwe_judge.py::_occurrence_payload`) se reconstruit directement
  depuis ce fichier + les segments du livre courant.
- **S3-definition-cluster** : chaque décision porte un ``cluster_id`` —
  reconstruit en regroupant `pipeline_out/mwe_decisions.jsonl` EXACTEMENT
  comme `mwe_judge.assign_cluster_definitions` (canon+POS+sense_id), puis en
  rejouant `mwe_judge._definition_request` (candidats WordNet/DBnary/
  idiomatch, déterministes — recalculer aujourd'hui reproduit la même liste
  qu'au moment de l'appel, sauf changement des ressources lexicales
  sous-jacentes depuis).

Hors périmètre, DÉLIBÉRÉMENT :

- **Les fichiers de cache UNITAIRES** (une réponse scalaire, sans enveloppe
  ``decisions``) : aucun identifiant n'y figure — l'occurrence_id/cluster_id
  n'existait que dans le PROMPT envoyé, jamais écrit sur disque
  (`llm_client.py::_cache_write` ne persiste que la réponse PARSÉE). Rien à
  réattribuer sans rejouer l'appel ; ces fichiers sont comptés et ignorés,
  ce n'est pas un manque de ce script.
- **S5-arbitrate** : `arbitrate_batch` n'a jamais eu d'appelant en
  production (voir sa docstring, `pipeline/senses.py`) — un fichier de
  cache par lot à ce task_id, s'il existe, vient d'un test ou d'un essai
  manuel dont le `request_id` ne correspond à aucune occurrence réelle du
  livre courant. Rien de fiable à reconstruire.
- **`pipeline_out/cache_SVG/`** (archive antérieure à la génération
  courante) : n'est PAS lu par ce script — considérée comme une archive
  morte, voir le plan.

Le nom du modèle n'est PAS dans le fichier de cache (la clé le hashait, ne
le stockait pas en clair) : il doit être passé explicitement via ``--model``
(tous les fichiers de `pipeline_out/cache/` courants viennent du même
backend catgpt sur ce dépôt — voir README.md).

Usage :
    uv run python -m tools.migrate_llm_cache --model catgpt/catgpt-browser --dry-run
    uv run python -m tools.migrate_llm_cache --model catgpt/catgpt-browser
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pipeline import config, llm_store, mwe_judge
from pipeline.corpus import load_segments


def _sniff_task(decisions: list) -> str | None:
    """Devine la tâche d'un fichier de cache en LOT à partir des clés du
    premier élément de ``decisions`` — les 3 tâches en lot présentes dans
    ce cache ont des schémas de réponse mutuellement exclusifs."""
    if not decisions or not isinstance(decisions[0], dict):
        return None
    keys = set(decisions[0])
    if "occurrence_id" in keys and "label" in keys:
        return "S3-judge-occurrence"
    if "cluster_id" in keys and "candidate_id" in keys:
        return "S3-definition-cluster"
    if "request_id" in keys and "selected_sense" in keys:
        return "S5-arbitrate"  # reconnu pour le rapport, jamais migré (hors périmètre)
    return None


def _load_occurrence_index() -> dict[str, tuple[str, dict]]:
    """``occurrence_id -> (idiom, occ)`` depuis
    `pipeline_out/mwe_candidates.jsonl` (candidats PRÉ-jugement — mêmes
    dicts que ceux que `mwe_judge.run()` construit avant tout appel LLM)."""
    index: dict[str, tuple[str, dict]] = {}
    if not config.MWE_CANDIDATES_PATH.exists():
        return index
    with config.MWE_CANDIDATES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            idiom = entry["idiom"]
            for occ in entry["occurrences"]:
                index[occ["occurrence_id"]] = (idiom, occ)
    return index


def _load_definition_clusters(segments_by_idx: dict) -> dict[str, dict]:
    """``cluster_id -> requête reconstruite`` (avec son ``payload``) depuis
    `pipeline_out/mwe_decisions.jsonl` — même regroupement que
    `mwe_judge.assign_cluster_definitions`."""
    clusters: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    if not config.MWE_DECISIONS_PATH.exists():
        return {}
    with config.MWE_DECISIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for occ in record.get("occurrences", []):
                decision = occ.get("occurrence_decision")
                if not decision or not decision.get("sense_id"):
                    continue
                if decision["label"] not in mwe_judge.LEXICALIZED_LABELS:
                    continue
                key = (decision["canonical_form"].casefold().strip(), decision["pos"], decision["sense_id"])
                clusters[key].append(occ)

    requests: dict[str, dict] = {}
    for (canonical_form, pos, sense_id), occurrences in clusters.items():
        cluster_id = f"{canonical_form}|{pos}|{sense_id}"
        requests[cluster_id] = mwe_judge._definition_request(
            canonical_form, pos, occurrences, segments_by_idx, cluster_id=cluster_id,
        )
    return requests


def migrate(*, model: str, dry_run: bool, cache_dir: Path | None = None) -> dict:
    """Explose chaque fichier de cache en LOT de `pipeline_out/cache/` en
    lignes unitaires dans `llm_store` (sauf ``dry_run``). Renvoie un rapport
    ``{compteur: n}`` pour affichage."""
    cache_dir = cache_dir or config.CACHE_DIR
    schema_variant = "default"
    protocol_occurrence = mwe_judge._occurrence_protocol(schema_variant)
    protocol_definition = f"{mwe_judge.S3_PROMPT_VERSION}:{mwe_judge.S3_DECISION_SCHEMA_VERSION}:definition"

    segments_by_idx = {s.idx: s for s in load_segments()}
    occurrence_index = _load_occurrence_index()
    definition_requests = _load_definition_clusters(segments_by_idx)
    wn_candidates_cache: dict[str, list] = {}

    def wn_candidates_for(idiom: str) -> list:
        if idiom not in wn_candidates_cache:
            wn_candidates_cache[idiom] = mwe_judge.wordnet_synset_candidates(idiom)
        return wn_candidates_cache[idiom]

    report = {
        "files_seen": 0, "files_unit_mode_skipped": 0, "files_unrecognized_skipped": 0,
        "S3-judge-occurrence_recovered": 0, "S3-judge-occurrence_skipped_no_match": 0,
        "S3-definition-cluster_recovered": 0, "S3-definition-cluster_skipped_no_match": 0,
        "S5-arbitrate_files_seen_not_migrated": 0,
    }
    rows: list[llm_store.ResultRow] = []

    for cache_file in sorted(cache_dir.glob("*.json")):
        report["files_seen"] += 1
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report["files_unrecognized_skipped"] += 1
            continue
        if not isinstance(raw, dict) or "decisions" not in raw:
            report["files_unit_mode_skipped"] += 1
            continue

        decisions = raw["decisions"]
        task_id = _sniff_task(decisions)

        if task_id == "S3-judge-occurrence":
            for item in decisions:
                occurrence_id = item.get("occurrence_id") if isinstance(item, dict) else None
                match = occurrence_index.get(occurrence_id)
                if match is None:
                    report["S3-judge-occurrence_skipped_no_match"] += 1
                    continue
                idiom, occ = match
                decision = mwe_judge._normalize_occurrence_result(
                    idiom, item, wn_candidates_for(idiom), schema_variant=schema_variant,
                )
                rows.append(llm_store.ResultRow(
                    task_id=task_id, model=model, protocol=protocol_occurrence,
                    unit_id=occurrence_id,
                    payload=mwe_judge._occurrence_payload(idiom, occ, segments_by_idx),
                    result=decision, source=f"migration:{cache_file.name}",
                ))
                report["S3-judge-occurrence_recovered"] += 1

        elif task_id == "S3-definition-cluster":
            for item in decisions:
                cluster_id = item.get("cluster_id") if isinstance(item, dict) else None
                request = definition_requests.get(cluster_id)
                if request is None:
                    report["S3-definition-cluster_skipped_no_match"] += 1
                    continue
                selection = mwe_judge._definition_selection(request, item)
                rows.append(llm_store.ResultRow(
                    task_id=task_id, model=model, protocol=protocol_definition,
                    unit_id=cluster_id, payload=request["payload"],
                    result=selection, source=f"migration:{cache_file.name}",
                ))
                report["S3-definition-cluster_recovered"] += 1

        elif task_id == "S5-arbitrate":
            report["S5-arbitrate_files_seen_not_migrated"] += 1
        else:
            report["files_unrecognized_skipped"] += 1

    if not dry_run and rows:
        llm_store.put_many(rows)
    report["rows_written"] = 0 if dry_run else len(rows)
    report["rows_would_write"] = len(rows) if dry_run else 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Modèle sous lequel écrire ces lignes (provider/nom) — absent des "
                             "fichiers de cache, doit être connu par ailleurs (voir README.md).")
    parser.add_argument("--dry-run", action="store_true",
                        help="N'écrit rien dans data/llm_results.sqlite3 ; affiche seulement le rapport.")
    args = parser.parse_args()

    report = migrate(model=args.model, dry_run=args.dry_run)

    print(f"Fichiers de cache examinés ({config.CACHE_DIR}) : {report['files_seen']}")
    print(f"  unitaires (aucun identifiant, ignorés) : {report['files_unit_mode_skipped']}")
    print(f"  non reconnus (ignorés)                 : {report['files_unrecognized_skipped']}")
    print(f"  S5-arbitrate (hors périmètre, ignorés)  : {report['S5-arbitrate_files_seen_not_migrated']}")
    print("S3-judge-occurrence :")
    print(f"  récupérée(s)             : {report['S3-judge-occurrence_recovered']}")
    print(f"  sans correspondance      : {report['S3-judge-occurrence_skipped_no_match']}")
    print("S3-definition-cluster :")
    print(f"  récupérée(s)             : {report['S3-definition-cluster_recovered']}")
    print(f"  sans correspondance      : {report['S3-definition-cluster_skipped_no_match']}")
    if args.dry_run:
        print(f"--dry-run : {report['rows_would_write']} ligne(s) auraient été écrites dans "
              f"{config.LLM_RESULTS_DB_PATH} — rien n'a été modifié.")
    else:
        print(f"-> {report['rows_written']} ligne(s) écrite(s) dans {config.LLM_RESULTS_DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
