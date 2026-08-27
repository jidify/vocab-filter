"""Supprime DÉFINITIVEMENT une unité (sense_id WordNet ou clé `mwe:...`)
et toute trace qui s'y rattache dans `pipeline_out/` — pour les cas où
une entrée n'est pas un mauvais SENS ni un mauvais MOT (voir
pipeline/review_ui.py, blocs A/B) mais n'a simplement rien à faire dans
le vocabulaire du livre : bruit d'entité nommée que les deux gardes
existantes (pipeline/select.py::is_likely_named_entity,
pipeline/score.py::is_named_entity_sense) n'attrapent pas — ex.
`york.n.01` ("the English royal house...") pour des occurrences qui
disent en réalité "New York" ; WordNet ne lui donne pas
d'`instance_hypernyms`, donc aucune des deux gardes ne la voit comme une
entité nommée.

Suppression PONCTUELLE, pas une liste d'exclusion permanente : rien
n'est mémorisé pour empêcher la clé de revenir. Un re-run complet à
partir de S1 (`run_pipeline.py`, qui relit le livre) la recréera à
l'identique. Un re-run à partir de S4 (`--from select`) ne la
recréera PAS, puisque pipeline_out/occurrences.jsonl (ce que select.py
relit) a déjà perdu ses tokens — voir la table de dépendances plus bas.

Piège que ce module existe pour éviter : ne JAMAIS filtrer un fichier
par sous-chaîne "york" — la plupart des lignes qui contiennent ce texte
n'ont rien à voir avec la clé (ex. "New York" dans le `contexte_en` de
14 lignes de sense_fr_adjudication.csv, aucune liée à york.n.01). Tout
le filtrage ci-dessous se fait sur des champs structurés : `best_sense`,
le couple (`word`, `pos`)/(`lemma`, `wn_pos`), ou la colonne CSV `key`.

Complet pour une clé `synset` (le cas visé ici, ex. `york.n.01`) : les
étapes 4-6 ci-dessous portent sur senses.jsonl/selected_types.jsonl/
occurrences.jsonl, qui n'indexent QUE des sense_id WordNet — jamais de
clé `mwe:`. Pour une clé `mwe:...`, ces trois étapes sont donc des
no-op silencieux (0 ligne retirée) : les étapes 1-3/7-9 s'appliquent
bien, mais pipeline_out/selected_mwe.jsonl et mwe_confirmed_spans.jsonl,
qui portent les occurrences réelles d'une expression, ne sont PAS
purgés par ce module — hors du besoin qui l'a motivé (une entité
nommée mal filtrée, toujours un synset dans ce pipeline).

Fichiers touchés, dans l'ordre :
    1. data/sense_fr.jsonl              (magasin permanent)
    2. data/sense_fr.lock.json          (voir pipeline/verify_fr_lock.py —
                                          sinon le verrou échoue au run
                                          suivant sur une clé disparue)
    3. data/manual_corrections.jsonl    (au cas où `key` était une cible
                                          de redirection)
    4. pipeline_out/senses.jsonl        (occ["best_sense"] == key)
    5. pipeline_out/selected_types.jsonl (un type seulement si PLUS
                                          AUCUN enregistrement de
                                          senses.jsonl ne survit pour son
                                          (lemma, wn_pos) — un type peut
                                          porter plusieurs sens)
    6. pipeline_out/occurrences.jsonl   (même condition que 5, tokens
                                          pré-désambiguïsation)
    7. pipeline_out/sense_id_suspects.csv   (colonne `key`)
    8. pipeline_out/sense_fr_adjudication.csv (colonne `key`)
    9. pipeline_out/purge_log.jsonl     (audit append-only, jamais relu
                                          par le pipeline — pas une liste
                                          d'exclusion, juste une trace)
   10. Ré-export (pipeline.export.run) -> vocab.csv/jsonl, review_queue.csv,
       report.md régénérés depuis les fichiers déjà purgés.
   11. pipeline_out/sense_fr_review.csv régénéré depuis le magasin purgé.

pipeline_out/frontier_benchmark_*.jsonl est délibérément laissé intact :
instantané figé d'évaluation (pipeline/eval_frontier_ablation.py), le
modifier fausserait un benchmark passé.

Usage :
    uv run python -m pipeline.purge_unit york.n.01
    uv run python -m pipeline.purge_unit york.n.01 --dry-run
    uv run python -m pipeline.purge_unit york.n.01 --reason "New York, pas un sens à apprendre"
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date

from pipeline import config, sense_fr, senses, verify_fr_lock


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _write_jsonl(path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _read_jsonl_tolerant(path) -> list[tuple[dict | None, str]]:
    """Comme _read_jsonl, mais renvoie (dict_parsé_ou_None, ligne_brute) —
    senses.jsonl porte de rares lignes tronquées par un chevauchement de
    deux runs successifs du batch S5 (voir score.py::build_records,
    n_corrupt), et ce module ne doit filtrer QUE sur des champs qu'il
    peut effectivement lire : une ligne illisible est donc toujours
    conservée telle quelle (raw), jamais reformatée ni supprimée à
    l'aveugle. `strict=False` comme senses.py::load_occurrences_by_sense,
    pour les mêmes textes de livre contenant des caractères de contrôle."""
    if not path.exists():
        return []
    decoder = json.JSONDecoder(strict=False)
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = decoder.decode(stripped)
            except json.JSONDecodeError:
                parsed = None
            out.append((parsed, stripped))
    return out


def _write_raw_lines(path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


# ------------------------------------------------------------------
# Fonctions pures (aucune I/O) — sélection des lignes à retirer.
# Séparées du reste pour être testables directement sur des listes en
# mémoire, voir test_purge_unit.py.
# ------------------------------------------------------------------

def split_senses_by_key(
    entries: list[tuple[dict | None, str]], key: str,
) -> tuple[list[str], set[tuple[str, str]], set[tuple[str, str]]]:
    """Sépare les lignes de senses.jsonl (déjà lues par
    _read_jsonl_tolerant) en (lignes_conservées, (word,pos)_retirés,
    (word,pos)_survivants) — une ligne illisible (`occ is None`) est
    toujours conservée, jamais comptée dans un des deux ensembles."""
    kept_lines: list[str] = []
    removed_word_pos: set[tuple[str, str]] = set()
    remaining_word_pos: set[tuple[str, str]] = set()
    for occ, raw in entries:
        if occ is not None and occ.get("best_sense") == key:
            removed_word_pos.add((occ["word"], occ["pos"]))
            continue
        kept_lines.append(raw)
        if occ is not None:
            remaining_word_pos.add((occ["word"], occ["pos"]))
    return kept_lines, removed_word_pos, remaining_word_pos


def fully_removed_types(
    removed_word_pos: set[tuple[str, str]], remaining_word_pos: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    """(lemma, wn_pos) qui n'ont plus AUCUN enregistrement survivant
    dans senses.jsonl — seuls ceux-là peuvent être retirés de
    selected_types.jsonl/occurrences.jsonl (un type porte souvent
    plusieurs sens)."""
    return removed_word_pos - remaining_word_pos


def filter_types_entries(types_entries: list[dict], types_fully_removed: set[tuple[str, str]]) -> list[dict]:
    return [t for t in types_entries if (t["lemma"], t["wn_pos"]) not in types_fully_removed]


def filter_occurrence_lines(
    entries: list[tuple[dict | None, str]], types_fully_removed: set[tuple[str, str]],
) -> list[str]:
    return [
        raw for occ, raw in entries
        if not (occ is not None and (occ["lemma"], occ["wn_pos"]) in types_fully_removed)
    ]


def filter_csv_rows(rows: list[dict], key: str) -> list[dict]:
    """Jamais un filtre texte : une ligne dont seul `contexte_en`
    mentionne le mot n'est jamais retirée, seule la colonne `key`
    compte."""
    return [r for r in rows if r.get("key") != key]


def _filter_csv_by_key(path, key: str) -> int:
    """Réécrit `path` en retirant les lignes dont la colonne `key` vaut
    `key` (voir filter_csv_rows). Renvoie le nombre de lignes retirées ;
    ne touche pas au fichier s'il n'existe pas."""
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    kept = filter_csv_rows(rows, key)
    removed = len(rows) - len(kept)
    if removed:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
    return removed


def purge(key: str, *, reason: str = "", dry_run: bool = False) -> dict:
    """Supprime `key` et tout ce qui s'y rattache. Lève ValueError si la
    clé est absente de data/sense_fr.jsonl (rien à supprimer — on ne
    devine jamais silencieusement). Renvoie {"key", "removed": {fichier:
    n}} ; `removed` est calculé même en `dry_run` (rien n'est écrit)."""

    store = sense_fr.load_store()
    if key not in store:
        raise ValueError(f"clé absente de {config.SENSE_FR_STORE_PATH} : {key!r}")

    removed: dict[str, int] = {}

    # 1. Magasin permanent.
    del store[key]
    removed["sense_fr_store"] = 1
    if not dry_run:
        sense_fr.write_store(store)

    # 2. Verrou — sinon verify_fr_lock échoue au prochain run sur une
    # clé verrouillée qui a disparu (voir sa docstring).
    lock = verify_fr_lock.load_lock() or {}
    if key in lock:
        removed["lock"] = 1
        if not dry_run:
            del lock[key]
            verify_fr_lock.write_lock(lock)
    else:
        removed["lock"] = 0

    # 3. Corrections manuelles où `key` était source OU cible.
    corrections = _read_jsonl(config.MANUAL_CORRECTIONS_PATH)
    kept_corrections = [
        c for c in corrections
        if c.get("wrong_sense_id") != key and c.get("new_key") != key
    ]
    removed["manual_corrections"] = len(corrections) - len(kept_corrections)
    if not dry_run and removed["manual_corrections"]:
        config.ensure_data_dir()
        _write_jsonl(config.MANUAL_CORRECTIONS_PATH, kept_corrections)

    # 4. senses.jsonl — occ["best_sense"] == key. Les rares lignes
    # corrompues (voir score.py::build_records) sont laissées telles
    # quelles, RAW (jamais reformatées) : on ne peut pas savoir si elles
    # visaient `key`.
    sense_entries = _read_jsonl_tolerant(config.SENSES_PATH)
    kept_sense_lines, removed_word_pos, remaining_word_pos = split_senses_by_key(sense_entries, key)
    removed["senses"] = len(sense_entries) - len(kept_sense_lines)

    types_fully_removed = fully_removed_types(removed_word_pos, remaining_word_pos)

    if not dry_run and removed["senses"]:
        _write_raw_lines(config.SENSES_PATH, kept_sense_lines)

    # 5. selected_types.jsonl — un type seulement si son (lemma, wn_pos)
    # n'a plus AUCUN enregistrement survivant dans senses.jsonl.
    types_entries = _read_jsonl(config.SELECTED_TYPES_PATH)
    kept_types = filter_types_entries(types_entries, types_fully_removed)
    removed["selected_types"] = len(types_entries) - len(kept_types)
    if not dry_run and removed["selected_types"]:
        _write_jsonl(config.SELECTED_TYPES_PATH, kept_types)

    # 6. occurrences.jsonl — même condition (tokens pré-désambiguïsation),
    # même tolérance aux lignes illisibles qu'à l'étape 4.
    occ_entries = _read_jsonl_tolerant(config.OCCURRENCES_PATH)
    kept_occ_lines = filter_occurrence_lines(occ_entries, types_fully_removed)
    removed["occurrences"] = len(occ_entries) - len(kept_occ_lines)
    if not dry_run and removed["occurrences"]:
        _write_raw_lines(config.OCCURRENCES_PATH, kept_occ_lines)

    # 7-8. Fichiers d'audit CSV, filtrés sur la colonne `key` uniquement.
    adjudication_path = config.OUT_DIR / "sense_fr_adjudication.csv"
    if dry_run:
        removed["sense_id_suspects"] = _count_csv_matches(config.SENSE_ID_SUSPECTS_PATH, key)
        removed["sense_fr_adjudication"] = _count_csv_matches(adjudication_path, key)
    else:
        removed["sense_id_suspects"] = _filter_csv_by_key(config.SENSE_ID_SUSPECTS_PATH, key)
        removed["sense_fr_adjudication"] = _filter_csv_by_key(adjudication_path, key)

    # 9. Journal d'audit — jamais relu par le pipeline, pure trace.
    if not dry_run:
        config.ensure_out_dir()
        purge_log_path = config.OUT_DIR / "purge_log.jsonl"
        with purge_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "key": key, "reason": reason,
                "purged_at": date.today().isoformat(), "removed": removed,
            }, ensure_ascii=False) + "\n")

        # 10. Ré-export depuis les fichiers déjà purgés (pur CPU, aucun
        # appel LLM — voir pipeline/export.py::run).
        from pipeline.export import run as export_run
        export_run()

        # 11. File de relecture, régénérée depuis le magasin purgé.
        sense_fr.write_review_csv(store, senses.load_occurrences_by_sense())

    return {"key": key, "removed": removed}


def _count_csv_matches(path, key: str) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        return sum(1 for r in csv.DictReader(f) if r.get("key") == key)


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="clé à supprimer (sense_id WordNet ou 'mwe:...')")
    parser.add_argument("--reason", default="", help="note d'audit (optionnelle)")
    parser.add_argument("--dry-run", action="store_true",
                         help="compte les lignes concernées sans rien écrire")
    args = parser.parse_args()

    try:
        result = purge(args.key, reason=args.reason, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"! {exc}")
        return 1

    label = "à supprimer (dry-run)" if args.dry_run else "supprimé(s)"
    print(f"{args.key!r} — {label} :")
    for fichier, n in result["removed"].items():
        print(f"  {fichier}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
