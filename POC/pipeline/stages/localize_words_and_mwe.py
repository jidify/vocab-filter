"""POC — localise A POSTERIORI, dans le livre, les éléments de vocabulaire
déjà retenus par la chaîne `extract_* -> translate_* ->
merge_word_and_mwe_analysis.py` (POC/pipeline/stages/word_and_mwe_analysis.csv),
et calcule pour chacun les tranches de 5 % du livre où il apparaît — pendant
du layout de zones de production (pipeline/zones.py), mais calculé après
coup plutôt que pendant l'analyse S1.

Différence assumée avec le pipeline de prod : `word_and_mwe_analysis.csv` ne
porte plus, pour chaque ligne, la liste des occurrences qui ont servi à
distinguer son sens/sa traduction (ce travail a déjà eu lieu dans
extract_word_contexts.py / extract_mwe_contexts.py puis a été perdu par le
passage en LLM de translate_*). La localisation ici ne peut donc PAS
distinguer les occurrences par sens : elle re-détecte simplement où, dans le
livre, la forme (lemme ou expression) réapparaît. Si un même lemme a
plusieurs lignes dans le CSV fusionné (plusieurs sens/traductions), ces
lignes reçoivent toutes les MÊMES colonnes de localisation.

Ne touche à rien dans pipeline/ ni pipeline_out/ : script autonome, jetable,
hors pipeline de production. Rejoue en mémoire la même chaîne de détection
que POC/pipeline/stages/extract_mwe_contexts.py (S0 -> S1 -> S2, sans
le juge LLM S3), en réutilisant directement les fonctions de prod (jamais
recopiées), à l'exception de `load_vpc_and_rules_plus_candidates`
ci-dessous, reprise telle quelle depuis extract_mwe_contexts.py (import
croisé entre deux scripts POC non idiomatique ici, le nom du dossier n'étant
pas un identifiant Python valide) :

  S0. `pipeline.corpus.load_segments(book_path)` — découpe le livre en
      Segment (réplique/didascalie), hors-œuvre exclu.
  S1. `pipeline.analyze.analyze_segments(...)` — épuisé une seule fois pour
      obtenir, dans le même passage spaCy : les occurrences de mots simples
      (lemma/surface/segment_idx), les sinks VPC et rules_plus (nécessaires
      à S2 ci-dessous), et `zone_sink` — la séquence, dans l'ordre de
      lecture, du segment_idx de CHAQUE token non-espace du livre,
      ponctuation comprise (voir pipeline/zones.py, point I). C'est
      exactement l'entrée qu'attend `zones.build_layout`.
  S2. `pipeline.mwe.find_candidates` (idiomatch) + les projections
      `load_vpc_candidates`/`load_rules_plus_candidates`, fusionnées par
      `pipeline.mwe.merge_candidate_sources` puis `structural_prefilter` et
      `group_by_type` — exactement la chaîne de `pipeline/mwe.py::run()`.

Puis, LA SEULE PARTIE PROPRE À CE SCRIPT :
  - `pipeline.zones.build_layout(zone_sink, source_text, zone_percent)` —
    repris tel quel, calcule le MÊME layout de zones que la production (même
    algorithme, mêmes bornes de segment) ; SEUL `zones.write()` n'est pas
    appelé, pour ne jamais écrire dans pipeline_out/zone_layout.json — le
    layout de ce script est sérialisé à part, dans CE dossier.
  - jointure de chaque ligne du CSV fusionné vers ses occurrences (par
    lemme pour un mot, par forme extraite pour un MWE) puis vers les zones
    couvertes par ces occurrences.

Usage :
    uv run python POC/pipeline/stages/localize_words_and_mwe.py
    uv run python POC/pipeline/stages/localize_words_and_mwe.py --book "books/Dark Matter - Blake Crouch.txt" --in ... --out ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# POC/pipeline/stages/localize_words_and_mwe.py -> POC/ est le parent(2).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_pipeline import analyze as analyze_module  # noqa: E402
from poc_pipeline import atomic, config  # noqa: E402
from poc_pipeline import mwe as mwe_module  # noqa: E402
from poc_pipeline import zones as zones_module  # noqa: E402
from poc_pipeline.corpus import load_segments  # noqa: E402

DEFAULT_BOOK_PATH = ROOT / "books" / "The Humans - Stephen Karam.txt"
DEFAULT_IN_PATH = Path(__file__).parent / "word_and_mwe_analysis.csv"
# Pas de titre de livre en dur dans le nom par défaut (piège documenté dans
# le plan "Pipeline POC autonome" : l'ancien défaut écrasait silencieusement
# le résultat d'un autre livre) — passer --out explicitement reste recommandé.
DEFAULT_OUT_PATH = Path(__file__).parent / "words_and_mwe_localized.csv"
DEFAULT_UNMATCHED_OUT_PATH = Path(__file__).parent / "localisation_unmatched.csv"
DEFAULT_LAYOUT_OUT_PATH = Path(__file__).parent / "zone_layout.json"
DEFAULT_VPC_CANDIDATES_OUT_PATH = Path(__file__).parent / "vpc_candidates.jsonl"
DEFAULT_RULES_PLUS_CANDIDATES_OUT_PATH = Path(__file__).parent / "rules_plus_candidates.jsonl"

# Colonnes attendues du CSV fusionné (POC/pipeline/stages/merge_word_and_mwe_analysis.py::CSV_HEADER).
INPUT_COLUMNS = [
    "type", "lemme", "extracted_form", "lexicalized_form", "mwe_type", "sense",
    "definition_en", "translations", "false_friend", "compositionality",
    "conventionality", "difficulty_for_non_native", "example",
]

LOCATION_COLUMNS = [
    "zone_ids", "zone_ordinals", "zone_ranges_pct",
    "first_zone", "last_zone", "nb_occurrences", "nb_zones", "nb_segments",
]

OUTPUT_COLUMNS = INPUT_COLUMNS + LOCATION_COLUMNS


# --------------------------------------------------------------------------
# S1 : candidats VPC + rules_plus, réutilisant pipeline.analyze SANS y
# toucher, en redirigeant localement où pipeline.mwe va les relire.
# Copié à l'identique de POC/pipeline/stages/extract_mwe_contexts.py
# (load_vpc_and_rules_plus_candidates) — voir la docstring du module.
# --------------------------------------------------------------------------

def load_vpc_and_rules_plus_candidates(
    segments: list, play_segments: list,
    vpc_out_path: Path, rules_plus_out_path: Path,
) -> tuple[list[dict], list[dict], list[dict], list[int]]:
    """Rejoue S1 en une seule passe spaCy et relit VPC/rules_plus via les
    projecteurs de prod `mwe.load_vpc_candidates`/`load_rules_plus_candidates`
    (pipeline/mwe.py:213, 285), sans dupliquer leur logique.
    `pipeline.config.VPC_CANDIDATES_PATH`/`RULES_PLUS_CANDIDATES_PATH` ne
    sont réaffectés qu'en mémoire, le temps de l'appel, puis restaurés —
    aucun fichier de pipeline/ ni pipeline_out/ n'est lu ou écrit.

    Renvoie (occurrences mots simples, vpc_candidates, rules_plus_candidates,
    zone_sink) : la même passe alimente aussi `zone_sink`, l'entrée de
    `zones.build_layout` (voir docstring du module)."""

    vpc_sink: list[dict] = []
    rules_plus_sink: list[dict] = []
    zone_sink: list[int] = []
    word_occurrences = list(analyze_module.analyze_segments(
        play_segments, vpc_sink, zone_sink, rules_plus_sink=rules_plus_sink,
    ))

    atomic.atomic_write_jsonl(vpc_out_path, vpc_sink)
    atomic.atomic_write_jsonl(rules_plus_out_path, rules_plus_sink)
    print(f"{len(word_occurrences)} occurrences de mots simples.")
    print(f"{len(vpc_sink)} candidats VPC (rejets compris) -> {vpc_out_path}")
    print(f"{len(rules_plus_sink)} candidats rules_plus -> {rules_plus_out_path}")

    original_vpc_path = config.VPC_CANDIDATES_PATH
    original_rules_plus_path = config.RULES_PLUS_CANDIDATES_PATH
    try:
        config.VPC_CANDIDATES_PATH = vpc_out_path
        config.RULES_PLUS_CANDIDATES_PATH = rules_plus_out_path
        vpc_candidates = mwe_module.load_vpc_candidates(segments)
        rules_plus_candidates = mwe_module.load_rules_plus_candidates(segments)
    finally:
        config.VPC_CANDIDATES_PATH = original_vpc_path
        config.RULES_PLUS_CANDIDATES_PATH = original_rules_plus_path

    return word_occurrences, vpc_candidates, rules_plus_candidates, zone_sink


# --------------------------------------------------------------------------
# Layout de zones (repris tel quel de pipeline/zones.py — jamais réécrit,
# jamais `zones.write()` : ce script ne doit RIEN écrire dans pipeline_out/).
# --------------------------------------------------------------------------

def build_zone_layout(play_segments: list, zone_sink: list[int], zone_percent: float) -> dict:
    source_text = "\n".join(s.en for s in play_segments)
    return zones_module.build_layout(zone_sink, source_text, zone_percent)


# --------------------------------------------------------------------------
# Index lemme -> segment_idx (mots simples) et idiome -> segment_idx (MWE),
# construits une fois sur l'ensemble du livre.
# --------------------------------------------------------------------------

def build_word_indexes(word_occurrences: list[dict]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """lemma_index : lemme -> [segment_idx, ...] (une entrée par occurrence,
    doublons conservés pour compter nb_occurrences). surface_index : repli
    sur la forme de surface en minuscules, pour les cas où le CSV fusionné
    porte une forme que la lemmatisation de prod n'a pas produite telle
    quelle (le CSV vient d'un pipeline de tokenisation légèrement différent,
    voir la docstring du module)."""

    lemma_index: dict[str, list[int]] = {}
    surface_index: dict[str, list[int]] = {}
    for occ in word_occurrences:
        lemma_index.setdefault(occ["lemma"], []).append(occ["segment_idx"])
        surface_index.setdefault(occ["surface"].casefold(), []).append(occ["segment_idx"])
    return lemma_index, surface_index


def build_mwe_index(
    idiomatch_segments: list, vpc_candidates: list[dict], rules_plus_candidates: list[dict],
) -> dict[str, list[int]]:
    """idiome -> [segment_idx, ...] à partir de la même chaîne de fusion S2
    que pipeline/mwe.py::run() et extract_mwe_contexts.py::build_mwe_contexts."""

    idiomatch_raw = list(mwe_module.find_candidates(idiomatch_segments))
    idiomatch_accepted = [c for c in idiomatch_raw if c["rejected_by"] is None]

    merged = mwe_module.merge_candidate_sources(
        idiomatch_accepted, vpc_candidates, rules_plus_candidates
    )
    filtered = mwe_module.structural_prefilter(merged)
    by_type = mwe_module.group_by_type(filtered)

    mwe_index: dict[str, list[int]] = {
        idiom: [c["segment_idx"] for c in occs] for idiom, occs in by_type.items()
    }
    print(f"{len(idiomatch_raw)} candidats idiomatch bruts, "
          f"{len(idiomatch_accepted)} acceptés par les portes S2.")
    print(f"{len(merged)} candidats fusionnés (idiomatch+VPC+rules_plus), "
          f"{len(filtered)} après pré-filtre structurel, {len(mwe_index)} idiomes distincts.")
    return mwe_index


# --------------------------------------------------------------------------
# segment_idx -> zone : agrégation en colonnes de localisation
# --------------------------------------------------------------------------

def zone_ordinal(zone_id: str) -> int:
    return int(zone_id.rsplit("-", 1)[-1])


def location_columns_for_segments(
    segment_idxs: list[int], seg_zone: dict[int, str], zone_by_id: dict[str, dict],
) -> dict[str, str]:
    """segment_idxs peut contenir des doublons (une entrée par occurrence) :
    nb_occurrences en tient compte, nb_segments/nb_zones non."""

    zone_ids_seen = {seg_zone[s] for s in segment_idxs if s in seg_zone}
    ordered_zone_ids = sorted(zone_ids_seen, key=zone_ordinal)

    ranges = []
    for zid in ordered_zone_ids:
        z = zone_by_id[zid]
        ranges.append(f"{z['actual_start_percent']:.1f}-{z['actual_end_percent']:.1f}")

    ordinals = [zone_ordinal(z) for z in ordered_zone_ids]

    return {
        "zone_ids": "/".join(ordered_zone_ids),
        "zone_ordinals": ",".join(str(o) for o in ordinals),
        "zone_ranges_pct": " | ".join(ranges),
        "first_zone": str(ordinals[0]) if ordinals else "",
        "last_zone": str(ordinals[-1]) if ordinals else "",
        "nb_occurrences": str(len(segment_idxs)),
        "nb_zones": str(len(ordered_zone_ids)),
        "nb_segments": str(len(set(segment_idxs))),
    }


EMPTY_LOCATION_COLUMNS = {col: "" for col in LOCATION_COLUMNS}


# --------------------------------------------------------------------------
# Jointure CSV -> localisation
# --------------------------------------------------------------------------

def localize_rows(
    rows: list[dict],
    lemma_index: dict[str, list[int]],
    surface_index: dict[str, list[int]],
    mwe_index: dict[str, list[int]],
    seg_zone: dict[int, str],
    zone_by_id: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Renvoie (lignes de sortie avec colonnes de localisation ajoutées,
    lignes non retrouvées pour le CSV compagnon)."""

    out_rows: list[dict] = []
    unmatched: list[dict] = []

    for row in rows:
        row_type = row["type"].strip()
        segment_idxs: list[int] | None = None

        if row_type == "word":
            key = row["lemme"].strip().casefold()
            segment_idxs = lemma_index.get(key)
            if segment_idxs is None:
                segment_idxs = surface_index.get(key)
        elif row_type == "mwe":
            key = row["extracted_form"].strip()
            segment_idxs = mwe_index.get(key)
            if segment_idxs is None:
                key = row["lexicalized_form"].strip()
                segment_idxs = mwe_index.get(key)
        else:
            key = None

        out_row = dict(row)
        if segment_idxs:
            out_row.update(location_columns_for_segments(segment_idxs, seg_zone, zone_by_id))
        else:
            out_row.update(EMPTY_LOCATION_COLUMNS)
            unmatched.append({
                "type": row_type,
                "cle": row.get("lemme") or row.get("extracted_form") or "",
                "raison": "type de ligne inconnu" if row_type not in ("word", "mwe")
                          else "aucune occurrence détectée",
            })
        out_rows.append(out_row)

    return out_rows, unmatched


# --------------------------------------------------------------------------
# I/O CSV
# --------------------------------------------------------------------------

def read_input_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in INPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} : colonnes manquantes {missing}")
        return list(reader)


def write_output_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_unmatched_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "cle", "raison"])
        writer.writeheader()
        writer.writerows(rows)


def write_layout_json(layout: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default=str(DEFAULT_BOOK_PATH),
                         help="Chemin du livre .txt (défaut : books/The Humans - Stephen Karam.txt)")
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN_PATH),
                         help="CSV fusionné mots+MWE en entrée (défaut : "
                              "POC/pipeline/stages/word_and_mwe_analysis.csv)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="CSV de sortie")
    parser.add_argument("--unmatched-out", default=str(DEFAULT_UNMATCHED_OUT_PATH),
                         help="CSV des lignes non localisées (audit)")
    parser.add_argument("--layout-out", default=str(DEFAULT_LAYOUT_OUT_PATH),
                         help="JSON du layout de zones calculé (audit, jamais écrit dans pipeline_out/)")
    parser.add_argument("--vpc-candidates-out", default=str(DEFAULT_VPC_CANDIDATES_OUT_PATH),
                         help="JSONL intermédiaire des candidats VPC (audit)")
    parser.add_argument("--rules-plus-candidates-out",
                         default=str(DEFAULT_RULES_PLUS_CANDIDATES_OUT_PATH),
                         help="JSONL intermédiaire des candidats rules_plus (audit)")
    parser.add_argument("--zone-percent", type=float, default=config.ZONE_PERCENT,
                         help=f"Taille des tranches en %% (défaut : {config.ZONE_PERCENT})")
    parser.add_argument("--skip-lines", type=int, default=0,
                         help="Nombre de lignes de tête (hors-œuvre : copyright, sommaire, "
                              "distribution...) à ignorer en plus de la détection par motifs "
                              "(0 = aucune, défaut ; 182 pour le livre complet The Humans). "
                              "Doit être identique à la valeur utilisée pour "
                              "extract_mwe_contexts.py sur le même livre.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    book_path = Path(args.book)
    in_path = Path(args.in_path)
    out_path = Path(args.out)
    unmatched_out_path = Path(args.unmatched_out)
    layout_out_path = Path(args.layout_out)
    vpc_out_path = Path(args.vpc_candidates_out)
    rules_plus_out_path = Path(args.rules_plus_candidates_out)

    if not book_path.exists():
        print(f"Livre introuvable : {book_path}")
        return 1
    if not in_path.exists():
        print(f"CSV d'entrée introuvable : {in_path}")
        return 1

    print(f"Livre : {book_path}")
    rows = read_input_csv(in_path)
    print(f"{len(rows)} lignes lues dans {in_path} "
          f"({sum(1 for r in rows if r['type'] == 'word')} word, "
          f"{sum(1 for r in rows if r['type'] == 'mwe')} mwe).")

    segments = load_segments(book_path, skip_lines=args.skip_lines)
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]
    print(f"{len(segments)} segments ({len(play_segments)} hors hors-œuvre).")

    print("Rejeu S1 (spaCy, mots simples + candidats VPC/rules_plus)...")
    word_occurrences, vpc_candidates, rules_plus_candidates, zone_sink = (
        load_vpc_and_rules_plus_candidates(segments, play_segments, vpc_out_path, rules_plus_out_path)
    )

    print("Calcul du layout de zones (pipeline.zones.build_layout)...")
    layout = build_zone_layout(play_segments, zone_sink, args.zone_percent)
    seg_zone = zones_module.segment_zone_map(layout)
    zone_by_id = {z["zone_id"]: z for z in layout["zones"]}
    write_layout_json(layout, layout_out_path)
    print(f"{layout['zone_count']} zones ({args.zone_percent}%) -> {layout_out_path} "
          f"(layout_id={layout['layout_id'][:19]}...)")

    lemma_index, surface_index = build_word_indexes(word_occurrences)

    print("Rejeu S2 (idiomatch + fusion des sources MWE)...")
    mwe_index = build_mwe_index(segments, vpc_candidates, rules_plus_candidates)

    out_rows, unmatched = localize_rows(
        rows, lemma_index, surface_index, mwe_index, seg_zone, zone_by_id
    )

    write_output_csv(out_rows, out_path)
    write_unmatched_csv(unmatched, unmatched_out_path)

    print(f"{len(out_rows)} lignes écrites -> {out_path}")
    if unmatched:
        print(f"{len(unmatched)} ligne(s) non localisée(s) -> {unmatched_out_path}")
    else:
        print("Toutes les lignes ont été localisées.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
