"""POC — extrait, pour les EXPRESSIONS MULTI-MOTS (MWE) uniquement, les
candidats produits par la chaîne de détection DÉTERMINISTE de prod (S0 -> S1
-> S2, sans le juge LLM S3), puis pour chaque expression retenue la liste
des phrases du livre qui la contiennent (sous une de ses formes de surface)
et leur nombre. Pendant du script pour les mots simples,
POC/traitement_word/claude/extract_word_contexts.py, avec la même structure
de sortie (moins les colonnes métriques Pknown/CEFR/Zipf/AoA, qui ne
s'appliquent qu'aux mots).

Ne touche à rien dans pipeline/ ni pipeline_out/ : script autonome, jetable,
hors pipeline de production. Rejoue en mémoire la chaîne suivante, en
réutilisant directement les fonctions de prod (jamais recopiées) :

  S0. `pipeline.corpus.load_segments(book_path)` — découpe le livre en
      Segment (réplique/didascalie), hors-œuvre exclu.
  S1. `pipeline.analyze.analyze_segments(...)` — épuisé une seule fois pour
      remplir les sinks VPC et rules_plus (mêmes détecteurs, même boucle
      nlp.pipe qu'en prod, voir pipeline/analyze.py:196-317). Les deux JSONL
      obtenus sont écrits dans CE dossier (jamais pipeline_out/), puis
      `pipeline.config.VPC_CANDIDATES_PATH` / `RULES_PLUS_CANDIDATES_PATH`
      sont redirigés LOCALEMENT (le temps d'un appel) vers eux, pour
      pouvoir appeler `pipeline.mwe.load_vpc_candidates`/
      `load_rules_plus_candidates` SANS EN DUPLIQUER la logique de
      projection — restaurés immédiatement après (voir
      `_load_vpc_and_rules_plus_candidates` ci-dessous).
  S2. `pipeline.mwe.find_candidates` (idiomatch segment par segment, portes
      `mwe_gates.classify` et alignement des membres `mwe_alignment.
      align_members` déjà appliqués à l'intérieur), puis
      `pipeline.mwe.merge_candidate_sources` (priorité idiomatch > VPC >
      rules_plus), `structural_prefilter`, `group_by_type` — exactement la
      chaîne de `pipeline/mwe.py::run()`.

Volontairement ABSENT (contrairement à un run de prod complet) :

  S3. `pipeline.mwe_judge` — le seul étage MWE qui appelle un LLM (2 tâches
      réelles : S3-judge-occurrence et S3-definition-cluster, voir
      pipeline/mwe_judge.py:790 et :509). Sans lui, pas de `label`
      (idiome/phrasal_verb/semi_fige), pas de `confidence`, donc AUCUN
      filtre de qualité sémantique sur les candidats de ce CSV — c'est du
      simple candidat S2, pas une expression confirmée. La colonne
      `sources` sert de signal de tri manuel à la place : `rules_plus`
      n'a aucun pouvoir de rejet (pipeline/mwe.py:291-294, "union avec
      spaCy sans pouvoir de rejet") donc un idiome vu par lui seul est le
      plus suspect ; VPC peut au moins rejeter en syntaxe
      (`rejected_syntax`, déjà filtré par `load_vpc_candidates`) ;
      idiomatch est le plus fiable des trois (voir la priorité de fusion).
      Un futur `translate_mwe_context.py` (pendant de
      `translate_word_context.py`) reste l'endroit naturel pour un
      jugement LLM, séparé de ce script — même découpage que côté mots.

  Pas de chaîne Pknown/CEFR/Zipf/cognat : ces ressources sont lexicales,
  pas phraséologiques, et ne s'appliquent délibérément pas aux MWE en
  production (pipeline/select.py:288-293 ; confirmé aussi par
  REVIEW_FIX_PIPELINE/RAPPORT/rapport_filtrage.md:7, les 498 lignes
  `unit_type == "mwe"` traversant le filtrage sans y être soumises).

Divergence assumée avec extract_word_contexts.py : ce script n'applique PAS
`fix_pipeline/detection_benchmark/tokenizer_boundary_fix.py::
patch_dash_after_punctuation` — ce patch n'est jamais entré en prod
(`pipeline.analyze.get_nlp()` ne l'applique pas) ; l'appliquer ici
décalerait la détection MWE par rapport à ce que voient réellement S1/S2.

Correction, au passage, d'une limite documentée dans
extract_word_contexts.py:66-70 ("VPC et rules_plus [...] hors de portée
d'un appel phrase par phrase isolé") : vraie pour un appel phrase par
phrase, mais un script qui rejoue la boucle SEGMENT PAR SEGMENT via
`analyze.analyze_segments` (comme celui-ci) atteint bien les 3 sources —
voir S1 ci-dessus.

Contexte de phrase (hors chaîne de détection ci-dessus) : la phrase spaCy
couvrant le span détecté (étendue à la phrase suivante si le span la
chevauche) ; si le texte obtenu mesure moins de MIN_SENTENCE_WORDS mots, la
phrase précédente et la phrase suivante du même segment (quand elles
existent) sont incluses avec lui — même règle qu'extract_word_contexts.py.

Usage :
    uv run python POC/traitement_mwe/claude/extract_mwe_contexts.py
    uv run python POC/traitement_mwe/claude/extract_mwe_contexts.py --book "books/Dark Matter - Blake Crouch.txt"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# POC/traitement_mwe/claude/extract_mwe_contexts.py -> POC/ est le parent(2).
# Racine du POC autonome (pas la racine du dépôt vocab-filter) : voir le
# plan "Pipeline POC autonome" — aucune dépendance vers pipeline/ de prod.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_pipeline import analyze as analyze_module  # noqa: E402
from poc_pipeline import atomic, config  # noqa: E402
from poc_pipeline import mwe as mwe_module  # noqa: E402

DEFAULT_BOOK_PATH = ROOT / "books" / "The Humans - Stephen Karam.txt"
DEFAULT_OUT_PATH = Path(__file__).parent / "mwe_contexts.csv"
DEFAULT_GATE_REJECTIONS_OUT_PATH = Path(__file__).parent / "mwe_gate_rejections.csv"
DEFAULT_VPC_CANDIDATES_OUT_PATH = Path(__file__).parent / "vpc_candidates.jsonl"
DEFAULT_RULES_PLUS_CANDIDATES_OUT_PATH = Path(__file__).parent / "rules_plus_candidates.jsonl"

# Phrase courte (hors chaîne de détection ci-dessus) : voir docstring du
# module — identique à extract_word_contexts.py.
MIN_SENTENCE_WORDS = 5


# --------------------------------------------------------------------------
# S1 : candidats VPC + rules_plus, réutilisant pipeline.analyze SANS y
# toucher, en redirigeant localement où pipeline.mwe va les relire.
# --------------------------------------------------------------------------

def load_vpc_and_rules_plus_candidates(
    segments: list, play_segments: list,
    vpc_out_path: Path, rules_plus_out_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Rejoue S1 (voir docstring du module) puis relit ses deux sorties via
    les projecteurs de prod `mwe.load_vpc_candidates`/
    `load_rules_plus_candidates` (pipeline/mwe.py:213, 285), sans dupliquer
    leur logique. `pipeline.config.VPC_CANDIDATES_PATH`/
    `RULES_PLUS_CANDIDATES_PATH` ne sont réaffectés qu'en mémoire, le temps
    de l'appel, puis restaurés — aucun fichier de pipeline/ ni pipeline_out/
    n'est lu ou écrit."""

    vpc_sink: list[dict] = []
    rules_plus_sink: list[dict] = []
    zone_sink: list[int] = []
    for _ in analyze_module.analyze_segments(
        play_segments, vpc_sink, zone_sink, rules_plus_sink=rules_plus_sink,
    ):
        pass  # les occurrences de mots simples yieldées ne servent pas ici

    atomic.atomic_write_jsonl(vpc_out_path, vpc_sink)
    atomic.atomic_write_jsonl(rules_plus_out_path, rules_plus_sink)
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

    return vpc_candidates, rules_plus_candidates


# --------------------------------------------------------------------------
# Contexte de phrase : réplique la règle d'extension d'
# extract_word_contexts.py, mais pour un SPAN (start_char, end_char)
# plutôt qu'un unique token.
# --------------------------------------------------------------------------

def sentence_index_at(sents: list, char_offset: int) -> int:
    """Index, dans `sents` (liste de spaCy Span triés par position), de la
    phrase couvrant `char_offset`. Retombe sur la dernière phrase si
    l'offset tombe exactement sur la borne finale du texte (occurrence en
    toute fin de segment)."""

    for i, s in enumerate(sents):
        if s.start_char <= char_offset < s.end_char:
            return i
    return len(sents) - 1


def phrase_text_for_span(doc, sents: list, start_char: int, end_char: int) -> tuple[str, int]:
    """Texte de contexte pour le span [start_char, end_char) de `doc`, avec
    extension aux phrases voisines si trop court (MIN_SENTENCE_WORDS) — voir
    docstring du module. Renvoie (texte, index de la phrase de tête, pour
    la clé de déduplication)."""

    sent_idx_start = sentence_index_at(sents, start_char)
    sent_idx_end = sentence_index_at(sents, max(start_char, end_char - 1))

    core_start_char = sents[sent_idx_start].start_char
    core_end_char = sents[sent_idx_end].end_char
    core_text = doc.text[core_start_char:core_end_char]

    if len(core_text.split()) < MIN_SENTENCE_WORDS:
        span_start_char = (
            sents[sent_idx_start - 1].start_char if sent_idx_start > 0 else core_start_char
        )
        span_end_char = (
            sents[sent_idx_end + 1].end_char if sent_idx_end < len(sents) - 1 else core_end_char
        )
    else:
        span_start_char, span_end_char = core_start_char, core_end_char

    phrase_text = doc.text[span_start_char:span_end_char].strip()
    return phrase_text, sent_idx_start


# --------------------------------------------------------------------------
# Agrégation par idiome
# --------------------------------------------------------------------------

def aggregate_by_idiom(
    by_type: dict[str, list[dict]], docs_by_segment: dict[int, object],
) -> dict[str, dict]:
    """idiome -> {"sources": set, "surfaces": set, "phrases": {phrase_key: text}}
    — même schéma d'agrégation que `by_lemma` dans extract_word_contexts.py."""

    aggregated: dict[str, dict] = {}
    for idiom, occs in by_type.items():
        entry = aggregated.setdefault(idiom, {"sources": set(), "surfaces": set(), "phrases": {}})
        for occ in occs:
            doc = docs_by_segment.get(occ["segment_idx"])
            if doc is None:
                continue
            sents = list(doc.sents)
            if not sents:
                continue
            phrase_text, sent_idx = phrase_text_for_span(
                doc, sents, occ["start_char"], occ["end_char"]
            )
            if not phrase_text:
                continue
            entry["sources"].add(occ["source"])
            entry["surfaces"].add(occ["surface"])
            phrase_key = (occ["segment_idx"], sents[sent_idx].start_char)
            entry["phrases"][phrase_key] = phrase_text
    return aggregated


def aggregate_rejected_by_idiom(
    idiomatch_rejected: list[dict], docs_by_segment: dict[int, object],
) -> dict[str, dict]:
    """Même agrégation que `aggregate_by_idiom`, pour les candidats
    idiomatch écartés par les portes S2 (`rejected_by` non nul) — pendant
    de `mwe_exclusions.csv` côté mots, ici pour audit (pas un filtre)."""

    aggregated: dict[str, dict] = {}
    for occ in idiomatch_rejected:
        entry = aggregated.setdefault(
            occ["idiom"], {"reasons": set(), "surfaces": set(), "phrases": {}}
        )
        doc = docs_by_segment.get(occ["segment_idx"])
        if doc is None:
            continue
        sents = list(doc.sents)
        if not sents:
            continue
        phrase_text, sent_idx = phrase_text_for_span(
            doc, sents, occ["start_char"], occ["end_char"]
        )
        if not phrase_text:
            continue
        entry["reasons"].add(occ["rejected_by"])
        entry["surfaces"].add(occ["surface"])
        phrase_key = (occ["segment_idx"], sents[sent_idx].start_char)
        entry["phrases"][phrase_key] = phrase_text
    return aggregated


# --------------------------------------------------------------------------
# Extraction principale
# --------------------------------------------------------------------------

def build_mwe_contexts(
    book_path: Path, vpc_out_path: Path, rules_plus_out_path: Path,
    skip_lines: int = 0,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, int]]:
    from poc_pipeline.corpus import load_segments

    segments = load_segments(book_path, skip_lines=skip_lines)
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]

    print("Chargement d'idiomatch (n=2)...")
    idiomatch_raw = list(mwe_module.find_candidates(segments))
    idiomatch_accepted = [c for c in idiomatch_raw if c["rejected_by"] is None]
    idiomatch_rejected = [c for c in idiomatch_raw if c["rejected_by"] is not None]

    vpc_raw, rules_plus_raw = load_vpc_and_rules_plus_candidates(
        segments, play_segments, vpc_out_path, rules_plus_out_path,
    )

    raw = mwe_module.merge_candidate_sources(idiomatch_accepted, vpc_raw, rules_plus_raw)
    filtered = mwe_module.structural_prefilter(raw)
    by_type = mwe_module.group_by_type(filtered)

    print("Segmentation en phrases (nlp.pipe, mêmes réglages que S1)...")
    nlp = analyze_module.get_nlp()
    docs_by_segment = {
        seg.idx: doc
        for seg, doc in zip(play_segments, nlp.pipe((s.en for s in play_segments), batch_size=64))
    }

    aggregated = aggregate_by_idiom(by_type, docs_by_segment)
    rejected_aggregated = aggregate_rejected_by_idiom(idiomatch_rejected, docs_by_segment)

    stats = {
        "idiomatch_raw": len(idiomatch_raw),
        "idiomatch_rejected": len(idiomatch_rejected),
        "vpc_candidates": len(vpc_raw),
        "rules_plus_candidates": len(rules_plus_raw),
        "merged": len(raw),
        "after_structural_prefilter": len(filtered),
        "distinct_idioms": len(by_type),
    }
    return aggregated, rejected_aggregated, stats


# --------------------------------------------------------------------------
# Écriture CSV
# --------------------------------------------------------------------------

def write_csv(aggregated: dict[str, dict], out_path: Path, max_phrases: int) -> None:
    rows = []
    for idiom, entry in aggregated.items():
        phrases = list(entry["phrases"].values())
        nb_phrases = len(phrases)
        shown = phrases if max_phrases <= 0 else phrases[:max_phrases]
        rows.append(
            (
                idiom,
                "/".join(sorted(entry["sources"])),
                "/".join(sorted(entry["surfaces"])),
                " || ".join(shown),
                nb_phrases,
            )
        )

    rows.sort(key=lambda r: (-r[4], r[0]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_form", "sources", "surface_forms", "contexte_en", "nb_phrases"])
        writer.writerows(rows)


def write_gate_rejections_csv(rejected_aggregated: dict[str, dict], out_path: Path) -> None:
    rows = sorted(
        (
            idiom,
            "/".join(sorted(entry["reasons"])),
            "/".join(sorted(entry["surfaces"])),
            len(entry["phrases"]),
        )
        for idiom, entry in rejected_aggregated.items()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_form", "rejected_by", "surface_forms", "nb_phrases"])
        writer.writerows(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default=str(DEFAULT_BOOK_PATH),
                         help="Chemin du livre .txt (défaut : The Humans)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH),
                         help="Chemin du CSV de sortie")
    parser.add_argument("--gate-rejections-out", default=str(DEFAULT_GATE_REJECTIONS_OUT_PATH),
                         help="Chemin du CSV des candidats idiomatch écartés par les portes S2")
    parser.add_argument("--vpc-candidates-out", default=str(DEFAULT_VPC_CANDIDATES_OUT_PATH),
                         help="Chemin du JSONL intermédiaire des candidats VPC (audit)")
    parser.add_argument("--rules-plus-candidates-out",
                         default=str(DEFAULT_RULES_PLUS_CANDIDATES_OUT_PATH),
                         help="Chemin du JSONL intermédiaire des candidats rules_plus (audit)")
    parser.add_argument("--max-phrases", type=int, default=0,
                         help="Plafond de phrases affichées par idiome dans la colonne "
                              "'phrases' (0 = toutes, défaut). Ne change pas nb_phrases.")
    parser.add_argument("--skip-lines", type=int, default=0,
                         help="Nombre de lignes de tête (hors-œuvre : copyright, sommaire, "
                              "distribution...) à ignorer en plus de la détection par motifs "
                              "(0 = aucune, défaut ; 182 pour le livre complet The Humans).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    book_path = Path(args.book)
    out_path = Path(args.out)
    gate_rejections_out_path = Path(args.gate_rejections_out)
    vpc_out_path = Path(args.vpc_candidates_out)
    rules_plus_out_path = Path(args.rules_plus_candidates_out)

    if not book_path.exists():
        print(f"Livre introuvable : {book_path}")
        return 1

    print(f"Livre : {book_path}")
    aggregated, rejected_aggregated, stats = build_mwe_contexts(
        book_path, vpc_out_path, rules_plus_out_path, skip_lines=args.skip_lines,
    )

    write_csv(aggregated, out_path, args.max_phrases)
    write_gate_rejections_csv(rejected_aggregated, gate_rejections_out_path)

    print()
    print("=== Entonnoir (mêmes compteurs que pipeline/mwe.py::run()) ===")
    print(f"Occurrences idiomatch brutes         : {stats['idiomatch_raw']}")
    print(f"  écartées par les portes S2          : {stats['idiomatch_rejected']}")
    print(f"Occurrences VPC (non rejetées syntaxe) : {stats['vpc_candidates']}")
    print(f"Occurrences rules_plus                 : {stats['rules_plus_candidates']}")
    print(f"Après fusion des 3 sources             : {stats['merged']}")
    print(f"Après pré-filtre structurel            : {stats['after_structural_prefilter']}")
    print(f"Idiomes distincts retenus              : {stats['distinct_idioms']}")
    print()
    print(f"-> {out_path}")
    print(f"-> {gate_rejections_out_path} ({len(rejected_aggregated)} idiome(s) écarté(s) par les portes S2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
