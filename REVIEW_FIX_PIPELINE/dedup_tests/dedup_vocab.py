"""Dédoublonnage de vocab_filtered.csv — test hors pipeline (scripts
jetables, aucun code de pipeline/ modifié), voir
REVIEW_FIX_PIPELINE/RAPPORT/rapport_dedoublonnage.md.

Trois familles traitées, mesurées sur pipeline_out/vocab.csv (2974 lignes)
-> REVIEW_FIX_PIPELINE/vocab_filtered.csv (1611 lignes, sortie de
filter_book_vocab.py) :

- A. MWE sur-éclatées par pipeline/mwe_judge.py::assign_sense_ids (196
  lignes / 57 groupes (canon, POS) dupliqués sur 498 MWE) : fusionnées
  par similarité d'EMBEDDING sur `definition_en` (voir
  compare_similarity.py — les embeddings dominent nettement le Jaccard
  utilisé par S3, qui sature sur des gloses synonymes sans mot commun).
- B. Identités non résolues par S5 (senses.py::_stable_recovery_id inclut
  segment_idx dans le hash -> une clé par segment) : fusion systématique,
  pas de seuil.
- C. Sens WordNet voisins d'un même (lemme, POS) (164 lignes / 76 groupes) :
  AUCUNE fusion automatique fiable mesurée (embeddings sur gloses WordNet :
  -7 lignes à 0.70 ; wup_similarity WordNet >= 0.90 : -0). Politique de
  SÉLECTION à la place : garder au plus N sens par (lemme, POS).

IMPORTANT — aucune règle ci-dessous ne lit meaning_fr / meaning_fr_official /
meaning_fr_alt / fr_status : le dédoublonnage doit rester valide avant S6
(traduction). Ces colonnes sont recopiées telles quelles dans la sortie,
prises sur la ligne représentante de chaque fusion, à titre informatif
seulement (voir "Hors périmètre" du plan pour un éventuel second passage
post-traduction).

Usage :
    uv run python REVIEW_FIX_PIPELINE/dedup_tests/dedup_vocab.py
"""

from __future__ import annotations

import csv
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("C:/DOCS/_perso/vocab-filter")
sys.path.insert(0, str(ROOT))

IN_PATH = ROOT / "REVIEW_FIX_PIPELINE" / "vocab_filtered.csv"
OUT_PATH = ROOT / "REVIEW_FIX_PIPELINE" / "vocab_deduped.csv"

from pipeline.mwe_judge import normalize_sense_gloss  # noqa: E402 — réutilisé pour l'audit Jaccard, pas pour fusionner

# --------------------------------------------------
# Champs jamais lus par aucune règle de fusion ci-dessous — vérifié par
# assert_fr_fields_unused() en fin de script, pas seulement déclaré ici.
# --------------------------------------------------

FR_FIELDS = ("meaning_fr", "meaning_fr_official", "meaning_fr_alt", "fr_status")

# --------------------------------------------------
# Famille A — MWE. EMBEDDING_MODEL en cache local
# (~/.cache/huggingface/hub) ; comparé à all-MiniLM-L6-v2 (modèle STS
# anglais dédié) dans compare_similarity.py — résultats proches à 0.60,
# LaBSE légèrement plus agressif et sépare "get it" exactement comme
# attendu (comprendre vs prendre en charge). MWE_SIM_THRESHOLD = 0.60
# est le point de fonctionnement mesuré dans le plan, pas une valeur
# validée sur l'ensemble — à relire via l'échantillon du rapport avant
# adoption définitive.
# --------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/LaBSE"
MWE_SIM_THRESHOLD = 0.60
MWE_LINKAGE = "average"

# Préfixe des sense_id frappés par pipeline/mwe_judge.py::custom_sense_id
# (voir CUSTOM_SENSE_VERSION) — un sense_id qui ne commence PAS par ce
# préfixe est un ID WordNet ou DBnary déjà tranché par ces inventaires ;
# jamais fusionné avec un autre sense_id non-custom, même similaire.
CUSTOM_SENSE_PREFIX = "mwe-custom-v1:"

# --------------------------------------------------
# Famille C — sens WordNet voisins. Pas de fusion automatique (mesuré
# non fiable, voir docstring) : politique de SÉLECTION.
# --------------------------------------------------

MAX_SENSES_PER_LEMMA_POS = 2
MIN_OCCURRENCES_TO_KEEP_EXTRA_SENSE = 2
# Fusion optionnelle par embeddings pour la famille C, mesurée comme
# marginale (-7 lignes à 0.70 sur 164) et laissée désactivée par défaut :
# activer en assignant un seuil (ex. 0.70) pour l'évaluer.
WORD_SIM_THRESHOLD: float | None = None

FIELDNAMES_IN = [
    "canonical_form", "surface_forms", "unit_type", "pos", "sense_id",
    "meaning_fr", "meaning_fr_official", "meaning_fr_alt", "contexte_en",
    "fr_status", "definition_en",
    "occurrences", "book_count", "dispersion",
    "zipf_need", "aoa_component", "fr_opacity", "sense_surprise", "confidence",
    "score_comprehension", "score_reuse", "score_default", "needs_review",
    "recovery_route", "recovery_reason", "candidate_senses", "review_action",
    "pknown_test", "aoa_test", "cefr_test", "zipf_wordfreq_en", "zipf_freqzipfus",
]
FIELDNAMES_OUT = FIELDNAMES_IN + [
    "dedup_family", "dedup_merged_from", "dedup_rule", "dedup_similarity",
]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_rows() -> list[dict]:
    with IN_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["_id"] = i  # identité stable pour dedup_merged_from, indépendante du contenu
        r["needs_review"] = r["needs_review"] == "True"
        for field in ("occurrences", "book_count", "dispersion"):
            r[field] = int(r[field])
        for field in ("zipf_need", "aoa_component", "fr_opacity", "sense_surprise",
                      "confidence", "score_comprehension", "score_reuse", "score_default"):
            r[field] = float(r[field]) if r[field] != "" else 0.0
    return rows


def surface_forms_list(r: dict) -> list[str]:
    return [s for s in r["surface_forms"].split("/") if s]


def candidate_senses_list(r: dict) -> list[str]:
    return [s for s in r["candidate_senses"].split("/") if s]


def is_custom_mwe_sense(sense_id: str) -> bool:
    return sense_id.startswith(CUSTOM_SENSE_PREFIX)


def recompute_scores(zipf_need: float, aoa_component: float, fr_opacity: float,
                      sense_surprise: float, book_count: int) -> tuple[float, float, float]:
    """Mêmes formules que pipeline/score.py:409-422/498-506 — la fusion
    change book_count (donc book_gain), jamais les coefficients."""
    book_gain = _clip01(math.log1p(book_count) / math.log1p(20))
    score_comprehension = (
        0.40 * zipf_need + 0.20 * aoa_component + 0.15 * fr_opacity
        + 0.15 * sense_surprise + 0.10 * book_gain
    )
    reuse = zipf_need
    score_reuse = 0.45 * zipf_need + 0.20 * aoa_component + 0.15 * fr_opacity + 0.20 * reuse
    score_default = 0.5 * score_comprehension + 0.5 * score_reuse
    return score_comprehension, score_reuse, score_default


def merge_rows(members: list[dict], *, family: str, rule: str, similarity: float | str) -> dict:
    """Fusionne un cluster de lignes en une seule, en conservant
    l'identité de la ligne au book_count le plus élevé (départagé par
    préférence pour un sense_id WordNet/DBnary déjà tranché — voir
    enforce_authority_invariant — puis par score_default) pour les
    champs qui ne peuvent pas être agrégés (sense_id, definition_en,
    tous les champs FR — jamais lus pour décider, seulement recopiés)."""
    ranked = sorted(
        members,
        key=lambda r: (-r["book_count"], is_custom_mwe_sense(r["sense_id"]), -r["score_default"]),
    )
    rep = ranked[0]

    total_book_count = sum(r["book_count"] for r in members)
    surface_forms = sorted({s for r in members for s in surface_forms_list(r)})
    candidate_senses = sorted({s for r in members for s in candidate_senses_list(r)})
    contexte_en = " | ".join(dict.fromkeys(r["contexte_en"] for r in members if r["contexte_en"]))
    confidence = (
        sum(r["confidence"] * r["book_count"] for r in members) / total_book_count
        if total_book_count else sum(r["confidence"] for r in members) / len(members)
    )
    zipf_need = sum(r["zipf_need"] * r["book_count"] for r in members) / total_book_count if total_book_count else rep["zipf_need"]
    aoa_component = sum(r["aoa_component"] * r["book_count"] for r in members) / total_book_count if total_book_count else rep["aoa_component"]
    fr_opacity = sum(r["fr_opacity"] * r["book_count"] for r in members) / total_book_count if total_book_count else rep["fr_opacity"]
    sense_surprise = sum(r["sense_surprise"] * r["book_count"] for r in members) / total_book_count if total_book_count else rep["sense_surprise"]

    score_comprehension, score_reuse, score_default = recompute_scores(
        zipf_need, aoa_component, fr_opacity, sense_surprise, total_book_count
    )

    merged = dict(rep)
    merged.update({
        "surface_forms": "/".join(surface_forms),
        "candidate_senses": "/".join(candidate_senses),
        "contexte_en": contexte_en,
        "occurrences": total_book_count,
        "book_count": total_book_count,
        "dispersion": max(r["dispersion"] for r in members),
        "confidence": confidence,
        "zipf_need": zipf_need,
        "aoa_component": aoa_component,
        "fr_opacity": fr_opacity,
        "sense_surprise": sense_surprise,
        "score_comprehension": score_comprehension,
        "score_reuse": score_reuse,
        "score_default": score_default,
        "needs_review": any(r["needs_review"] for r in members),
        "dedup_family": family,
        "dedup_merged_from": "|".join(str(r["_id"]) for r in ranked),
        "dedup_rule": rule,
        "dedup_similarity": similarity,
    })
    return merged


def passthrough(r: dict, family: str = "", rule: str = "", similarity: str = "") -> dict:
    merged = dict(r)
    merged.setdefault("dedup_family", family)
    merged.setdefault("dedup_merged_from", "")
    merged.setdefault("dedup_rule", rule)
    merged.setdefault("dedup_similarity", similarity)
    return merged


# --------------------------------------------------
# Famille A — MWE
# --------------------------------------------------

def cluster_by_embedding(rows: list[dict], embeddings: dict, threshold: float) -> list[list[dict]]:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    if len(rows) == 1:
        return [rows]
    X = np.array([embeddings[r["_id"]] for r in rows])
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - threshold,
        metric="cosine", linkage=MWE_LINKAGE,
    ).fit_predict(X)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r, lab in zip(rows, labels):
        buckets[lab].append(r)
    return list(buckets.values())


def enforce_authority_invariant(clusters: list[list[dict]]) -> list[list[dict]]:
    """Un sense_id WordNet/DBnary (non mwe-custom-v1:*) porte une
    décision déjà tranchée par ces inventaires — jamais fusionné avec un
    AUTRE sense_id non-custom, même dans le même cluster d'embedding.
    Un cluster avec >=2 sense_id non-custom distincts est explosé :
    chaque ligne autoritaire redevient singleton, les lignes custom du
    cluster restent groupées entre elles (probablement le même sens,
    mais on ne devine pas à laquelle des autoritaires elles appartiennent)."""
    result = []
    for cluster in clusters:
        authoritative = [r for r in cluster if not is_custom_mwe_sense(r["sense_id"])]
        custom = [r for r in cluster if is_custom_mwe_sense(r["sense_id"])]
        if len(authoritative) <= 1:
            result.append(cluster)
        else:
            for r in authoritative:
                result.append([r])
            if custom:
                result.append(custom)
    return result


def dedup_mwe(mwe_rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in mwe_rows:
        groups[(r["canonical_form"], r["pos"])].append(r)

    from sentence_transformers import SentenceTransformer

    print(f"  Famille A : chargement {EMBEDDING_MODEL} ...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL)
    defs = [r["definition_en"] or "" for r in mwe_rows]
    vecs = model.encode(defs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    embeddings = {r["_id"]: v for r, v in zip(mwe_rows, vecs)}

    out = []
    n_merges = 0
    n_rows_merged_away = 0
    samples = []
    for key, group in groups.items():
        if len(group) == 1:
            out.append(passthrough(group[0], family="A"))
            continue
        raw_clusters = cluster_by_embedding(group, embeddings, MWE_SIM_THRESHOLD)
        clusters = enforce_authority_invariant(raw_clusters)
        for cluster in clusters:
            if len(cluster) == 1:
                out.append(passthrough(cluster[0], family="A"))
            else:
                n_merges += 1
                n_rows_merged_away += len(cluster) - 1
                merged = merge_rows(
                    cluster, family="A",
                    rule=f"embedding[{EMBEDDING_MODEL.split('/')[-1]}]>={MWE_SIM_THRESHOLD:.2f}",
                    similarity=f">={MWE_SIM_THRESHOLD:.2f}",
                )
                out.append(merged)
                if len(samples) < 20:
                    samples.append((key, cluster, merged))

    stats = {"n_merges": n_merges, "n_rows_merged_away": n_rows_merged_away, "samples": samples}
    return out, stats


# --------------------------------------------------
# Famille B — identités non résolues (S5)
# --------------------------------------------------

def dedup_unresolved(word_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Renvoie (lignes_traitées, lignes_restantes_pour_C, stats)."""
    unresolved = [r for r in word_rows if r["sense_id"].startswith("unresolved.")]
    resolved = [r for r in word_rows if not r["sense_id"].startswith("unresolved.")]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in unresolved:
        groups[(r["canonical_form"], r["pos"])].append(r)

    out = []
    n_merges = 0
    n_rows_merged_away = 0
    for key, group in groups.items():
        if len(group) == 1:
            out.append(passthrough(group[0], family="B"))
            continue
        n_merges += 1
        n_rows_merged_away += len(group) - 1
        merged = merge_rows(group, family="B", rule="unresolved.human_review (une clé par segment, artefact de hash)", similarity="n/a")
        merged["sense_id"] = "unresolved.human_review"
        out.append(merged)

    stats = {"n_merges": n_merges, "n_rows_merged_away": n_rows_merged_away}
    return out, resolved, stats


# --------------------------------------------------
# Famille C — sens WordNet voisins : politique de sélection
# --------------------------------------------------

def select_top_senses(word_rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in word_rows:
        groups[(r["canonical_form"], r["pos"])].append(r)

    out = []
    dropped = []
    for key, group in groups.items():
        if len(group) == 1:
            out.append(passthrough(group[0], family="C"))
            continue
        ranked = sorted(group, key=lambda r: -r["score_default"])
        kept = [ranked[0]]
        for r in ranked[1:]:
            if len(kept) >= MAX_SENSES_PER_LEMMA_POS:
                dropped.append(r)
                continue
            if r["occurrences"] >= MIN_OCCURRENCES_TO_KEEP_EXTRA_SENSE:
                kept.append(r)
            else:
                dropped.append(r)
        for r in kept:
            rule = ("top_sense" if r is kept[0]
                    else f"extra_sense(occurrences>={MIN_OCCURRENCES_TO_KEEP_EXTRA_SENSE})")
            out.append(passthrough(r, family="C", rule=rule))

    stats = {
        "n_groups_reduced": sum(1 for v in groups.values() if len(v) > 1),
        "n_rows_dropped": len(dropped),
        "occurrences_dropped": sum(r["occurrences"] for r in dropped),
        "dropped_sample": dropped[:15],
    }
    return out, stats


def assert_fr_fields_unused() -> None:
    """Vérifie par introspection du code source de ce fichier qu'aucune
    règle de fusion ne référence un champ FR — filet de sécurité en plus
    de la revue humaine, voir la contrainte du plan. Ne scanne que le
    code AU-DESSUS de cette fonction (toute la logique de fusion) pour
    ne pas se déclencher sur ses propres motifs de recherche."""
    source = Path(__file__).read_text(encoding="utf-8")
    source = source.split("def assert_fr_fields_unused", 1)[0]
    # Ignore les lignes de commentaire/docstring qui MENTIONNENT les
    # champs FR pour les exclure (celles-ci sont légitimes) ; ne cherche
    # que du code qui les INDEXE (r["meaning_fr"], r['fr_status']...).
    for field in FR_FIELDS:
        for quote in ('"', "'"):
            pattern = "r[" + quote + field + quote + "]"
            if pattern in source:
                raise RuntimeError(
                    f"violation détectée : {pattern!r} utilisé dans une règle de fusion "
                    f"(dédoublonnage doit rester valide avant S6/traduction)."
                )


def main() -> None:
    assert_fr_fields_unused()

    rows = load_rows()
    total_occurrences_before = sum(r["occurrences"] for r in rows)
    mwe_rows = [r for r in rows if r["unit_type"] == "mwe"]
    word_rows = [r for r in rows if r["unit_type"] == "word"]
    print(f"{len(rows)} lignes lues ({len(word_rows)} word, {len(mwe_rows)} mwe) depuis {IN_PATH}")

    print("\n=== Famille A — MWE (embeddings sur definition_en) ===")
    mwe_out, a_stats = dedup_mwe(mwe_rows)
    print(f"  {len(mwe_rows)} -> {len(mwe_out)} lignes MWE "
          f"({a_stats['n_merges']} fusions, {a_stats['n_rows_merged_away']} lignes absorbées)")

    print("\n=== Famille B — identités non résolues (S5) ===")
    unresolved_out, resolved_words, b_stats = dedup_unresolved(word_rows)
    print(f"  {b_stats['n_merges']} fusions, {b_stats['n_rows_merged_away']} lignes absorbées")

    print("\n=== Famille C — sens WordNet voisins (sélection, pas de fusion) ===")
    word_out, c_stats = select_top_senses(resolved_words)
    print(f"  {c_stats['n_groups_reduced']} groupes réduits, "
          f"{c_stats['n_rows_dropped']} lignes écartées "
          f"({c_stats['occurrences_dropped']} occurrences écartées, jamais redistribuées)")
    print(f"\n  Échantillon de lignes écartées (score_default le plus bas du groupe) :")
    for r in c_stats["dropped_sample"]:
        print(f"     - ({r['canonical_form']!r}, {r['pos']}) {r['sense_id']:24} "
              f"occ={r['occurrences']:>2} score={r['score_default']:.3f} | {(r['definition_en'] or '')[:55]}")

    final_rows = mwe_out + unresolved_out + word_out
    total_occurrences_after_conserving = sum(
        r["occurrences"] for r in mwe_out + unresolved_out + word_out
    )

    # --------------------------------------------------
    # Contrôles
    # --------------------------------------------------
    print("\n=== Contrôles ===")
    conserved_before = total_occurrences_before - c_stats["occurrences_dropped"]
    print(f"  occurrences avant fusion            : {total_occurrences_before}")
    print(f"  occurrences écartées (famille C)     : {c_stats['occurrences_dropped']}")
    print(f"  occurrences conservées (A+B+C gardées): {total_occurrences_after_conserving}")
    assert total_occurrences_after_conserving == conserved_before, (
        f"invariant violé : {total_occurrences_after_conserving} != {conserved_before} "
        f"(une fusion des familles A/B a dû perdre des occurrences)"
    )
    print("  OK — aucune occurrence perdue hors des lignes explicitement écartées (famille C).")

    # cas témoins
    print("\n=== Cas témoins ===")
    by_key = defaultdict(list)
    for r in final_rows:
        by_key[(r["canonical_form"], r["pos"], r["unit_type"])].append(r)
    for canon, pos, unit in [
        ("care package", "NOUN", "mwe"), ("piece of work", "NOUN", "mwe"),
        ("be going to", "VERB", "mwe"), ("have got to", "VERB", "mwe"),
        ("get it", "VERB", "mwe"), ("good", "a", "word"), ("good", "r", "word"),
    ]:
        n = len(by_key.get((canon, pos, unit), []))
        print(f"  ({canon!r}, {pos}, {unit}) -> {n} ligne(s)")

    # échantillon de fusions MWE pour relecture manuelle
    print(f"\n=== Échantillon de fusions MWE (famille A, {len(a_stats['samples'])} sur {a_stats['n_merges']}) ===")
    for key, cluster, merged in a_stats["samples"]:
        print(f"\n  {key} -> sense_id retenu {merged['sense_id']} "
              f"(book_count={merged['book_count']}, {len(cluster)} lignes fusionnées)")
        for r in cluster:
            print(f"     - {r['sense_id']:34} book_count={r['book_count']:>2} | {(r['definition_en'] or '')[:70]}")

    final_rows.sort(key=lambda r: -r["score_default"])
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_OUT, extrasaction="ignore")
        writer.writeheader()
        for r in final_rows:
            row = dict(r)
            row["needs_review"] = bool(row["needs_review"])
            for field in ("zipf_need", "aoa_component", "fr_opacity", "sense_surprise",
                          "confidence", "score_comprehension", "score_reuse", "score_default"):
                row[field] = round(float(row[field]), 3)
            writer.writerow(row)

    print(f"\n{len(rows)} lignes -> {len(final_rows)} lignes (-{len(rows) - len(final_rows)})")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
