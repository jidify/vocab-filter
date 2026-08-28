"""Adaptateur de normalisation (Phase 2) : projette les schémas de sortie
réels de chaque détecteur de production vers le format de candidat commun
attendu par ``fix_pipeline/detection_benchmark/scorer.py``
(``segment_idx``, ``surface``, ``start_char``, ``end_char``, ``category``,
``source``).

Schémas d'entrée confirmés par lecture du code (pas supposés) :

- ``pipeline.multi_token.detect()`` : liste de dicts avec ``start_char``/
  ``end_char`` déjà absolus dans le segment, ``candidate_types`` (liste de
  labels tels que ``"named_entity:GPE"`` ou ``"nominal_compound"``).
- ``pipeline.mwe.merge_candidate_sources()`` + ``structural_prefilter()``
  (fusion idiomatch + VPC, LE chemin de production réel — voir
  ``pipeline/mwe.py::run()``) : liste de dicts avec ``start_char``/
  ``end_char`` (l'enveloppe complète du match), ``member_char_spans``
  (verbe + particule(s) pour VPC, membres alignés pour idiomatch),
  ``idiom``, ``source`` (``"idiomatch"`` ou ``"vpc"``).
- Occurrences mot-simple : mêmes objets que ``pipeline/analyze.py``
  produit pour ``occurrences.jsonl``, filtrés par les 4 conditions
  EXACTES de ``pipeline/select.py::iter_content_occurrences`` (sans son
  ``is_covered`` — cette étape de réservation appartient à S3/S4, hors
  périmètre de ce benchmark de DÉTECTION) : ``is_alpha`` vrai, ``is_stop``
  faux, ``wn_pos`` non nul, ``len(lemma) >= 3``.

Ce module ne lance aucun détecteur — Phase 2 (``phase2_run_baselines.py``)
s'en charge et lui passe les sorties déjà produites.
"""

from __future__ import annotations

from typing import Any


def normalize_multi_token(rows: list[dict]) -> list[dict[str, Any]]:
    """``pipeline.multi_token.detect()`` -> candidats communs.

    ``category`` est dérivée de ``candidate_types`` pour l'audit humain
    uniquement : le scorer ne s'appuie JAMAIS sur la catégorie du
    candidat pour l'appariement (seule celle du span GOLD compte, voir
    scorer.py) — une dérivation approximative ici n'affecte aucune métrique.
    """
    out = []
    for row in rows:
        types = row.get("candidate_types") or []
        category = (
            "multi_token_entity"
            if any(t.startswith("named_entity") for t in types)
            else "nominal_compound"
        )
        out.append({
            "segment_idx": row["segment_idx"],
            "surface": row["surface"],
            "start_char": row["start_char"],
            "end_char": row["end_char"],
            "category": category,
            "source": "multi_token",
        })
    return out


def normalize_mwe(rows: list[dict]) -> list[dict[str, Any]]:
    """Sortie de ``mwe.structural_prefilter(mwe.merge_candidate_sources(...))``
    -> candidats communs. ``start_char``/``end_char`` sont déjà l'enveloppe
    complète du match (idiomatch : span retourné par le matcher, incluant
    tout objet interposé pour un phrasal verb séparable ; VPC : min/max de
    ``token_char_spans``, voir ``mwe.load_vpc_candidates``) — c'est ce que
    le corpus gold annote comme span unique, donc pas besoin de
    ``member_spans`` pour l'appariement ; on les transmet quand même à
    titre informatif.
    """
    out = []
    for row in rows:
        candidate: dict[str, Any] = {
            "segment_idx": row["segment_idx"],
            "surface": row["surface"],
            "start_char": row["start_char"],
            "end_char": row["end_char"],
            "category": "phrasal_verb_or_idiom",
            "source": row["source"],
            "idiom": row["idiom"],
        }
        members = row.get("member_char_spans")
        if members:
            candidate["member_spans"] = [
                {"start_char": a, "end_char": b} for a, b in members
            ]
        out.append(candidate)
    return out


def is_simple_word_candidate(occ: dict) -> bool:
    """Mêmes 4 conditions EXACTES que
    ``pipeline/select.py::iter_content_occurrences`` (sans son
    ``is_covered`` — voir docstring du module)."""
    return (
        occ["is_alpha"]
        and not occ["is_stop"]
        and occ["wn_pos"] is not None
        and len(occ["lemma"]) >= 3
    )


def normalize_simple_words(occurrences: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "segment_idx": occ["segment_idx"],
            "surface": occ["surface"],
            "start_char": occ["start_char"],
            "end_char": occ["end_char"],
            "category": "simple_word",
            "source": "occurrence_filter",
        }
        for occ in occurrences
        if is_simple_word_candidate(occ)
    ]


def combine(*candidate_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for lst in candidate_lists:
        combined.extend(lst)
    return combined
