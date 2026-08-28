"""Candidats S1 de composés nominaux et d'entités multi-tokens.

Ces lignes sont des hypothèses auditables, jamais des réservations. Seuls
les spans confirmés produits plus tard par S3/S4 peuvent masquer un token.
"""

from __future__ import annotations

import json
from collections import defaultdict

from pipeline import config, rules_plus

SCHEMA_VERSION = "s1-multi-token-v1"

# Q0-3 Phase 6 (fix_pipeline/detection_benchmark/phase6_decision.md) :
# score attribué aux candidats "rules_plus" Groupe A (bornes structurelles,
# zéro jugement sémantique — voir pipeline/rules_plus.py). Calé sur le
# type de correction : une chaîne à trait d'union libre n'a aucun autre
# signal (spacy_ner/spacy_dependency) que la ponctuation elle-même (0.70) ;
# une extension/troncature CORRIGE un candidat déjà scoré par la boucle
# NER/compound ci-dessous, donc hérite du niveau de confiance de la
# famille qu'elle corrige (0.82 = nominal_compound, 0.95 = named_entity).
_RULES_PLUS_GROUP_A_SCORES = {
    "rules_plus_hyphen_chain": 0.70,
    "rules_plus_hyphen_extend": 0.82,
    "rules_plus_possessive_trim": 0.95,
}


def _span_record(segment_idx: int, doc, start: int, end: int, source: str,
                 label: str, score: float) -> dict:
    tokens = list(doc[start:end])
    start_char = tokens[0].idx
    end_char = tokens[-1].idx + len(tokens[-1].text)
    return {
        "candidate_id": f"mt:{segment_idx}:{start_char}:{end_char}",
        "schema_version": SCHEMA_VERSION,
        "segment_idx": segment_idx,
        "surface": doc.text[start_char:end_char],
        "start_char": start_char,
        "end_char": end_char,
        "token_start": tokens[0].i,
        "token_end": tokens[-1].i + 1,
        "member_char_spans": [[t.idx, t.idx + len(t.text)] for t in tokens],
        "candidate_types": [label],
        "score": score,
        "provenance": [{"source": source, "label": label, "score": score}],
    }


def _span_record_from_chars(segment_idx: int, doc, start_char: int, end_char: int,
                             source: str, label: str, score: float) -> dict | None:
    """Même schéma que ``_span_record``, mais construit à partir d'un span
    de CARACTÈRES (les générateurs ``rules_plus`` Groupe A travaillent sur
    le texte brut, pas sur des indices de token — voir
    ``pipeline/rules_plus.py``). Les membres sont tous les tokens dont le
    span tombe entièrement dans ``[start_char, end_char)`` ; abstention
    (``None``) si aucun token ne s'y trouve ou un seul (bruit de
    tokenisation, pas une MWE — même seuil que ``_span_record`` via
    ``detect()::add``)."""
    tokens = [t for t in doc if t.idx >= start_char and t.idx + len(t.text) <= end_char]
    if len(tokens) < 2:
        return None
    return {
        "candidate_id": f"mt:{segment_idx}:{start_char}:{end_char}",
        "schema_version": SCHEMA_VERSION,
        "segment_idx": segment_idx,
        "surface": doc.text[start_char:end_char],
        "start_char": start_char,
        "end_char": end_char,
        "token_start": tokens[0].i,
        "token_end": tokens[-1].i + 1,
        "member_char_spans": [[t.idx, t.idx + len(t.text)] for t in tokens],
        "candidate_types": [label],
        "score": score,
        "provenance": [{"source": source, "label": label, "score": score}],
    }


def detect(doc, segment_idx: int) -> list[dict]:
    """Fusionne NER et dépendances ``compound`` sur un même span physique."""
    found: dict[tuple[int, int], dict] = {}

    def merge(row: dict, label: str, score: float) -> None:
        key = (row["start_char"], row["end_char"])
        old = found.get(key)
        if old is None:
            found[key] = row
            return
        old["provenance"].extend(row["provenance"])
        old["candidate_types"] = sorted(set(old["candidate_types"] + [label]))
        old["score"] = max(old["score"], score)

    def add(start: int, end: int, source: str, label: str, score: float) -> None:
        if end - start < 2:
            return
        merge(_span_record(segment_idx, doc, start, end, source, label, score), label, score)

    for ent in doc.ents:
        add(ent.start, ent.end, "spacy_ner", f"named_entity:{ent.label_}", 0.95)

    # Un groupe de dépendances compound relié au même nom tête. Les bornes
    # doivent être contiguës : on ne transforme pas une enveloppe discontinue
    # en composé et on ne capture pas les adjectifs ordinaires.
    by_head: dict[int, set[int]] = defaultdict(set)
    for token in doc:
        if token.dep_ == "compound" and token.head.i != token.i:
            by_head[token.head.i].update((token.i, token.head.i))
    for member_ids in by_head.values():
        ordered = sorted(member_ids)
        if ordered == list(range(ordered[0], ordered[-1] + 1)):
            add(ordered[0], ordered[-1] + 1, "spacy_dependency",
                "nominal_compound", 0.82)

    # Q0-3 Phase 6 -- rules_plus Groupe A (bornes structurelles, zéro
    # jugement sémantique) : chaîne à trait d'union libre + extension à
    # gauche d'un candidat déjà trouvé à travers un trait d'union +
    # troncature du possessif. Même statut que le reste de ce module :
    # hypothèses auditables, jamais des réservations (voir
    # pipeline/rules_plus.py et phase3_rules_plus_report.md pour les
    # cas mesurés : "turn-of-the-century", "ground-floor apartment",
    # "New York City's" -> "New York City").
    text_by_segment = {segment_idx: doc.text}
    existing_rows = list(found.values())
    for cand in (
        rules_plus.hyphen_chain_candidates(segment_idx, doc.text)
        + rules_plus.hyphen_extend_existing(existing_rows, text_by_segment)
        + rules_plus.possessive_trim_existing(existing_rows, text_by_segment)
    ):
        label = cand["category"]
        score = _RULES_PLUS_GROUP_A_SCORES[cand["source"]]
        row = _span_record_from_chars(
            segment_idx, doc, cand["start_char"], cand["end_char"],
            cand["source"], label, score,
        )
        if row is not None:
            merge(row, label, score)

    return sorted(found.values(), key=lambda r: (r["start_char"], r["end_char"]))


def validate(candidates: list[dict], source_by_segment: dict[int, str]) -> None:
    for row in candidates:
        source = source_by_segment[row["segment_idx"]]
        if source[row["start_char"]:row["end_char"]] != row["surface"]:
            raise ValueError(f"offsets multi-token invalides pour {row['candidate_id']}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema multi-token invalide pour {row['candidate_id']}")
        if not row.get("provenance") or not 0.0 <= row.get("score", -1) <= 1.0:
            raise ValueError(f"preuve multi-token invalide pour {row['candidate_id']}")


def load_by_segment() -> dict[int, list[dict]]:
    """Lecteur public pour S2+ ; l'absence d'artefact reste compatible."""
    result: dict[int, list[dict]] = defaultdict(list)
    if not config.MULTI_TOKEN_CANDIDATES_PATH.exists():
        return {}
    with config.MULTI_TOKEN_CANDIDATES_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[row["segment_idx"]].append(row)
    return dict(result)


def covering(occurrence: dict, candidates: list[dict]) -> list[dict]:
    """Hypothèses couvrant le token, sans décision ni effet de réservation."""
    return [row for row in candidates
            if row["start_char"] <= occurrence["start_char"]
            and occurrence["end_char"] <= row["end_char"]]
