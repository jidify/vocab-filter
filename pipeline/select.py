"""S4 — Porte de sélection permissive, au niveau du TYPE (lemme, POS).

Corrige deux défauts identifiés dans vocab-filter-resume.md / proposition_1 :

- `Pknown` est un plancher de validité, pas un score continu : 90,6 % des
  mots du livre ont déjà Pknown >= 0.99 (effet de plafond mesuré en §1.5),
  donc il ne peut discriminer que par le bas. On l'utilise ici uniquement
  pour écarter le jargon / les noms propres / le bruit — pas pour classer.
- La jointure CEFR se fait PAR POS (comme word_senses.py, pas comme
  prevalence_test.py) : "water" nom (A1) est exclu, "water" verbe (B2)
  est conservé, comme deux types distincts.

Volontairement sur-inclusif par ailleurs (voir le plan, correction 4) :
un lemme dont le sens dominant est A1 mais qui a un sens rare pertinent
survit ici et sera re-filtré au niveau du SENS en S6, une fois que S5 a
déterminé quel sens est réellement employé.
"""

from __future__ import annotations

import json
from collections import defaultdict

from pipeline import config, lexicon
from pipeline.mwe import get_idiom_definition


def load_confirmed_mwe_spans() -> dict[int, list[dict]]:
    if not config.MWE_SPANS_PATH.exists():
        return {}
    spans_by_segment: dict[int, list[dict]] = {}
    with config.MWE_SPANS_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            spans_by_segment[row["segment_idx"]] = row["spans"]
    return spans_by_segment


def is_covered(occ: dict, spans: list[dict]) -> bool:
    """Un token simple est réservé par une expression confirmée s'il
    tombe entièrement dans un de ses spans (proposition_1 §3.3 :
    "une expression confirmée réserve ses composants uniquement dans
    le span concerné" — une autre occurrence autonome du même mot,
    ailleurs, n'est PAS affectée puisqu'on filtre occurrence par
    occurrence, pas lemme par lemme)."""

    return any(
        s["start_char"] <= occ["start_char"] and occ["end_char"] <= s["end_char"]
        for s in spans
    )


def iter_content_occurrences(mwe_spans_by_segment: dict[int, list[dict]] | None = None):
    mwe_spans_by_segment = mwe_spans_by_segment or {}
    with config.OCCURRENCES_PATH.open(encoding="utf-8") as f:
        for line in f:
            occ = json.loads(line)
            if occ["kind"] == "hors_oeuvre":
                continue
            if not occ["is_alpha"] or occ["is_stop"]:
                continue
            if occ["wn_pos"] is None:
                continue
            if len(occ["lemma"]) < 3:
                continue
            if is_covered(occ, mwe_spans_by_segment.get(occ["segment_idx"], [])):
                continue
            yield occ


def build_types(mwe_spans_by_segment: dict[int, list[dict]] | None = None) -> dict[tuple[str, str], dict]:
    types: dict[tuple[str, str], dict] = {}

    for occ in iter_content_occurrences(mwe_spans_by_segment):
        key = (occ["lemma"], occ["wn_pos"])
        entry = types.setdefault(
            key,
            {
                "lemma": occ["lemma"],
                "wn_pos": occ["wn_pos"],
                "surface_forms": set(),
                "occurrences": [],
                "segments": set(),
            },
        )
        entry["surface_forms"].add(occ["surface"])
        entry["occurrences"].append(occ)
        entry["segments"].add(occ["segment_idx"])

    return types


def gate(entry: dict) -> tuple[bool, dict]:
    """Retourne (garder, métadonnées) pour un type."""

    lemma = entry["lemma"]
    wn_pos = entry["wn_pos"]

    prevalence = lexicon.load_prevalence().get(lemma)
    meta = {
        "pknown": None,
        "nobs": None,
        "zipf": None,
        "prevalence_source": None,
        "cefr_levels": sorted(lexicon.cefr_levels_for(lemma, wn_pos)),
    }

    if prevalence is None:
        # Absent du jeu de données : gardé, signalé (le plan : "ou
        # absent du jeu de données, gardé + signalé"). Peut être un mot
        # rare légitime, ou un artefact d'OCR/nom propre — S6 pourra
        # aussi s'appuyer sur analysis_confidence pour distinguer.
        meta["prevalence_source"] = "absent"
    else:
        meta["pknown"] = prevalence.pknown
        meta["nobs"] = prevalence.nobs
        meta["zipf"] = prevalence.zipf
        meta["prevalence_source"] = "found"

        if prevalence.nobs >= config.MIN_NOBS and prevalence.pknown < config.MIN_PKNOWN:
            return False, meta

    if lexicon.should_exclude_cefr(set(meta["cefr_levels"])):
        return False, meta

    return True, meta


def build_mwe_units(mwe_spans_by_segment: dict[int, list[dict]]) -> list[dict]:
    """Les expressions confirmées en S3 n'ont pas de plancher Pknown/CEFR
    (ces ressources sont lexicales, pas phraséologiques) : elles sont
    conservées directement, avec la glose d'idioms.yml comme "sens"."""

    by_idiom: dict[str, list[dict]] = defaultdict(list)
    for seg_idx, spans in mwe_spans_by_segment.items():
        for s in spans:
            by_idiom[s["idiom"]].append({**s, "segment_idx": seg_idx})

    units = []
    for idiom, occs in by_idiom.items():
        units.append({
            "canonical_form": idiom,
            "label": occs[0]["label"],
            "confidence": occs[0]["confidence"],
            "definition_en": get_idiom_definition(idiom),
            "surface_forms": sorted({o["surface"] for o in occs}),
            "occurrence_segment_idxs": sorted({o["segment_idx"] for o in occs}),
            "book_count": len(occs),
            "dispersion": len({o["segment_idx"] for o in occs}),
        })
    return units


def run() -> int:
    config.ensure_out_dir()
    mwe_spans_by_segment = load_confirmed_mwe_spans()

    types = build_types(mwe_spans_by_segment)

    kept = 0
    dropped = 0

    with config.SELECTED_TYPES_PATH.open("w", encoding="utf-8") as f:
        for key, entry in types.items():
            keep, meta = gate(entry)
            if not keep:
                dropped += 1
                continue

            kept += 1
            record = {
                "lemma": entry["lemma"],
                "wn_pos": entry["wn_pos"],
                "surface_forms": sorted(entry["surface_forms"]),
                "book_count": len(entry["occurrences"]),
                "dispersion": len(entry["segments"]),
                "occurrence_segment_idxs": sorted(entry["segments"]),
                **meta,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{kept} types (mots) conservés, {dropped} exclus (Pknown/CEFR) -> "
          f"{config.SELECTED_TYPES_PATH}")

    mwe_units = build_mwe_units(mwe_spans_by_segment)
    with config.SELECTED_MWE_PATH.open("w", encoding="utf-8") as f:
        for u in mwe_units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(f"{len(mwe_units)} expressions multi-mots confirmées -> {config.SELECTED_MWE_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
