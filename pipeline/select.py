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

from nltk.corpus import wordnet as nwn

from pipeline import atomic, config, inventory, lexicon
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
    tombe entièrement dans un des MEMBRES exacts d'un de ses spans — pas
    dans son enveloppe (défaut A, plan Partie 2 point D) : "turned the
    lantern off" ne réserve que "turned"/"off", jamais "lantern" (mesuré :
    21/259 réservations à tort avant ce correctif, dont "lantern",
    "belongings", "spotlight", "money"...). Une autre occurrence autonome
    du même mot, ailleurs, n'est PAS affectée puisqu'on filtre occurrence
    par occurrence, pas lemme par lemme (proposition_1 §3.3).

    Repli sur l'enveloppe (`start_char`/`end_char`) si un span n'a pas de
    `member_char_spans` — uniquement pour lire un `mwe_confirmed_spans.jsonl`
    généré avant le Lot 1 ; un run frais n'écrit plus que des spans avec
    membres exacts (pipeline/mwe_judge.py::select_mwe_spans)."""

    for s in spans:
        members = s.get("member_char_spans")
        if members:
            if any(a <= occ["start_char"] and occ["end_char"] <= b for a, b in members):
                return True
        elif s["start_char"] <= occ["start_char"] and occ["end_char"] <= s["end_char"]:
            return True
    return False


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


def has_common_synset(lemma: str, wn_pos: str) -> bool:
    """Au moins un synset NON-instance (donc un nom COMMUN, pas une
    entité nommée) pour ce lemme/POS. Sert de garde-fou à
    is_likely_named_entity : un lemme mal-taggé PROPN par spaCy mais
    qui a un sens commun en WordNet (ex. "offstage", "melee", "acne" —
    fréquents dans une pièce de théâtre, où les didascalies/répliques
    capitalisées trompent le tagger) ne doit jamais être écarté ici."""

    wanted = {"a", "s"} if wn_pos == "a" else {wn_pos}
    return any(
        synset.pos() in wanted and not synset.instance_hypernyms()
        for synset in nwn.synsets(lemma)
    )


def is_likely_named_entity(entry: dict, meta: dict) -> bool:
    """Écarte un type SEULEMENT si les trois conditions sont réunies :
    toutes ses occurrences sont taggées PROPN par spaCy, il est absent
    du jeu de données de prévalence (donc pas mesuré comme mot courant
    du français-anglais partagé), ET aucun synset commun n'existe pour
    lui. La conjonction est ce qui rend la garde sûre — chacune seule
    laisserait passer trop de faux positifs (voir has_common_synset).

    Repli conservateur pour le gros du bruit théâtral qu'aucun autre
    filtre n'attrapait : prénoms/toponymes (aimee, deirdre, brigid —
    ce dernier consommait à lui seul 105 arbitrages LLM en S5 pour finir
    'aucun_sens_adapte' 105 fois, WordNet ne connaissant "brigid" que
    comme sainte irlandaise), bruit OCR, onomatopées (ewww, woooooo),
    contractions (coulda, shoulda), marques (verizon, klonopin) et
    formes fléchies mal lemmatisées (mumbled, noticing). Complète, sans
    le remplacer, le re-filtrage au niveau du SENS fait par
    score.py::is_named_entity_sense (ex. "queens", "lord", "god" ne
    sont pas toujours taggés PROPN et passent cette porte-ci)."""

    if meta["prevalence_source"] != "absent":
        return False
    if any(occ["upos"] != "PROPN" for occ in entry["occurrences"]):
        return False
    return not has_common_synset(entry["lemma"], entry["wn_pos"])


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
        # rare légitime — le cas nom propre/artefact d'OCR est écarté
        # juste en dessous par is_likely_named_entity, avant que S6
        # n'ait à le voir.
        meta["prevalence_source"] = "absent"
    else:
        meta["pknown"] = prevalence.pknown
        meta["nobs"] = prevalence.nobs
        meta["zipf"] = prevalence.zipf
        meta["prevalence_source"] = "found"

        if prevalence.nobs >= config.MIN_NOBS and prevalence.pknown < config.MIN_PKNOWN:
            meta["drop_reason"] = "pknown"
            return False, meta

    if is_likely_named_entity(entry, meta):
        meta["drop_reason"] = "named_entity"
        return False, meta

    if lexicon.should_exclude_cefr(set(meta["cefr_levels"])):
        meta["drop_reason"] = "cefr"
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
    dropped_by_reason: dict[str, int] = defaultdict(int)
    kept_records = []
    # Lot 3 (point E) : une ligne par occurrence RETENUE, mot simple ou MWE
    # — voir pipeline/inventory.py. Alimenté ici (mots) et plus bas (MWE),
    # jamais par les occurrences exclues par gate()/is_covered().
    inventory_rows: list[dict] = []

    for key, entry in types.items():
        keep, meta = gate(entry)
        if not keep:
            dropped_by_reason[meta["drop_reason"]] += 1
            continue

        kept += 1
        kept_records.append({
            "lemma": entry["lemma"],
            "wn_pos": entry["wn_pos"],
            "surface_forms": sorted(entry["surface_forms"]),
            "book_count": len(entry["occurrences"]),
            "dispersion": len(entry["segments"]),
            "occurrence_segment_idxs": sorted(entry["segments"]),
            **meta,
        })
        unit_key = f"{entry['lemma']}:{entry['wn_pos']}"
        for occ in entry["occurrences"]:
            inventory_rows.append({
                "occurrence_id": occ["occurrence_id"],
                "unit_key": unit_key,
                "segment_idx": occ["segment_idx"],
                "start_char": occ["start_char"],
                "end_char": occ["end_char"],
                "zone_id": None,  # Lot 5, pas encore fait — attendu
            })
    atomic.atomic_write_jsonl(config.SELECTED_TYPES_PATH, kept_records)

    dropped = sum(dropped_by_reason.values())
    print(f"{kept} types (mots) conservés, {dropped} exclus "
          f"(pknown={dropped_by_reason['pknown']}, "
          f"entité nommée={dropped_by_reason['named_entity']}, "
          f"cefr={dropped_by_reason['cefr']}) -> "
          f"{config.SELECTED_TYPES_PATH}")

    mwe_units = build_mwe_units(mwe_spans_by_segment)
    atomic.atomic_write_jsonl(config.SELECTED_MWE_PATH, mwe_units)
    print(f"{len(mwe_units)} expressions multi-mots confirmées -> {config.SELECTED_MWE_PATH}")

    for seg_idx, spans in mwe_spans_by_segment.items():
        for s in spans:
            inventory_rows.append({
                "occurrence_id": s["occurrence_id"],
                "unit_key": f"mwe:{s['idiom']}:{s['label']}",
                "segment_idx": seg_idx,
                "start_char": s["start_char"],
                "end_char": s["end_char"],
                "zone_id": None,
            })

    digest = inventory.write(inventory_rows)
    print(f"{len(inventory_rows)} occurrences dans l'inventaire figé -> "
          f"{config.LEXICAL_INVENTORY_PATH} ({digest[:12]}...)")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
