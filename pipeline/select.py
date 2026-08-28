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
from nltk.corpus.reader.wordnet import WordNetError

from pipeline import atomic, config, inventory, lexicon, zones
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


def load_zone_map() -> dict[int, str]:
    """Lot 5 (point H) : charge le layout de zones écrit par `analyze.py`
    à chaque run. `select.py` tourne toujours après `analyze` dans
    `run_pipeline.py` — un layout absent signifie que `analyze` n'a jamais
    tourné (ou un `pipeline_out/` d'avant ce lot), pas une course."""

    layout = zones.load()
    if layout is None:
        raise SystemExit(
            f"select : {config.ZONE_LAYOUT_PATH.name} absent — lance `analyze` "
            f"(S1) avant cette étape pour calculer le layout de zones (Lot 5)."
        )
    return zones.segment_zone_map(layout)


def annotate_mwe_spans_with_zones(
    mwe_spans_by_segment: dict[int, list[dict]], seg_zone: dict[int, str]
) -> dict[int, list[dict]]:
    """Ajoute `zone_id`/`touched_zone_ids` à chaque span MWE confirmé (point
    H/I) et réécrit `mwe_confirmed_spans.jsonl` en conséquence :
    `mwe_judge.py`, qui produit ce fichier, ne connaît pas le layout de
    zones (seul `select.py` le charge). Un span MWE ne traverse jamais deux
    segments (une occurrence est toujours prise dans un seul segment_idx —
    voir mwe_judge.py::select_mwe_spans), donc `touched_zone_ids` n'a
    aujourd'hui jamais qu'un seul élément ; le champ existe pour la même
    raison que `dispersion`/`occurrence_segment_idxs` ailleurs dans ce
    module : homogénéité de schéma, pas une anticipation de spans
    multi-segments."""

    annotated: dict[int, list[dict]] = {}
    for seg_idx, spans in mwe_spans_by_segment.items():
        zone_id = seg_zone.get(seg_idx)
        annotated[seg_idx] = [
            {**s, "zone_id": zone_id, "touched_zone_ids": [zone_id] if zone_id else []}
            for s in spans
        ]
    atomic.atomic_write_jsonl(
        config.MWE_SPANS_PATH,
        ({"segment_idx": seg_idx, "spans": spans} for seg_idx, spans in annotated.items()),
    )
    return annotated


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


def wordnet_mwe_definition(sense_id: str | None) -> str | None:
    """Glose WordNet choisie par mwe_judge.py (point G) pour un idiome
    absent d'idioms.yml (ex. "wake up", apporté par la fusion VPC — voir
    Lot 3 — jamais présent dans idiomatch). WordNet reste une simple
    source de glose ici ; l'identité est désormais le `sense_id` S3-2."""
    if not sense_id:
        return None
    try:
        return nwn.synset(sense_id).definition()
    except (WordNetError, ValueError):
        return None


def build_mwe_units(mwe_spans_by_segment: dict[int, list[dict]]) -> list[dict]:
    """Les expressions confirmées en S3 n'ont pas de plancher Pknown/CEFR
    (ces ressources sont lexicales, pas phraséologiques) : elles sont
    conservées directement, avec la glose d'idioms.yml comme "sens" —
    ou, à défaut (idiome absent d'idioms.yml), la glose WordNet
    sélectionnée par mwe_judge.py (point G)."""

    by_sense: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for seg_idx, spans in mwe_spans_by_segment.items():
        for s in spans:
            key = (s.get("canonical_form", s["idiom"]), s["pos"], s["sense_id"])
            by_sense[key].append({**s, "segment_idx": seg_idx})

    units = []
    for (canonical_form, pos, sense_id), occs in by_sense.items():
        definitions = {o.get("definition_en") for o in occs if o.get("definition_en")}
        definitions_conflict = len(definitions) > 1
        if len(definitions) == 1:
            definition = next(iter(definitions))
        else:
            # Données anciennes ou cluster incohérent : ne jamais réintroduire
            # implicitement le premier sens d'idioms.yml.
            definition = wordnet_mwe_definition(sense_id)
            if definition is None:
                definition = occs[0].get("contextual_paraphrase")
        units.append({
            "canonical_form": canonical_form,
            "pos": pos,
            "sense_id": sense_id,
            "unit_key": inventory.make_unit_key(
                canonical_form, pos, sense_id, kind="mwe"
            ),
            "sense_id_source": occs[0].get("sense_id_source"),
            "occurrence_ids": sorted(o["occurrence_id"] for o in occs),
            "occurrence_refs": sorted(
                ({"occurrence_id": o["occurrence_id"], "segment_idx": o["segment_idx"]} for o in occs),
                key=lambda r: (r["segment_idx"], r["occurrence_id"]),
            ),
            "label": occs[0]["label"],
            "confidence": occs[0]["confidence"],
            "definition_en": definition,
            "definition_source": occs[0].get("definition_source"),
            "definition_candidate_id": occs[0].get("definition_candidate_id"),
            "definition_needs_review": definitions_conflict or any(
                o.get("definition_needs_review", True) for o in occs
            ),
            "surface_forms": sorted({o["surface"] for o in occs}),
            "occurrence_segment_idxs": sorted({o["segment_idx"] for o in occs}),
            "book_count": len(occs),
            "dispersion": len({o["segment_idx"] for o in occs}),
        })
    return units


def run() -> int:
    config.ensure_out_dir()
    seg_zone = load_zone_map()
    mwe_spans_by_segment = annotate_mwe_spans_with_zones(load_confirmed_mwe_spans(), seg_zone)

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
        unit_key = inventory.make_unit_key(
            entry["lemma"], entry["wn_pos"], None, kind="word"
        )
        for occ in entry["occurrences"]:
            zone_id = seg_zone.get(occ["segment_idx"])
            inventory_rows.append({
                "occurrence_id": occ["occurrence_id"],
                "unit_key": unit_key,
                "unit_kind": "word",
                "canonical_form": entry["lemma"],
                "pos": entry["wn_pos"],
                "sense_id": None,
                "segment_idx": occ["segment_idx"],
                "start_char": occ["start_char"],
                "end_char": occ["end_char"],
                "zone_id": zone_id,
                "touched_zone_ids": [zone_id] if zone_id else [],
                # Copie auto-suffisante pour la future ouverture d'inventaire
                # S5 : aucun second parcours du livre ni de spaCy necessaire.
                "analysis_version": occ.get("analysis_version"),
                "analysis": occ.get("analysis", {
                    "version": "legacy",
                    "primary": {"lemma": occ["lemma"], "upos": occ["upos"],
                                "wn_pos": occ["wn_pos"], "source": "legacy"},
                    "alternatives": [],
                }),
                # Hypothèses S1-2 transmises à S5 pour examen futur. Leur
                # présence seule ne réserve et ne supprime aucun token.
                "multi_token_candidates": occ.get("multi_token_candidates", []),
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
                "unit_key": inventory.make_unit_key(
                    s.get("canonical_form", s["idiom"]), s["pos"],
                    s["sense_id"], kind="mwe"
                ),
                "unit_kind": "mwe",
                "canonical_form": s.get("canonical_form", s["idiom"]),
                "pos": s["pos"],
                "sense_id": s["sense_id"],
                "segment_idx": seg_idx,
                "start_char": s["start_char"],
                "end_char": s["end_char"],
                "zone_id": s["zone_id"],
                "touched_zone_ids": s["touched_zone_ids"],
            })

    digest = inventory.write(inventory_rows)
    print(f"{len(inventory_rows)} occurrences dans l'inventaire figé -> "
          f"{config.LEXICAL_INVENTORY_PATH} ({digest[:12]}...)")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
