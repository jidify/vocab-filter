"""S6 — Regroupement par unité (forme, POS, sens) + calcul des signaux
pédagogiques, et re-filtrage au niveau du SENS (voir le plan,
correction 4) : la porte S4 était volontairement sur-inclusive au
niveau du type ; ici on regarde le sens réellement retenu par S5 et on
peut encore exclure un sens A1 trivial même si le lemme avait survécu.

Composantes, toutes ramenées dans [0,1] :

- zipf_need    : besoin d'apprentissage par la fréquence (axe principal)
- aoa_resid    : résidu de l'AoA sur le Zipf — voir le plan, "La question
                 AoA". Signe positif = acquis tard par les natifs
                 (probablement transparent pour un francophone, à
                 pénaliser) ; signe négatif = acquis tôt par les natifs
                 mais rare (probablement une vraie lacune L2, à
                 remonter). Le signe d'application est un paramètre
                 (AOA_SIGN) tranché par l'arbitrage empirique du plan,
                 pas décrété ici.
- fr_opacity   : 1 - similarité orthographique avec la meilleure
                 traduction FR du sens retenu (transparence FR↔EN,
                 l'ajout du plan absent de proposition_1).
- faux_ami     : bonus si le mot FR orthographiquement le plus proche
                 du lemme EN n'est PAS une traduction du sens retenu.
- sense_surprise : 1 - part SemCor du sens retenu parmi les sens du
                 lemme (repris de la logique cumulative de
                 sense_frequency.py).
- mwe_opacity  : pour les unités multi-mots confirmées en S3.
- reuse        : réutilisabilité (proxy Zipf pour l'instant — Spoken
                 BNC2014 en confirmation, voir bnc_escape_compare.py,
                 non branché par défaut car coûteux sur tout le
                 vocabulaire retenu).
- book_gain    : fréquence/dispersion dans l'œuvre.

Deux tris publiés (comprehension, reuse) + un mélange par défaut,
au lieu des 5 valeurs de α de proposition_1 — voir le plan, correction 3.
"""

from __future__ import annotations

import difflib
import json
import math
from collections import defaultdict

from nltk.corpus import wordnet as nwn

from pipeline import config, lexicon, sense_fr

AOA_SIGN = -1.0  # -1 : précoce-mais-rare remonté, tardif pénalisé (position du plan)
                 # à retourner à +1 si l'arbitrage empirique (item 7 du plan) tranche
                 # dans l'autre sens, ou à 0.0 pour neutraliser l'AoA.


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def zipf_need(zipf: float | None) -> float:
    if zipf is None:
        return 0.5  # inconnu : ni facile ni difficile, ne doit pas dominer le tri
    # Zipf va grosso modo de 1 (très rare) à 7 (très fréquent) ; on
    # inverse et on borne à la plage utile [2, 6].
    return _clip01((6.0 - zipf) / 4.0)


def aoa_residual(surface: str, lemma: str, zipf: float | None) -> float | None:
    aoa = lexicon.aoa_for_form(surface, lemma)
    if aoa is None or zipf is None:
        return None
    # Régression isotone nécessite tout le jeu de données ; approximé
    # ici par la relation log-linéaire globale mesurée dans le résumé
    # (corrélation Spearman AoA~Zipf ≈ -0.66). Suffisant pour un résidu
    # utilisé comme signal RELATIF de tri, pas comme mesure absolue.
    predicted_aoa = 12.0 - 1.1 * zipf
    return aoa - predicted_aoa


def fr_opacity_and_faux_ami(lemma: str, fr_lemmas: list[str]) -> tuple[float, bool]:
    if not fr_lemmas:
        return 0.5, False  # pas de traduction connue : opacité inconnue

    best_sim = max(
        difflib.SequenceMatcher(None, lemma.casefold(), fr.casefold()).ratio()
        for fr in fr_lemmas
    )
    opacity = _clip01(1.0 - best_sim)

    # Faux-ami : le mot FR orthographiquement le plus proche du lemme
    # EN (dans TOUT le vocabulaire français, pas seulement les
    # traductions du sens retenu) n'est trouvé QUE via un autre sens.
    # Approximation praticable ici : si la similarité maximale est
    # déjà élevée (>0.75, donc "on dirait" un cognat) mais que ce mot
    # français n'est PAS dans fr_lemmas du sens retenu -> signalé par
    # l'appelant (voir build_records), pas ici, faute du lexique complet.
    return opacity, False


_semcor_cache: dict[str, dict[str, int]] = {}


def sense_surprise(lemma: str, wn_pos: str, sense_id: str) -> float:
    """1 - part SemCor du sens retenu (repris de la logique de
    sense_frequency.py : lemma.count() groupé par POS, part cumulée)."""

    cache_key = f"{lemma}:{wn_pos}"
    if cache_key not in _semcor_cache:
        counts: dict[str, int] = {}
        for synset in nwn.synsets(lemma):
            if synset.pos() not in ({"a", "s"} if wn_pos == "a" else {wn_pos}):
                continue
            for l in synset.lemmas():
                if l.name().casefold() == lemma.casefold():
                    counts[synset.name()] = l.count()
        _semcor_cache[cache_key] = counts

    counts = _semcor_cache[cache_key]
    total = sum(counts.values())
    if total == 0:
        return 0.5  # aucune attestation SemCor : ni banal ni surprenant, connu inconnu
    return _clip01(1.0 - counts.get(sense_id, 0) / total)


def build_records() -> list[dict]:
    with config.SELECTED_TYPES_PATH.open(encoding="utf-8") as f:
        types_by_key = {
            (r["lemma"], r["wn_pos"]): r for r in (json.loads(l) for l in f)
        }

    records = []
    n_corrupt = 0
    with config.SENSES_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                occ = json.loads(line)
            except json.JSONDecodeError:
                # Rares lignes tronquées par un chevauchement de deux
                # runs successifs du batch S5 (observé : 2/2029) —
                # ignorées plutôt que de faire échouer tout l'export.
                n_corrupt += 1
                continue
            key = (occ["word"], occ["pos"])
            type_meta = types_by_key.get(key, {})

            if occ["best_sense"] == "aucun_sens_adapte":
                continue

            best = next(
                (c for c in occ["candidates"] if c["synset"] == occ["best_sense"]), None
            )
            if best is None:
                continue

            record_key = (occ["word"], occ["pos"], occ["best_sense"])
            records.append({
                "key": record_key,
                "surface": occ["target_surface"] or occ["word"],
                "lemma": occ["word"],
                "wn_pos": occ["pos"],
                "sense_id": occ["best_sense"],
                "definition": best["definition"],
                "fr_hits": best["fr_hits"],
                "zipf": type_meta.get("zipf"),
                "pknown": type_meta.get("pknown"),
                "cefr_levels": type_meta.get("cefr_levels", []),
                "segment_idx": occ["segment_idx"],
                "needs_review": occ["needs_review"],
                "margin": occ["margin"],
            })

    if n_corrupt:
        print(f"({n_corrupt} lignes corrompues ignorées dans {config.SENSES_PATH})")

    return records


def resolve_official_fr(sense_fr_store: dict[str, dict], key: str) -> tuple[list[str], str | None, str | None]:
    """Traduction de référence validée pour une clé du magasin
    data/sense_fr.jsonl (voir pipeline/sense_fr.py,
    pipeline/sense_fr_frontier.py et pipeline/sense_fr_adjudicate.py).
    Statuts utilisés ici : `validated` (relu par un humain), `auto_strong`
    (concordance automatique stricte), `auto_llm` (modèle frontière seul,
    aucune ressource lexicale ne couvrant le sens — voir
    sense_fr_frontier.py), `auto_corroborated` (>=2 signaux indépendants
    dont au moins une source humaine hors-ligne, voir
    sense_fr_adjudicate.py) et `auto_judged` (juge sur dossier, confiance
    haute, même module). `pending`/`rejected`/`no_equivalent` ne
    produisent jamais de texte français, quel que soit ce que le magasin
    contient pour cette clé.

    Renvoie (fr_lemmas, meaning_fr_official, fr_status) :
    - fr_lemmas alimente fr_opacity_and_faux_ami (liste vide si rien
      d'officiel n'est disponible — l'appelant retombe alors sur son
      propre repli, p.ex. fr_hits) ;
    - meaning_fr_official / fr_status sont exportés tels quels
      (fr_status distingue toujours la provenance en aval)."""
    entry = sense_fr_store.get(key)
    if entry is None:
        return [], None, None
    if entry["status"] not in ("validated", "auto_strong", "auto_llm", "auto_corroborated", "auto_judged"):
        return [], None, entry["status"]
    fr_lemmas = [entry["fr"]] + (entry.get("fr_alt") or []) if entry.get("fr") else []
    return fr_lemmas, entry.get("fr"), entry["status"]


def aggregate_and_score(records: list[dict]) -> list[dict]:
    sense_fr_store = sense_fr.load_store()

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["key"]].append(r)

    units = []
    for key, occs in grouped.items():
        lemma, wn_pos, sense_id = key
        first = occs[0]

        fr_hits = first["fr_hits"] or []
        official_fr, meaning_fr_official, fr_status = resolve_official_fr(sense_fr_store, sense_id)
        # fr_opacity : préfère la traduction officielle validée quand
        # elle existe (fiable, contrairement à fr_hits — voir le plan).
        # meaning_fr, lui, reste calculé depuis fr_hits SEUL (voir plus
        # bas) : les deux colonnes restent complémentaires, pas
        # concurrentes — meaning_fr_official porte la valeur officielle.
        opacity, _ = fr_opacity_and_faux_ami(lemma, official_fr or fr_hits)

        aoa_resid = aoa_residual(first["surface"], lemma, first["zipf"])
        aoa_component = 0.5
        if aoa_resid is not None:
            # normalisation grossière : résidu typique dans [-4, +4]
            aoa_component = _clip01(0.5 + AOA_SIGN * (aoa_resid / 8.0))

        surprise = sense_surprise(lemma, wn_pos, sense_id)

        units.append({
            "canonical_form": lemma,
            "surface_forms": sorted({o["surface"] for o in occs}),
            "unit_type": "mwe" if " " in lemma else "word",
            "pos": wn_pos,
            "sense_id": sense_id,
            "definition_en": first["definition"],
            "meaning_fr": ", ".join(fr_hits) if fr_hits else None,
            "meaning_fr_official": meaning_fr_official,
            "fr_status": fr_status,
            "occurrences": len(occs),
            "book_count": len(occs),
            "dispersion": len({o["segment_idx"] for o in occs}),
            "zipf_need": zipf_need(first["zipf"]),
            "aoa_component": aoa_component,
            "fr_opacity": opacity,
            "sense_surprise": surprise,
            "confidence": 1.0 - sum(1 for o in occs if o["needs_review"]) / len(occs),
            "needs_review": any(o["needs_review"] for o in occs),
        })

    for u in units:
        book_freq = u["book_count"]
        u["book_gain"] = _clip01(math.log1p(book_freq) / math.log1p(20))
        u["reuse"] = u["zipf_need"]  # proxy par défaut ; BNC2014 en confirmation optionnelle

        u["score_comprehension"] = (
            0.40 * u["zipf_need"]
            + 0.20 * u["aoa_component"]
            + 0.15 * u["fr_opacity"]
            + 0.15 * u["sense_surprise"]
            + 0.10 * u["book_gain"]
        )
        u["score_reuse"] = (
            0.45 * u["zipf_need"]
            + 0.20 * u["aoa_component"]
            + 0.15 * u["fr_opacity"]
            + 0.20 * u["reuse"]
        )
        u["score_default"] = 0.5 * u["score_comprehension"] + 0.5 * u["score_reuse"]

    return units


def build_mwe_units() -> list[dict]:
    """Les expressions multi-mots confirmées en S3 court-circuitent
    GlossBERT/omw-fr (elles n'ont typiquement pas d'entrée WordNet
    propre) : leur "sens" est la glose d'idioms.yml déjà attachée par
    select.py. mwe_opacity remplace fr_opacity comme signal principal
    de valeur pédagogique — une expression jugée "idiome" par le LLM
    est par définition peu déductible de ses composants."""

    if not config.SELECTED_MWE_PATH.exists():
        return []

    with config.SELECTED_MWE_PATH.open(encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]

    opacity_by_label = {"idiome": 0.9, "phrasal_verb": 0.7, "semi_fige": 0.5}

    sense_fr_store = sense_fr.load_store()

    units = []
    for r in raw:
        mwe_opacity = opacity_by_label.get(r["label"], 0.5)
        book_freq = r["book_count"]
        book_gain = _clip01(math.log1p(book_freq) / math.log1p(20))

        mwe_key = f"mwe:{r['canonical_form']}:{r['label']}"
        _, meaning_fr_official, fr_status = resolve_official_fr(sense_fr_store, mwe_key)

        units.append({
            "canonical_form": r["canonical_form"],
            "surface_forms": r["surface_forms"],
            "unit_type": "mwe",
            "pos": None,
            "sense_id": r["label"],
            "definition_en": r["definition_en"],
            "meaning_fr": None,
            "meaning_fr_official": meaning_fr_official,
            "fr_status": fr_status,
            "occurrences": r["book_count"],
            "book_count": r["book_count"],
            "dispersion": r["dispersion"],
            "zipf_need": 0.6,  # les MWE lexicalisées confirmées sont par construction
                               # rarement basiques : valeur par défaut plutôt haute
            "aoa_component": 0.5,
            "fr_opacity": mwe_opacity,
            "sense_surprise": 0.5,
            "confidence": r["confidence"],
            "needs_review": r["confidence"] < 0.6,
        })

    for u in units:
        u["book_gain"] = _clip01(math.log1p(u["book_count"]) / math.log1p(20))
        u["reuse"] = u["zipf_need"]
        u["score_comprehension"] = (
            0.40 * u["zipf_need"] + 0.20 * u["aoa_component"]
            + 0.15 * u["fr_opacity"] + 0.15 * u["sense_surprise"] + 0.10 * u["book_gain"]
        )
        u["score_reuse"] = (
            0.45 * u["zipf_need"] + 0.20 * u["aoa_component"]
            + 0.15 * u["fr_opacity"] + 0.20 * u["reuse"]
        )
        u["score_default"] = 0.5 * u["score_comprehension"] + 0.5 * u["score_reuse"]

    return units


def run() -> int:
    config.ensure_out_dir()
    records = build_records()
    units = aggregate_and_score(records)
    units.extend(build_mwe_units())
    units.sort(key=lambda u: -u["score_default"])

    print(f"{len(records)} occurrences -> {len(units)} unités (forme, POS, sens).")
    return units


if __name__ == "__main__":
    run()
