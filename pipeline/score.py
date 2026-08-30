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
from wordfreq import zipf_frequency

from pipeline import config, fr_norm, inventory, lexicon, senses, sense_fr

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


_instance_cache: dict[str, bool] = {}


def is_named_entity_sense(sense_id: str) -> bool:
    """Sens WordNet désignant une ENTITÉ NOMMÉE : il a un
    instance_hypernym ("est une instance de") plutôt qu'un hypernym
    ("est une sorte de"). Marqueur propre de WordNet, porté par le sens
    RETENU et non par le tag spaCy — indispensable ici, le texte étant
    une pièce dont les didascalies/répliques capitalisées font
    mal-tagger spaCy en PROPN des noms communs (voir select.py::gate,
    qui filtre lui aussi mais sur le TYPE, pas le sens). Mesuré sur The
    Humans : 22 unités de vocab.jsonl/913 avaient un sens d'entité
    nommée (scranton, detroit, bethlehem, mary, god...)."""

    if sense_id not in _instance_cache:
        try:
            _instance_cache[sense_id] = bool(nwn.synset(sense_id).instance_hypernyms())
        except Exception:
            _instance_cache[sense_id] = False  # sens illisible : ne pas écarter à l'aveugle
    return _instance_cache[sense_id]


def load_manual_corrections() -> dict[tuple[str, str, str], dict]:
    """Corrections d'occurrences mal groupées par S5 à cause d'un bug
    d'ingestion (tokenisation qui coupe un composé, MWE non détectée...),
    appliquées ICI À L'EXPORT plutôt que de rejouer S1-S5 — voir
    data/manual_corrections.jsonl et le plan du 2026-08-27 "Correction
    manuelle smart-ass / e-mail sans re-run complet". Clé =
    (word, pos, wrong_sense_id) tel que S5 l'a réellement produit ;
    n'affecte donc QUE les occurrences visées, jamais un homographe
    correctement désambiguïsé sous un autre sense_id (ex. "ass"/n vers
    buttocks.n.01 reste intact, seul ass.n.02 est corrigé)."""

    if not config.MANUAL_CORRECTIONS_PATH.exists():
        return {}
    corrections = {}
    with config.MANUAL_CORRECTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            corrections[(c["word"], c["pos"], c["wrong_sense_id"])] = c
    return corrections


def build_records() -> list[dict]:
    with config.SELECTED_TYPES_PATH.open(encoding="utf-8") as f:
        types_by_key = {
            (r["lemma"], r["wn_pos"]): r for r in (json.loads(l) for l in f)
        }
    manual_corrections = load_manual_corrections()

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
            # S5-4 keeps an auditable occurrence record but it is an explicit
            # structural exclusion, not a lexical sense to aggregate/export.
            # The decision lives on this physical occurrence only; autonomous
            # occurrences of the same lemma remain separate lines and survive.
            if (occ.get("resolution_status") == "excluded"
                    and occ.get("exclusion_reason") == "covered_by_confirmed_multi_token"):
                continue
            key = (occ["word"], occ["pos"])
            type_meta = types_by_key.get(key, {})

            best = next(
                (c for c in occ["candidates"] if c["synset"] == occ["best_sense"]), None
            )
            if best is None:
                # Compatibilite avec les anciens senses.jsonl : S5-3 ne
                # permet plus qu'une occurrence incertaine disparaisse parce
                # que son candidat synthétique n'a pas encore ete materialise.
                unresolved_id = senses._stable_recovery_id("unresolved.human_review", occ)
                occ["best_sense"] = unresolved_id
                occ["needs_review"] = True
                occ["recovery"] = {
                    "route": "human_review", "attempts": [],
                    "action": "select another analysis or justify a custom sense",
                }
                best = {"synset": unresolved_id,
                        "definition": "No candidate sense fits this occurrence; human lexical review required.",
                        "fr_hits": []}

            correction = manual_corrections.get((occ["word"], occ["pos"], occ["best_sense"]))
            surface = occ["target_surface"] or occ["word"]
            definition = best["definition"]
            if correction:
                new_pos = correction.get("new_pos", occ["pos"])
                record_key = (correction["canonical_form"], new_pos, correction["new_key"])
                # Reconstruit la vraie forme de surface (ex. "smart-ass",
                # "e-mailed") quand un surface_prefix a été fourni (écrit à
                # la main — voir data/manual_corrections.jsonl) ; à défaut
                # (entrées générées par sense_fr_commit.py depuis
                # reassigner_vers), repli sur canonical_form seul — moins
                # précis mais toujours correct, jamais inventé au-delà de
                # ce que le correcteur a lui-même écrit.
                prefix = correction.get("surface_prefix")
                surface = (prefix + surface) if prefix is not None else correction["canonical_form"]
                definition = correction.get("definition_en") or definition
            else:
                record_key = (occ["word"], occ["pos"], occ["best_sense"])
            records.append({
                "key": record_key,
                "surface": surface,
                "lemma": occ["word"],
                "wn_pos": occ["pos"],
                "sense_id": occ["best_sense"],
                "definition": definition,
                "fr_hits": best["fr_hits"],
                "zipf": type_meta.get("zipf"),
                "pknown": type_meta.get("pknown"),
                "cefr_levels": type_meta.get("cefr_levels", []),
                "segment_idx": occ["segment_idx"],
                "needs_review": occ["needs_review"],
                "margin": occ["margin"],
                "context": occ.get("context"),
                "candidate_senses": [c.get("synset") for c in occ.get("candidates", [])],
                "recovery_route": (occ.get("recovery") or {}).get("route"),
                "recovery_reason": "; ".join(
                    f"{a.get('branch')}:{a.get('status')}"
                    for a in (occ.get("recovery") or {}).get("attempts", [])
                ) or None,
                "review_action": (occ.get("recovery") or {}).get("action"),
            })

    if n_corrupt:
        print(f"({n_corrupt} lignes corrompues ignorées dans {config.SENSES_PATH})")

    return records


def _fr_alt_frequency_key(candidate: str) -> float:
    """Clé de tri décroissant par fréquence d'usage réelle (wordfreq),
    filet de sécurité INDÉPENDANT du modèle sur l'ordre de fr_alt (voir
    le plan §6 : le modèle est déjà chargé de trier par fréquence dans
    sa réponse, ceci ne fait que corriger un ordre implausible). Pour une
    expression à plusieurs mots, le mot de contenu le moins fréquent
    détermine le rang — un candidat n'est "courant" que si tous ses mots
    le sont."""
    words = fr_norm.readable_content_words(candidate)
    if not words:
        return -zipf_frequency(candidate, "fr")
    return -min(zipf_frequency(w, "fr") for w in words)


def _sort_fr_alt(fr_alt: list[str]) -> list[str]:
    return sorted(fr_alt, key=_fr_alt_frequency_key)


def resolve_official_fr(sense_fr_store: dict[str, dict], key: str) -> tuple[list[str], str | None, str | None]:
    """Traduction de référence validée pour une clé du magasin
    data/sense_fr.jsonl (voir pipeline/sense_fr.py,
    pipeline/sense_fr_frontier.py et pipeline/sense_fr_adjudicate.py).
    Statuts utilisés ici : `validated` (relu par un humain), `auto_strong`
    (concordance automatique stricte), `auto_llm` (modèle frontière seul,
    aucune ressource lexicale ne couvrant le sens — voir
    sense_fr_frontier.py), `auto_corroborated` (>=2 signaux indépendants
    dont au moins une source humaine hors-ligne, voir
    sense_fr_adjudicate.py), `auto_judged` (juge sur dossier, confiance
    haute, même module) et `auto_joint` (décision conjointe POS/sense_id
    sur un `pending` structurel, voir sense_fr_reassign.py).
    `pending`/`rejected`/`no_equivalent` ne
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
    if entry["status"] not in (
        "validated", "auto_strong", "auto_llm", "auto_corroborated", "auto_judged", "auto_joint",
    ):
        return [], None, entry["status"]
    fr_lemmas = [entry["fr"]] + _sort_fr_alt(entry.get("fr_alt") or []) if entry.get("fr") else []
    return fr_lemmas, entry.get("fr"), entry["status"]


def aggregate_and_score(records: list[dict]) -> list[dict]:
    sense_fr_store = sense_fr.load_store()
    # Phrases du LIVRE COURANT (jamais lues depuis le magasin permanent,
    # cross-livres — voir sense_fr.format_occurrences_en) : recalculées
    # ici, une seule fois pour tout l'appel, pour que vocab.csv/vocab.jsonl
    # soient auto-suffisants à l'audit sans rouvrir data/sense_fr.jsonl.
    occurrences_by_sense = senses.load_occurrences_by_sense()
    # Ré-indexe aussi sous la clé CORRIGÉE (load_manual_corrections) : sinon
    # une unité manuellement recorrigée (build_records) se retrouve avec un
    # contexte_en vide, puisque senses.jsonl porte toujours l'ancien
    # sense_id — le texte du livre, lui, n'a pas besoin d'être corrigé.
    for correction in load_manual_corrections().values():
        occs = occurrences_by_sense.get(correction["wrong_sense_id"])
        if occs:
            occurrences_by_sense[correction["new_key"]] = occs

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["key"]].append(r)

    units = []
    for key, occs in grouped.items():
        lemma, wn_pos, sense_id = key
        if is_named_entity_sense(sense_id):
            # Re-filtrage au niveau du SENS (docstring du module) : le
            # type a pu passer S4 (select.py::gate ne voit que le tag
            # spaCy, sur-inclusif par construction), mais le sens
            # réellement retenu par S5 est celui d'une entité nommée
            # (ex. scranton.n.01, bethlehem.n.02) — jamais du
            # vocabulaire à apprendre.
            continue
        first = occs[0]

        fr_hits = first["fr_hits"] or []
        official_fr, meaning_fr_official, fr_status = resolve_official_fr(sense_fr_store, sense_id)
        contexte_en = sense_fr.format_occurrences_en(occurrences_by_sense.get(sense_id, []))
        if not contexte_en:
            contexte_en = " | ".join(dict.fromkeys(o["context"] for o in occs if o.get("context")))
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
            "meaning_fr_alt": "/".join(official_fr[1:]) if len(official_fr) > 1 else None,
            "contexte_en": contexte_en,
            "fr_status": fr_status,
            "occurrences": len(occs),
            "book_count": len(occs),
            "dispersion": len({o["segment_idx"] for o in occs}),
            "zipf_need": zipf_need(first["zipf"]),
            "aoa_component": aoa_component,
            "fr_opacity": opacity,
            "sense_surprise": surprise,
            "confidence": 1.0 - sum(1 for o in occs if o["needs_review"]) / len(occs),
            # S6-2 (plan §6) : une traduction officielle encore `pending`/
            # `rejected`/`no_equivalent` (meaning_fr_official vide, voir
            # resolve_official_fr ci-dessus) doit rendre la ligne visible en
            # révision, jamais silencieuse — sans ce terme, une occurrence par
            # ailleurs pleinement confiante (needs_review=False côté S5) sortait
            # de vocab.csv avec une case FR vide mais needs_review=False,
            # absente de review_queue.csv (write_review_queue ne filtre que sur
            # needs_review) : la ligne se présentait comme finalisée sans
            # l'être (cas mesuré : 131 lignes, voir fix_pipeline/plan_action_fix_pipeline.md §6 S6-2).
            "needs_review": any(o["needs_review"] for o in occs) or not meaning_fr_official,
            "candidate_senses": sorted({s for o in occs for s in o.get("candidate_senses", []) if s}),
            "recovery_route": first.get("recovery_route"),
            "recovery_reason": first.get("recovery_reason"),
            "review_action": first.get("review_action"),
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
    # Phrases du livre courant — voir aggregate_and_score. Couvre aussi les
    # clés `mwe:*` : senses.load_occurrences_by_sense() fusionne désormais
    # senses.jsonl (occurrences "word") ET senses.load_mwe_occurrences_by_key()
    # (occurrences "mwe", via selected_mwe.jsonl + mwe_confirmed_spans.jsonl).
    occurrences_by_sense = senses.load_occurrences_by_sense()

    units = []
    for r in raw:
        mwe_opacity = opacity_by_label.get(r["label"], 0.5)
        book_freq = r["book_count"]
        book_gain = _clip01(math.log1p(book_freq) / math.log1p(20))

        mwe_key = r.get("unit_key") or inventory.make_unit_key(
            r["canonical_form"], r["pos"], r["sense_id"], kind="mwe"
        )
        official_fr, meaning_fr_official, fr_status = resolve_official_fr(sense_fr_store, mwe_key)
        contexte_en = sense_fr.format_occurrences_en(occurrences_by_sense.get(mwe_key, []))

        units.append({
            "canonical_form": r["canonical_form"],
            "surface_forms": r["surface_forms"],
            "unit_type": "mwe",
            "pos": r["pos"],
            "sense_id": r["sense_id"],
            "definition_en": r["definition_en"],
            "meaning_fr": None,
            "meaning_fr_official": meaning_fr_official,
            "meaning_fr_alt": "/".join(official_fr[1:]) if len(official_fr) > 1 else None,
            "contexte_en": contexte_en,
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
            # S6-2 : même correctif que le mot simple ci-dessus (aggregate_and_score) —
            # une MWE dont la traduction officielle est encore vide ne doit
            # jamais paraître finalisée dans vocab.csv.
            "needs_review": r["confidence"] < 0.6 or r.get("definition_needs_review", False) or not meaning_fr_official,
            # S6-1 : distinct de "needs_review" ci-dessus (qui mélange confiance
            # ET fiabilité de la définition) — collect_targets() (pipeline/sense_fr.py)
            # a besoin du signal BRUT "définition non validée" seul, pour bloquer
            # un verrouillage automatique côté S6 indépendamment de tout ce que
            # le modèle de traduction déclare lui-même (voir sense_fr.blocks_auto_lock).
            "definition_needs_review": r.get("definition_needs_review", False),
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
