"""Phase 3 — ``rules_plus`` : générateurs de candidats sans LLM, construits
exactement comme demandé par ``plan_detection_benchmark_funnel.md`` (Phase 3) :

- lexiques MWE existants — déjà couverts par l'union avec la sortie du
  pipeline de production (idiomatch + lexique custom + VPC/PARSEME) faite
  par ``phase3_run_rules_plus.py`` ; rien de nouveau ici, voir sa docstring ;
- **WordNet comme source de candidats seulement** (jamais de pouvoir de
  rejet) : ``wordnet_nominal_candidates`` (composés nominaux/adjectivaux
  multi-mots) et ``wordnet_phrasal_verb_lexicon`` (verbes multi-mots WordNet
  dont les mots après le verbe sont tous des particules d'une classe fermée
  — un complément à PARSEME, voir sa docstring) ;
- **lexique PARSEME** : ``load_parseme_pairs`` relit le magasin gelé déjà
  présent en production (``data/vpc/parseme-en-1.3-train-vpc-lexicon.json``,
  86 entrées TRAIN) — pas un nouveau téléchargement ;
- **patrons de phrasal verbs séparables** : ``scan_phrasal_verb_candidates``,
  un scanner linéaire sur les lemmes/POS spaCy (jamais sur les dépendances —
  spaCy ne reçoit aucun pouvoir de rejet ici), avec fenêtre bornée et arrêt
  sur frontière de proposition (second verbe fléchi rencontré avant la
  particule) ;
- **règles de bornes** : ``hyphen_chain_candidates``/``hyphen_extend_existing``
  (traits d'union), ``possessive_trim_existing`` (possessif), et
  ``crosses_hard_boundary`` (ponctuation de dialogue/frontières de
  proposition), appliquée partout où une nouvelle borne est calculée.

Rien ici ne réimplémente un détecteur de production : PARSEME/VPC restent
consommés tels quels par ``pipeline.mwe``/``pipeline.vpc`` (Phase 2) ; ce
module n'ajoute que des générateurs ABSENTS du pipeline actuel.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from nltk.corpus import wordnet as nwn

from pipeline import config

PARSEME_LEXICON_PATH = config.ROOT / "data" / "vpc" / "parseme-en-1.3-train-vpc-lexicon.json"

# Classe fermée de particules/prépositions valides pour une lecture
# verbe+particule — évite qu'un lemme WordNet composé sans rapport
# ("come_of_age") ou une simple préposition de lieu homographe ne soit lu
# comme une particule. Couvre les particules effectivement observées dans
# le corpus gold (up/down/in/out/on/off/away/back/through/...).
PARTICLE_CLOSED_CLASS = frozenset({
    "up", "down", "in", "out", "on", "off", "away", "back", "through", "over",
    "along", "about", "around", "together", "forward", "ahead", "by", "under",
    "aside", "apart", "across", "behind", "with", "to", "for", "of",
})
_PARTICLE_POS = frozenset({"ADP", "PART", "ADV"})

# Une fenêtre de recherche verbe -> particule bornée à la plus longue
# séparation réellement observée dans le corpus gold ("puts the blanket and
# pan down" = 4 mots interposés, "give something this nice away" = 3) plus
# une marge — évite l'explosion combinatoire d'une fenêtre non bornée.
MAX_GAP_TOKENS = 6

# Ponctuation qui interrompt TOUJOURS une fenêtre verbe->particule ou un
# span calculé (ponctuation de dialogue/didascalie), même sans franchir de
# phrase spaCy — motivé par les pièges "stomps around?—we"/"the Mary
# statue?—we've" du corpus gold (edge_type=dialogue_dash).
_HARD_BOUNDARY_RE = re.compile(r"[?!;:—–\[\]…]|\.\s*\.\s*\.")

POSSESSIVE_SUFFIXES = ("’s", "'s")


def crosses_hard_boundary(text: str, start: int, end: int) -> bool:
    """Ponctuation de dialogue/didascalie (§ règles de bornes du plan,
    Phase 3) : un span ne doit jamais s'étendre à travers ces caractères."""
    return bool(_HARD_BOUNDARY_RE.search(text[start:end]))


# --------------------------------------------------------------------------
# PARSEME (verbe + particule) — magasin gelé déjà utilisé en production par
# pipeline/vpc (voir pipeline/vpc/resources/vpc_reference.py). Relu ici tel
# quel, jamais réécrit.
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_parseme_pairs() -> dict[str, tuple[tuple[str, ...], ...]]:
    data = json.loads(PARSEME_LEXICON_PATH.read_text(encoding="utf-8"))
    by_verb: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for ref in data["references"]:
        by_verb[ref["verb_lemma"]].add(tuple(ref["particle_lemmas"]))
    return {verb: tuple(sorted(seqs)) for verb, seqs in by_verb.items()}


# --------------------------------------------------------------------------
# WordNet — candidats seulement, jamais de pouvoir de rejet (voir plan,
# Phase 3). Deux lexiques distincts : verbes multi-mots (phrasal verbs,
# bornes discontinues possibles) et noms/adjectifs multi-mots (composés,
# bornes contiguës).
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def wordnet_phrasal_verb_lexicon() -> dict[str, tuple[tuple[str, ...], ...]]:
    """Lemmes verbaux WordNet à 2+ mots dont TOUS les mots après le verbe
    appartiennent à ``PARTICLE_CLOSED_CLASS`` — complète PARSEME (86 entrées
    TRAIN seulement, insuffisant seul : vérifié empiriquement absent de
    "figure_out"/"burn_out"/"calm_down"/"clean_up"/"turn_off" etc., tous
    présents comme lemmes WordNet)."""
    by_verb: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for synset in nwn.all_synsets(pos=nwn.VERB):
        for lemma in synset.lemmas():
            words = tuple(lemma.name().casefold().split("_"))
            if len(words) < 2:
                continue
            verb, particles = words[0], words[1:]
            if all(w in PARTICLE_CLOSED_CLASS for w in particles):
                by_verb[verb].add(particles)
    return {verb: tuple(sorted(seqs)) for verb, seqs in by_verb.items()}


@lru_cache(maxsize=1)
def wordnet_nominal_lexicon() -> dict[tuple[str, ...], str]:
    """Lemmes nominaux/adjectivaux WordNet à 2-5 mots (composés/expressions
    figées). Exclut les verbes (traités séparément ci-dessus : un
    verbe+particule a des règles de bornes différentes, objet interposable)."""
    out: dict[tuple[str, ...], str] = {}
    for pos in (nwn.NOUN, nwn.ADJ):
        for synset in nwn.all_synsets(pos=pos):
            for lemma in synset.lemmas():
                words = tuple(lemma.name().casefold().split("_"))
                if 2 <= len(words) <= 5 and all(w.isalpha() for w in words):
                    out.setdefault(words, pos)
    return out


def merged_phrasal_verb_lexicon() -> dict[str, tuple[tuple[str, ...], ...]]:
    """PARSEME ∪ WordNet, sur la même clé (verbe) — c'est CE lexique fusionné
    que ``scan_phrasal_verb_candidates`` interroge : ni l'un ni l'autre seul
    ne suffit (voir docstrings ci-dessus)."""
    merged: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for source in (load_parseme_pairs(), wordnet_phrasal_verb_lexicon()):
        for verb, seqs in source.items():
            merged[verb].update(seqs)
    return {verb: tuple(sorted(seqs)) for verb, seqs in merged.items()}


# --------------------------------------------------------------------------
# Patron de phrasal verb séparable/inséparable — scanner linéaire sur
# lemmes/POS spaCy, JAMAIS sur les dépendances (spaCy ne reçoit aucun
# pouvoir de rejet ici, contrairement à pipeline.vpc). Frontière de
# proposition = second verbe fléchi rencontré avant la particule : arrête
# la recherche (règle de bornes "frontières de propositions" du plan).
# --------------------------------------------------------------------------


def _is_coordination_noise(tok, verb_tok) -> bool:
    """Un token étiqueté VERB/AUX par erreur de tagging POS à l'intérieur
    d'un syntagme coordonné rattaché au verbe de départ lui-même
    (``dep_ == "conj"`` vers ce même verbe) n'est jamais une frontière de
    proposition — c'est une erreur de tag, pas un second verbe fléchi
    indépendant. Cas vérifié sur seg2485 du corpus gold ("puts the
    blanket and pan down") : spaCy (en_core_web_sm) étiquette "pan"
    VERB/conj vers "puts" (au lieu de NOUN/conj vers "blanket"), ce qui
    fait avorter à tort la recherche de la particule "down" sans ce garde-
    fou. Ceci ASSOUPLIT la frontière de proposition (laisse passer un
    candidat que spaCy aurait fait manquer) — ça ne donne jamais à spaCy
    un pouvoir de rejet supplémentaire, au contraire."""
    return tok.dep_ == "conj" and tok.head.i == verb_tok.i


def _find_particle_span(
    tokens: list,
    verb_pos: int,
    particle_seq: tuple[str, ...],
    *,
    require_particle_pos: bool = True,
):
    """Cherche ``particle_seq`` (ex. ("out",) ou ("up","with")) après
    ``tokens[verb_pos]``, dans une fenêtre de ``MAX_GAP_TOKENS``. Retourne
    les tokens de la particule (contigus entre eux) ou None. S'arrête net
    (frontière de proposition) sur un second verbe/auxiliaire fléchi
    rencontré avant la particule — jamais sur une simple virgule/"and"
    interne à l'objet interposé (voir 'puts the blanket and pan down' dans
    le corpus gold), ni sur un mot coordonné mal étiqueté VERB par spaCy
    (``_is_coordination_noise``, même exemple).

    ``require_particle_pos=False`` : utilisé par
    ``scan_custom_idiom_candidates`` pour un lexique restreint et déjà
    validé manuellement en production (``pipeline.mwe.CUSTOM_IDIOMS`` +
    ``pipeline.custom_lexicon``), où le dernier mot n'est pas forcément une
    particule/préposition (ex. "ass" dans "smart ass", NOUN)."""
    n = len(tokens)
    verb_tok = tokens[verb_pos]
    i = verb_pos + 1
    gap = 0
    while i < n and gap <= MAX_GAP_TOKENS:
        tok = tokens[i]
        if tok.pos_ in ("VERB", "AUX") and not _is_coordination_noise(tok, verb_tok):
            return None  # frontière de proposition : un second verbe fléchi
        particle_ok = not require_particle_pos or tok.pos_ in _PARTICLE_POS
        if tok.lemma_.casefold() == particle_seq[0] and particle_ok:
            matched = [tok]
            ok = True
            j = i + 1
            for extra in particle_seq[1:]:
                if j < n and tokens[j].lemma_.casefold() == extra:
                    matched.append(tokens[j])
                    j += 1
                else:
                    ok = False
                    break
            if ok:
                return matched
        i += 1
        gap += 1
    return None


def scan_phrasal_verb_candidates(doc, segment_idx: int, lexicon: dict[str, tuple[tuple[str, ...], ...]]) -> list[dict[str, Any]]:
    text = doc.text
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for sent in doc.sents:
        tokens = list(sent)
        for vi, vtok in enumerate(tokens):
            if vtok.pos_ not in ("VERB", "AUX"):
                continue
            particle_seqs = lexicon.get(vtok.lemma_.casefold())
            if not particle_seqs:
                continue
            for seq in particle_seqs:
                matched = _find_particle_span(tokens, vi, seq)
                if matched is None:
                    continue
                start_char = vtok.idx
                end_char = matched[-1].idx + len(matched[-1].text)
                if crosses_hard_boundary(text, start_char, end_char):
                    continue
                key = (start_char, end_char)
                if key in seen:
                    continue
                seen.add(key)
                separable = matched[0].i != vtok.i + 1
                member_spans = [
                    {"start_char": vtok.idx, "end_char": vtok.idx + len(vtok.text)},
                    *[
                        {"start_char": t.idx, "end_char": t.idx + len(t.text)}
                        for t in matched
                    ],
                ]
                out.append({
                    "segment_idx": segment_idx,
                    "surface": text[start_char:end_char],
                    "start_char": start_char,
                    "end_char": end_char,
                    "category": "phrasal_verb_separable" if separable else "phrasal_verb_inseparable",
                    "source": "rules_plus_phrasal_verb_scan",
                    "member_spans": member_spans,
                })
    return out


# --------------------------------------------------------------------------
# Lexique custom de production — re-scan à fenêtre large. `pipeline.mwe`
# enregistre "crack open"/"smart ass" (CUSTOM_IDIOMS + data/custom_lexicon.jsonl)
# comme idiomes pour idiomatch, mais le matcher idiomatch tourne à slop=2
# (pipeline/mwe.py::get_matcher, "n=2") : il ne peut jamais relier un idiome
# à travers un objet interposé de 3+ mots ("cracks THE BATHROOM DOOR open").
# Rejoue ce même lexique, déjà validé manuellement en production, avec le
# scanner à fenêtre de ``MAX_GAP_TOKENS`` ci-dessus — jamais un nouveau
# lexique, jamais une réimplémentation d'idiomatch lui-même.
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_custom_idiom_sequences() -> tuple[tuple[str, ...], ...]:
    from pipeline import custom_lexicon
    from pipeline.mwe import CUSTOM_IDIOMS

    entries = list(CUSTOM_IDIOMS) + custom_lexicon.load_idioms()
    out = []
    for entry in entries:
        words = tuple(entry["lemma"].casefold().split())
        if len(words) >= 2:
            out.append(words)
    return tuple(out)


def scan_custom_idiom_candidates(doc, segment_idx: int, sequences: tuple[tuple[str, ...], ...]) -> list[dict[str, Any]]:
    text = doc.text
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for sent in doc.sents:
        tokens = list(sent)
        for vi, tok in enumerate(tokens):
            for seq in sequences:
                if tok.lemma_.casefold() != seq[0]:
                    continue
                matched = _find_particle_span(tokens, vi, seq[1:], require_particle_pos=False)
                if matched is None:
                    continue
                start_char = tok.idx
                end_char = matched[-1].idx + len(matched[-1].text)
                if crosses_hard_boundary(text, start_char, end_char):
                    continue
                key = (start_char, end_char)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "segment_idx": segment_idx,
                    "surface": text[start_char:end_char],
                    "start_char": start_char,
                    "end_char": end_char,
                    "category": "idiom",
                    "source": "rules_plus_custom_idiom_rescan",
                    "member_spans": [
                        {"start_char": tok.idx, "end_char": tok.idx + len(tok.text)},
                        *[{"start_char": t.idx, "end_char": t.idx + len(t.text)} for t in matched],
                    ],
                })
    return out


# --------------------------------------------------------------------------
# WordNet — composés nominaux/adjectivaux multi-mots. N-grammes de LEMMES
# spaCy, tokens NON filtrés (voir docstring) : une ponctuation interposée a
# un lemme qui ne matchera jamais une entrée WordNet, donc un span ne peut
# jamais silencieusement enjamber une virgule/tiret — pas besoin d'un
# filtre séparé pour ce cas.
# --------------------------------------------------------------------------


def scan_wordnet_nominal_candidates(doc, segment_idx: int, lexicon: dict[tuple[str, ...], str]) -> list[dict[str, Any]]:
    text = doc.text
    out: list[dict[str, Any]] = []
    for sent in doc.sents:
        tokens = list(sent)
        n_tokens = len(tokens)
        for n in range(2, 6):
            for i in range(0, n_tokens - n + 1):
                window = tokens[i:i + n]
                key = tuple(t.lemma_.casefold() for t in window)
                pos = lexicon.get(key)
                if pos is None:
                    continue
                start_char = window[0].idx
                end_char = window[-1].idx + len(window[-1].text)
                if crosses_hard_boundary(text, start_char, end_char):
                    continue
                out.append({
                    "segment_idx": segment_idx,
                    "surface": text[start_char:end_char],
                    "start_char": start_char,
                    "end_char": end_char,
                    "category": "nominal_compound" if pos == nwn.NOUN else "idiom",
                    "source": "rules_plus_wordnet_mwe",
                })
    return out


# --------------------------------------------------------------------------
# Règles de bornes — traits d'union et possessif. S'appliquent en deux
# temps : (a) chaîne à trait d'union libre (aucun candidat de base requis —
# couvre "turn-of-the-century"/"smart-ass", jamais proposés par
# multi_token) ; (b) extension à GAUCHE d'un candidat multi_token existant à
# travers un trait d'union (couvre "ground-floor apartment"/"triple-A
# school"/"phys-ed classes", tronqués par multi_token car spaCy tokenise le
# trait d'union en ponctuation séparée — voir phase2_baselines_report.md).
# --------------------------------------------------------------------------

_HYPHEN_CHAIN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+")


def hyphen_chain_candidates(segment_idx: int, text: str) -> list[dict[str, Any]]:
    out = []
    for m in _HYPHEN_CHAIN_RE.finditer(text):
        out.append({
            "segment_idx": segment_idx,
            "surface": m.group(0),
            "start_char": m.start(),
            "end_char": m.end(),
            "category": "multi_token_entity",
            "source": "rules_plus_hyphen_chain",
        })
    return out


def hyphen_extend_existing(multi_token_rows: list[dict], text_by_segment: dict[int, str]) -> list[dict[str, Any]]:
    out = []
    for row in multi_token_rows:
        text = text_by_segment.get(row["segment_idx"])
        if text is None:
            continue
        start = row["start_char"]
        while start > 0 and text[start - 1] == "-":
            j = start - 1
            while j > 0 and text[j - 1].isalnum():
                j -= 1
            if j == start - 1:
                break  # trait d'union sans mot avant : abandon défensif
            start = j
        if start != row["start_char"]:
            out.append({
                "segment_idx": row["segment_idx"],
                "surface": text[start:row["end_char"]],
                "start_char": start,
                "end_char": row["end_char"],
                "category": "nominal_compound",
                "source": "rules_plus_hyphen_extend",
            })
    return out


def possessive_trim_existing(multi_token_rows: list[dict], text_by_segment: dict[int, str]) -> list[dict[str, Any]]:
    out = []
    for row in multi_token_rows:
        surface = row["surface"]
        for suffix in POSSESSIVE_SUFFIXES:
            if surface.endswith(suffix):
                new_end = row["end_char"] - len(suffix)
                text = text_by_segment.get(row["segment_idx"])
                if text is None or new_end <= row["start_char"]:
                    break
                out.append({
                    "segment_idx": row["segment_idx"],
                    "surface": text[row["start_char"]:new_end],
                    "start_char": row["start_char"],
                    "end_char": new_end,
                    "category": "multi_token_entity",
                    "source": "rules_plus_possessive_trim",
                })
                break
    return out


# --------------------------------------------------------------------------
# Assemblage
# --------------------------------------------------------------------------


def build_new_candidates(
    subset_segments,
    docs_by_segment: dict[int, Any],
    multi_token_rows: list[dict],
) -> list[dict[str, Any]]:
    """Tout ce que ``rules_plus`` ajoute EN PLUS de la sortie du pipeline
    actuel (Baseline 2, Phase 2) — l'union des deux se fait dans
    ``phase3_run_rules_plus.py``, jamais ici."""
    text_by_segment = {s.idx: s.en for s in subset_segments}
    pv_lexicon = merged_phrasal_verb_lexicon()
    nominal_lexicon = wordnet_nominal_lexicon()
    custom_idiom_sequences = load_custom_idiom_sequences()

    candidates: list[dict[str, Any]] = []
    for seg in subset_segments:
        doc = docs_by_segment[seg.idx]
        candidates.extend(scan_phrasal_verb_candidates(doc, seg.idx, pv_lexicon))
        candidates.extend(scan_wordnet_nominal_candidates(doc, seg.idx, nominal_lexicon))
        candidates.extend(scan_custom_idiom_candidates(doc, seg.idx, custom_idiom_sequences))
        candidates.extend(hyphen_chain_candidates(seg.idx, seg.en))

    candidates.extend(hyphen_extend_existing(multi_token_rows, text_by_segment))
    candidates.extend(possessive_trim_existing(multi_token_rows, text_by_segment))
    return candidates
