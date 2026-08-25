"""Chargement des ressources lexicales partagées : Word Prevalence,
CEFR (par POS — voir word_senses.py:53-94, contrairement à
prevalence_test.py qui ignore le POS), et AoA (Kuperman).

Un seul chargement par processus (caches module-level), réutilisé par
select.py (S4) et score.py (S6).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

from pipeline import config

_prevalence_cache: dict[str, "PrevalenceRow"] | None = None
_cefr_cache: dict[str, dict[str, set[str]]] | None = None
_aoa_cache: dict[str, float] | None = None


@dataclass
class PrevalenceRow:
    word: str
    pknown: float
    nobs: int
    prevalence: float
    zipf: float


def load_prevalence() -> dict[str, PrevalenceRow]:
    """word-prevalence.txt n'a pas d'en-tête : word,Pknown,Nobs,Prevalence,FreqZipfUS
    (repris de prevalence_test.py:166-198)."""

    global _prevalence_cache
    if _prevalence_cache is not None:
        return _prevalence_cache

    rows: dict[str, PrevalenceRow] = {}
    if not config.PREVALENCE_PATH.exists():
        _prevalence_cache = rows
        return rows

    with config.PREVALENCE_PATH.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 5:
                continue
            word = parts[0].strip().casefold()
            try:
                pknown = float(parts[1])
                nobs = int(float(parts[2]))
                prevalence = float(parts[3])
                zipf = float(parts[4])
            except ValueError:
                continue
            rows[word] = PrevalenceRow(word, pknown, nobs, prevalence, zipf)

    _prevalence_cache = rows
    return rows


def load_cefr_by_word_pos() -> dict[str, dict[str, set[str]]]:
    """cefr.csv, jointure PAR POS — reprend word_senses.py:53-94
    (headword;pos;CEFR;..., délimiteur ';', utf-8-sig).
    Retourne {headword: {pos_name: {levels}}}, pos_name in
    {noun, verb, adjective, adverb, ...} (tel quel dans le fichier)."""

    global _cefr_cache
    if _cefr_cache is not None:
        return _cefr_cache

    cefr_by_word: dict[str, dict[str, set[str]]] = {}
    if not config.CEFR_PATH.exists():
        _cefr_cache = cefr_by_word
        return cefr_by_word

    with config.CEFR_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            headword = row["headword"].strip().casefold()
            pos = row["pos"].strip().casefold()
            cefr = row["CEFR"].strip().upper()
            if not headword:
                continue
            cefr_by_word.setdefault(headword, {}).setdefault(pos, set()).add(cefr)

    _cefr_cache = cefr_by_word
    return cefr_by_word


def cefr_levels_for(word: str, wn_pos: str | None) -> set[str]:
    """Niveaux CEFR connus pour ce mot, restreints au POS si fourni
    (mappé via config.POS_TO_WN inversé). Si le POS ne matche aucune
    entrée mais que le mot existe sous un autre POS, on retombe sur
    l'union de tous les POS connus plutôt que de perdre l'info."""

    cefr_by_word = load_cefr_by_word_pos()
    pos_data = cefr_by_word.get(word.casefold())
    if not pos_data:
        return set()

    if wn_pos:
        wn_to_cefr_pos = {v: k for k, v in config.POS_TO_WN.items()}
        cefr_pos_name = wn_to_cefr_pos.get(wn_pos)
        if cefr_pos_name and cefr_pos_name in pos_data:
            return pos_data[cefr_pos_name]

    union: set[str] = set()
    for levels in pos_data.values():
        union |= levels
    return union


def should_exclude_cefr(levels: set[str]) -> bool:
    """Exclu seulement si TOUS les niveaux connus sont A1/A2 (reprend
    prevalence_test.py:142) ; niveau inconnu ('?', vide) => conservé."""

    known = levels & set(config.CEFR_ORDER)
    if not known:
        return False
    return known.issubset(config.EXCLUDED_CEFR)


_AOA_FORM_RE = re.compile(r".*(s|ed|ing|es)$")


def load_aoa() -> dict[str, float]:
    """kuperman-aoa.csv : Word, ..., Rating.Mean, ... (repris de
    hello-aoa.py:11-13). Clé = forme exacte casefold, pas de
    lemmatisation ici — voir aoa_for_form qui applique le correctif
    §1.3 (chercher la forme exacte d'abord, ne lemmatiser que si le
    mot ressemble à une forme fléchie, pour éviter opera -> opus)."""

    global _aoa_cache
    if _aoa_cache is not None:
        return _aoa_cache

    aoa: dict[str, float] = {}
    if not config.AOA_PATH.exists():
        _aoa_cache = aoa
        return aoa

    with config.AOA_PATH.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("Word", "").strip().casefold()
            rating = row.get("Rating.Mean", "").strip()
            if not word or not rating:
                continue
            try:
                aoa[word] = float(rating)
            except ValueError:
                continue

    _aoa_cache = aoa
    return aoa


def aoa_for_form(surface: str, lemma: str) -> float | None:
    """Forme exacte d'abord ; lemme seulement si la surface ressemble
    à une flexion (-s/-ed/-ing/-es) et n'est pas elle-même dans le
    fichier — correctif du bug opera->opus documenté en §1.3."""

    aoa = load_aoa()
    key = surface.casefold()
    if key in aoa:
        return aoa[key]

    if lemma and lemma.casefold() != key and _AOA_FORM_RE.match(key):
        return aoa.get(lemma.casefold())

    return None
