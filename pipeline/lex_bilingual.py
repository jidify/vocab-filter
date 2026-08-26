"""Dictionnaires bilingues anglais-français CONSTRUITS PAR DES HUMAINS —
étape 3 du dispositif d'arbitrage sans relecture humaine (voir le plan
"Valider / corriger suggested_fr et suggested_fr_alt").

Contrairement à omw-fr (WOLF) et WoNeF, construites AUTOMATIQUEMENT et qui
proposent régulièrement des contresens (voir la docstring de
pipeline/sense_fr_frontier.py), les deux ressources ici sont écrites par
des humains — ce sont les seuls signaux du dispositif qui ne proviennent
d'aucun modèle :

- **DBnary** (extrait du Wiktionnaire anglais, http://kaiko.getalp.org —
  `en_dbnary_ontolex.ttl.bz2`, ~200 Mo compressés) : chaque traduction
  française y est rattachée à une GLOSE de sens (comparable à une
  définition WordNet, pas seulement un lemme nu). On aligne cette glose
  sur la définition WordNet du synset visé par cosinus LaBSE
  (sentence-transformers) et on ne retient les traductions que du
  meilleur sens correspondant -> ressource FR humaine ET liée au sens du
  projet, la plus forte de ce module.
- **Apertium fra-eng** (`apertium-fra-eng.fra-eng.dix`, ~1,4 Mo) :
  dictionnaire bilingue écrit à la main, mais au niveau du LEMME SEUL
  (pas de sens — c'est un `.dix` de traduction automatique à règles,
  pas un dictionnaire de sens). Sert de test d'ATTESTATION plus faible
  mais indépendant : "ce mot français a-t-il déjà été utilisé comme
  traduction de ce lemme anglais, dans N'IMPORTE quel sens ?"

Format DBnary — exploré directement sur le flux réel avant d'écrire ce
parseur (voir le fil de travail) : chaque traduction française est un
sujet Turtle AUTONOME nommé `eng:__tr_fra_<n>_<lemme>__<POS>__<sens>`,
portant `dbnary:isTranslationOf`, `dbnary:targetLanguage lexvo:fra`,
`dbnary:writtenForm "..."@fr`, et une référence `dbnary:gloss` vers un
nœud `eng:__en_gloss_<hash>_<lemme>__<POS>__<sens>` dont le `rdf:value`
porte une glose COURTE, de style WordNet, PROPRE À CE SENS (le nœud
gloss est toujours défini juste avant les traductions qui le
référencent, dans l'export réel — ce module s'appuie sur cette
régularité). Ce format est assez RÉGULIER pour un parseur ligne à ligne
dédié — pas de graphe RDF général construit en mémoire (2 Go décompressés
sur la machine de développement) : `extract_dbnary` ne fait qu'UNE passe
streaming sur le flux bz2, et n'applique les expressions régulières
qu'aux lignes des stances `__en_gloss_*` et `__tr_fra_*` — le lemme cible
étant encodé DANS L'IDENTIFIANT du sujet, la quasi-totalité du flux
(traductions vers les ~150 autres langues, lemmes hors du livre) est
ignorée sans même être testée par une regex.

Le fichier DBnary brut n'est PAS committé (voir .gitignore, `pipeline_out/`
est déjà exclu — le cache va là) : `build()` le télécharge une fois dans
ce cache, et n'écrit dans le dépôt qu'un EXTRAIT COMPACT restreint aux
lemmes réellement rencontrés par ce livre (`collect_frontier_targets()`),
`data/bilingual_en_fr.json` — committable, régénérable par --build.

Usage :
    uv run python -m pipeline.lex_bilingual --build
    uv run python -m pipeline.lex_bilingual --build --skip-download   # réutilise le cache déjà présent
"""

from __future__ import annotations

import bz2
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from pipeline import config

DBNARY_URL = "https://kaiko.getalp.org/static/ontolex/latest/en_dbnary_ontolex.ttl.bz2"
APERTIUM_URL = (
    "https://raw.githubusercontent.com/apertium/apertium-fra-eng/master/"
    "apertium-fra-eng.fra-eng.dix"
)

DBNARY_CACHE_PATH = config.CACHE_DIR / "en_dbnary_ontolex.ttl.bz2"
APERTIUM_CACHE_PATH = config.CACHE_DIR / "apertium-fra-eng.dix"

BILINGUAL_EXTRACT_PATH = config.DATA_DIR / "bilingual_en_fr.json"

# dbnary:partOfSpeech (chaîne littérale, ex. "Noun") par code WordNet.
DBNARY_POS_BY_WN = {"n": "Noun", "v": "Verb", "a": "Adjective", "s": "Adjective", "r": "Adverb"}

# Seuil de similarité LaBSE en dessous duquel on refuse de retenir un
# candidat DBnary — PROVISOIRE, à affiner une fois pipeline/eval_sense_fr.py
# disponible (voir le plan). Calibré à la main sur 5 cas connus du magasin
# (mesuré en développant ce module) : van.n.05 vs van__Noun__1 = 0.653
# (vrai match) ; cancer.n.01 vs cancer__Noun__1 = 0.528 (vrai match) vs
# Cancer__Noun__1 "star sign" = 0.132 (faux sens, bonne séparation) ;
# trust.n.01 vs trust__Noun__1 "group of businessmen" = 0.364 (faux sens,
# correctement écarté). 0.55 rejetait le vrai match cancer (0.528) — les
# gloses DBnary sont courtes et télégraphiques, la similarité brute entre
# ce style et une définition WordNet ne grimpe pas aussi haut qu'entre deux
# textes du même style ; 0.45 sépare les 5 cas mesurés sans faux positif.
DBNARY_MATCH_THRESHOLD = 0.45


# ------------------------------------------------------------------
# Découpage des identifiants DBnary (lemme__POS__sens)
# ------------------------------------------------------------------

# Ancré à la FIN de la chaîne : robuste aux lemmes multi-mots contenant
# déjà des underscores ("able_seaman__Noun__1") puisque la catégorie
# grammaticale ET le numéro de sens sont eux-mêmes sans ambiguïté.
_ID_TAIL_RE = re.compile(r"^(.+)__(\w+)__(\d+)$")


def _split_dbnary_id(local_id: str) -> tuple[str, str, str] | None:
    match = _ID_TAIL_RE.match(local_id)
    if not match:
        return None
    lemma, pos_label, sense_n = match.groups()
    return lemma.replace("_", " "), pos_label, sense_n


_GLOSS_PREFIX = "__en_gloss_"


def _gloss_lemma(local_id: str) -> str | None:
    """Lemme d'un identifiant de glose (`__en_gloss_<hash>_<lemme>__<POS>__
    <sens>`) : `_split_dbnary_id` isole correctement POS/sens (ancré en
    fin de chaîne) mais laisse le préfixe `__en_gloss_<hash>_` collé au
    "lemme". Le hash observé (ex. "uSxvaw--") ne contient pas d'underscore
    -> on le retire en coupant sur le PREMIER underscore restant après le
    préfixe connu, ce qui laisse le lemme intact même multi-mots
    ("able_seaman" -> "able seaman")."""
    split = _split_dbnary_id(local_id)
    if split is None:
        return None
    rest = split[0].replace(" ", "_")  # _split_dbnary_id a déjà substitué "_"->" "
    if not rest.startswith(_GLOSS_PREFIX):
        return None
    remainder = rest[len(_GLOSS_PREFIX):]
    _hash, _sep, lemma_part = remainder.partition("_")
    if not lemma_part:
        return None
    return lemma_part.replace("_", " ")


# ------------------------------------------------------------------
# Téléchargement (cache local, jamais committé)
# ------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  téléchargement : {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


# ------------------------------------------------------------------
# DBnary — extraction ciblée en une seule passe streaming
# ------------------------------------------------------------------

_EN_VALUE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"@en\b')
_FR_VALUE_RE = re.compile(r'"((?:[^"\\]|\\.)*)"@fr\b')
_ISTRANSOF_RE = re.compile(r"dbnary:isTranslationOf\s+eng:(\S+?)\s*[;.]")
_TARGETLANG_RE = re.compile(r"dbnary:targetLanguage\s+lexvo:(\w+)")
_GLOSSREF_RE = re.compile(r"dbnary:gloss\s+eng:(\S+?)\s*[;.]")


def extract_dbnary(archive_path: Path, target_lemmas: set[str]) -> dict[str, dict]:
    """Une seule passe sur le flux bz2 décompressé. `target_lemmas` :
    lemmes anglais en casefold (typiquement tous les `lemmas_en` du
    magasin — voir build()). Renvoie
    {"<lemme>__<POS>__<sens>": {"gloss_en": str|None, "fr": [str,...]}},
    restreint aux lemmes fournis."""
    glosses: dict[str, str] = {}
    result: dict[str, dict] = {}

    state: str | None = None  # "gloss" | "tr_fra" | None
    current_id: str | None = None
    tr_fields: dict = {}

    def flush_tr() -> None:
        if tr_fields.get("target_lang") != "fra":
            return
        fr_forms = tr_fields.get("fr")
        translation_of = tr_fields.get("is_translation_of")
        if not fr_forms or not translation_of:
            return
        split = _split_dbnary_id(translation_of)
        if split is None:
            return
        lemma, pos_label, sense_n = split
        if lemma.casefold() not in target_lemmas:
            return
        key = f"{lemma}__{pos_label}__{sense_n}"
        entry = result.setdefault(key, {"gloss_en": None, "fr": []})
        gloss_id = tr_fields.get("gloss_id")
        if gloss_id and gloss_id in glosses:
            entry["gloss_en"] = glosses[gloss_id]
        for form in fr_forms:
            if form not in entry["fr"]:
                entry["fr"].append(form)

    with bz2.open(archive_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue

            if line[0] not in (" ", "\t") and line.startswith("eng:__"):
                # Nouvelle ligne de sujet (peut porter des prédicats sur
                # cette même ligne, cf. les stances "Page" compactes) ->
                # clôt le sujet `tr_fra` précédent s'il n'a pas déjà été
                # terminé par un "." (filet de sécurité).
                if state == "tr_fra":
                    flush_tr()
                bare_subject = line.split(None, 1)[0]
                local_id = bare_subject[len("eng:"):]
                if local_id.startswith("__en_gloss_"):
                    lemma = _gloss_lemma(local_id)
                    if lemma is not None and lemma.casefold() in target_lemmas:
                        state, current_id = "gloss", local_id
                    else:
                        state, current_id = None, None
                elif local_id.startswith("__tr_fra_"):
                    state, current_id, tr_fields = "tr_fra", local_id, {"target_lang": "fra"}
                else:
                    state, current_id = None, None
                # ne PAS `continue` : un prédicat peut suivre sur cette
                # même ligne physique (stances compactes à une ligne).

            if state == "gloss":
                if "rdf:value" in line:
                    match = _EN_VALUE_RE.search(line)
                    if match:
                        glosses[current_id] = match.group(1)
            elif state == "tr_fra":
                match = _ISTRANSOF_RE.search(line)
                if match:
                    tr_fields["is_translation_of"] = match.group(1)
                match = _TARGETLANG_RE.search(line)
                if match:
                    tr_fields["target_lang"] = match.group(1)
                match = _GLOSSREF_RE.search(line)
                if match:
                    tr_fields["gloss_id"] = match.group(1)
                for match in _FR_VALUE_RE.finditer(line):
                    tr_fields.setdefault("fr", []).append(match.group(1))
                if line.rstrip().endswith("."):
                    flush_tr()
                    state, current_id, tr_fields = None, None, {}

    return result


# ------------------------------------------------------------------
# Apertium fra-eng (.dix) — dictionnaire bilingue à plat, niveau lemme
# ------------------------------------------------------------------


def _dix_text(el: ET.Element) -> str:
    """Concatène le texte d'un élément <l>/<r> Apertium : <b/> (frontière
    de mot dans une entrée multi-mots) -> espace, <s .../> (marqueurs de
    catégorie grammaticale, sans texte propre) simplement traversé."""
    parts: list[str] = [el.text or ""]
    for child in el:
        if child.tag == "b":
            parts.append(" ")
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def extract_apertium(dix_path: Path) -> dict[str, list[str]]:
    """{lemme_anglais_casefold: [mots_français,...]} — sens non distingué
    (dictionnaire à plat), voir la docstring du module."""
    tree = ET.parse(dix_path)
    result: dict[str, list[str]] = {}
    for e in tree.getroot().iter("e"):
        p = e.find("p")
        if p is None:
            continue
        l_el, r_el = p.find("l"), p.find("r")
        if l_el is None or r_el is None:
            continue
        fr_text, en_text = _dix_text(l_el), _dix_text(r_el)
        if not fr_text or not en_text:
            continue
        bucket = result.setdefault(en_text.casefold(), [])
        if fr_text not in bucket:
            bucket.append(fr_text)
    return result


# ------------------------------------------------------------------
# Construction de l'extrait committable (data/bilingual_en_fr.json)
# ------------------------------------------------------------------


def build(book_lemmas: set[str] | None = None, skip_download: bool = False) -> dict:
    if book_lemmas is None:
        from pipeline.sense_fr_frontier import collect_frontier_targets
        resolved, _unresolved = collect_frontier_targets()
        book_lemmas = {lemma.casefold() for t in resolved for lemma in t["lemmas_en"]}
    print(f"{len(book_lemmas)} lemme(s) cible(s) (vocabulaire du livre).")

    if not skip_download:
        _download(APERTIUM_URL, APERTIUM_CACHE_PATH)
    apertium = extract_apertium(APERTIUM_CACHE_PATH)
    apertium_filtered = {k: v for k, v in apertium.items() if k in book_lemmas}
    print(f"Apertium : {len(apertium)} lemme(s) au total, {len(apertium_filtered)} retenu(s).")

    if not skip_download:
        _download(DBNARY_URL, DBNARY_CACHE_PATH)
    print("Extraction DBnary (une passe sur le flux bz2, peut prendre quelques minutes)...")
    dbnary = extract_dbnary(DBNARY_CACHE_PATH, book_lemmas)
    print(f"DBnary : {len(dbnary)} entrée(s) lemme+POS+sens retenue(s).")

    extract = {"dbnary": dbnary, "apertium": apertium_filtered}
    config.ensure_data_dir()
    BILINGUAL_EXTRACT_PATH.write_text(
        json.dumps(extract, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"Extrait écrit -> {BILINGUAL_EXTRACT_PATH}")
    return extract


# ------------------------------------------------------------------
# API de lecture (utilisée par pipeline/sense_fr_adjudicate.py)
# ------------------------------------------------------------------

_extract_cache: dict | None = None


def load_extract() -> dict:
    global _extract_cache
    if _extract_cache is None:
        if BILINGUAL_EXTRACT_PATH.exists():
            _extract_cache = json.loads(BILINGUAL_EXTRACT_PATH.read_text(encoding="utf-8"))
        else:
            _extract_cache = {"dbnary": {}, "apertium": {}}
    return _extract_cache


def dbnary_senses_for_lemma(lemma: str, wn_pos: str | None) -> list[tuple[str, dict]]:
    """Toutes les entrées DBnary (clé, {"gloss_en","fr"}) dont le lemme
    (casefold) — et la catégorie grammaticale si `wn_pos` est connu —
    correspondent."""
    extract = load_extract()
    pos_label = DBNARY_POS_BY_WN.get(wn_pos) if wn_pos else None
    out = []
    for key, entry in extract["dbnary"].items():
        split = _split_dbnary_id(key)
        if split is None:
            continue
        entry_lemma, entry_pos, _sense_n = split
        if entry_lemma.casefold() != lemma.casefold():
            continue
        if pos_label and entry_pos != pos_label:
            continue
        out.append((key, entry))
    return out


_labse_model = None


def get_labse_model():
    """Modèle LaBSE partagé (chargé une seule fois par processus) —
    exposé publiquement : pipeline/sense_fr_adjudicate.py le réutilise
    pour son propre test de discrimination (candidat FR vs définitions
    des synsets frères), pas seulement pour l'alignement DBnary."""
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    return _labse_model


def best_dbnary_match(
    definition_en: str, lemma: str, wn_pos: str | None, threshold: float = DBNARY_MATCH_THRESHOLD
) -> tuple[list[str], float] | None:
    """Meilleure entrée DBnary pour ce lemme (+POS si connu), par
    similarité cosinus LaBSE entre la définition WordNet du synset visé
    et la glose DBnary de chaque sens candidat. Renvoie (candidats_fr,
    score) si le meilleur score dépasse `threshold`, sinon None.

    IMPORTANT : la comparaison a lieu même avec un SEUL candidat DBnary —
    un seul sens catalogué pour ce lemme+POS ne veut pas dire que c'est
    LE sens visé, seulement que Wiktionnaire n'en catalogue qu'un
    (mesuré sur ce livre : "trust" n'a qu'une entrée DBnary "Noun",
    glosée "a group of businessmen or traders" — le cartel, pas le sens
    fiduciaire cherché par trust.n.01 dans ce livre ; accepter ce
    candidat sans comparaison aurait promu une corroboration entre deux
    sens simplement différents)."""
    candidates = dbnary_senses_for_lemma(lemma, wn_pos)
    candidates = [(k, e) for k, e in candidates if e.get("gloss_en") and e.get("fr")]
    if not candidates:
        return None

    model = get_labse_model()
    texts = [definition_en] + [entry["gloss_en"] for _key, entry in candidates]
    embeddings = model.encode(texts, normalize_embeddings=True)
    target = embeddings[0]
    sims = embeddings[1:] @ target
    best_i = int(sims.argmax())
    if float(sims[best_i]) < threshold:
        return None
    return candidates[best_i][1]["fr"], float(sims[best_i])


def apertium_attests(lemma_en: str, fr_candidate: str) -> bool:
    """Test d'ATTESTATION (pas de sens) : le candidat français partage-t-il
    un mot de contenu avec une traduction Apertium connue de ce lemme,
    dans N'IMPORTE quel sens ? Signal plus faible que DBnary, jamais
    suffisant seul (voir pipeline/sense_fr_adjudicate.py)."""
    from pipeline import fr_norm

    known = load_extract()["apertium"].get(lemma_en.casefold(), [])
    return fr_norm.any_match([fr_candidate], known)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="(Re)construit data/bilingual_en_fr.json.")
    parser.add_argument("--skip-download", action="store_true",
                         help="Réutilise les fichiers déjà présents dans pipeline_out/cache/ sans retélécharger.")
    args = parser.parse_args()
    if args.build:
        build(skip_download=args.skip_download)
    else:
        print("Rien à faire sans --build (voir le module pour l'API de lecture).")
