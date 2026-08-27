"""S5 — Identifier le sens réellement utilisé, occurrence par occurrence.

Extraction et adaptation de `sense_in_context.py` (14/14 sur le harnais
manuel — voir vocab-filter-resume.md §3.6). Les fonctions de score sont
reprises À L'IDENTIQUE (mêmes constantes FR_BASE_OMW / FR_BASE_WONEF /
FR_CLAIM_DISCOUNT, même formule `informativité × facteur_réclamé`) ;
seuls trois points changent, comme prévu par le plan :

1. Le contexte élargi est construit à partir de l'INDEX DE SEGMENT
   (on sait déjà exactement où est la phrase, puisque c'est nous qui
   avons segmenté le livre en S0) au lieu de la recherche par flux de
   tokens de `build_wide_context` — supprime la limitation §9.2
   "dépendance à la mise en forme du texte source".
2. La phrase française est lue depuis le segment (livre bilingue),
   pas écrite à la main dans `TESTS` — supprime la limitation §9.2
   "traduction française absente du texte source". Si aucun bilingue
   n'est chargé, `fr=None` et le score de preuve française est 0 :
   GlossBERT décide seul (perte documentée sur `view`/`butt`).
3. GlossBERT est appelé une fois par occurrence mais le modèle est
   chargé une seule fois par processus (paresseux) au lieu d'un import
   top-level qui bloquerait tout le reste du pipeline même quand la
   désambiguïsation n'est pas nécessaire.

L'arbitrage LLM (proposition_1 §5.3) n'intervient que si la marge
GlossBERT+FR entre le 1er et le 2e synset est sous MARGIN_THRESHOLD,
ou si aucun synset n'a de score positif, ou si le meilleur synset
GlossBERT et le meilleur synset "preuve FR" diffèrent (désaccord).
"""

from __future__ import annotations

import json
import re
import unicodedata

import wn
from nltk.corpus import wordnet as nwn
from lemminflect import getAllInflections, getAllInflectionsOOV

from pipeline import atomic, config, llm
from pipeline.corpus import Segment, load_segments

MARGIN_THRESHOLD = 0.15

_gloss_model = None
_EN = None
_wonef_cache: dict | None = None


GLOSSBERT_MAX_SEQ_LENGTH = 192


def get_gloss_model():
    """`glossbert.GlossBERT._convert_to_features` pad TOUJOURS à 512
    tokens quel que soit le contenu réel (glossbert.py:129, appelé
    sans argument dans __call__:193) — sur CPU, l'auto-attention BERT
    étant en O(L²), ceci rendait le batch S5 ~7x plus lent que
    nécessaire (mesuré : ETA ~235 min sur 2922 occurrences). Nos
    contextes élargis font ~7 mots/segment × 5 segments en médiane —
    192 tokens est une marge large, avec troncature sûre au-delà
    (_truncate_seq_pair) plutôt qu'une erreur."""

    global _gloss_model
    if _gloss_model is None:
        from glossbert import GlossBERT
        print("Chargement de GlossBERT...")
        _gloss_model = GlossBERT()
        original = _gloss_model._convert_to_features
        _gloss_model._convert_to_features = (
            lambda candidates, max_seq_length=GLOSSBERT_MAX_SEQ_LENGTH:
            original(candidates, max_seq_length=max_seq_length)
        )
        print("GlossBERT chargé.")
    return _gloss_model


def get_en_wordnet():
    global _EN
    if _EN is None:
        _EN = wn.Wordnet(config.EN_LEXICON)
    return _EN


# ============================================================
# WordNet (NLTK) — sélection des synsets candidats
# (identique à sense_in_context.py:224-269)
# ============================================================

def normalize_pos(pos):
    if pos == "a":
        return {"a", "s"}
    return {pos}


def get_synsets(word, pos):
    wanted_pos = normalize_pos(pos)
    results = []
    for synset in nwn.synsets(word):
        if synset.pos() not in wanted_pos:
            continue
        for lemma in synset.lemmas():
            if lemma.name().replace("_", " ").casefold() == word.casefold():
                results.append(synset)
                break
    return results


def get_synonyms(synset):
    return [lemma.name().replace("_", " ") for lemma in synset.lemmas()]


def synset_offset(synset):
    return f"{synset.offset():08d}"


# ============================================================
# Localisation du mot cible (identique à :297-370)
# ============================================================

def candidate_surface_forms(word, pos):
    upos = config.POS_TO_UPOS.get(pos)
    forms = {word}
    if upos:
        inflections = getAllInflections(word, upos=upos) or getAllInflectionsOOV(word, upos=upos)
        for tags in inflections.values():
            forms.update(tags)
    return sorted(forms, key=len, reverse=True)


def locate_target_word(word, pos, text, restrict=None):
    window_start, window_end = restrict if restrict is not None else (0, len(text))
    search_zone = text[window_start:window_end].casefold()

    for form in candidate_surface_forms(word, pos):
        pattern = r"\b" + re.escape(form.casefold()) + r"\b"
        match = re.search(pattern, search_zone)
        if match:
            start = window_start + match.start()
            end = window_start + match.end()
            return start, end, text[start:end]
    return None


# ============================================================
# Contexte élargi — VERSION S5 : par index de segment, plus fiable
# que la recherche par flux de tokens de build_wide_context.
# ============================================================

_segment_index_cache: dict[int, tuple[dict, list]] = {}


def _segment_lookup(segments: list[Segment]):
    """Construit (une seule fois par liste de segments, mise en cache
    par id() de la liste) l'index {idx: Segment} et la liste triée des
    idx — reconstruire ceci à chaque occurrence (3000+ fois sur ce
    livre) était le principal goulot d'étranglement du batch S5."""

    key = id(segments)
    cached = _segment_index_cache.get(key)
    if cached is not None:
        return cached
    by_idx = {s.idx: s for s in segments}
    all_idxs = sorted(by_idx)
    _segment_index_cache[key] = (by_idx, all_idxs)
    return by_idx, all_idxs


def build_wide_context_from_segments(
    segments: list[Segment], seg_idx: int, window: int = config.CONTEXT_WINDOW
) -> dict:
    """Équivalent de build_wide_context, mais on connaît déjà l'index
    du segment (c'est nous qui avons segmenté le livre) : pas de
    recherche floue nécessaire."""

    by_idx, all_idxs = _segment_lookup(segments)
    pos_in_list = all_idxs.index(seg_idx)

    window_idxs = all_idxs[max(0, pos_in_list - window): pos_in_list + window + 1]

    context_text = ""
    sentence_start = sentence_end = None
    for idx in window_idxs:
        if context_text:
            context_text += " "
        char_start = len(context_text)
        context_text += by_idx[idx].en
        char_end = len(context_text)
        if idx == seg_idx:
            sentence_start, sentence_end = char_start, char_end

    return {"text": context_text, "sentence_start": sentence_start, "sentence_end": sentence_end}


# ============================================================
# Preuve française (identique à sense_in_context.py:570-1148)
# ============================================================

def strip_accents(text):
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def fr_stem(word):
    normalized = strip_accents(word.casefold()).strip()
    if len(normalized) > 3 and normalized[-1] in ("s", "x"):
        normalized = normalized[:-1]
    return normalized


def fr_tokens(text):
    cleaned = text.replace("’", "'")
    return [t for t in re.split(r"[^a-zA-Zà-ÿ]+", cleaned) if len(t) > 2]


def normalize_fr_phrase(text):
    cleaned = text.replace("’", "'").replace("‘", "'")
    cleaned = strip_accents(cleaned.casefold())
    cleaned = re.sub(r"[-]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9' ]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def stems_of(text):
    return {fr_stem(w) for w in text.split() if len(w) > 2}


_fr_lemmas_for_synset_cache: dict[tuple, list[str]] = {}


def fr_lemmas_for_synset(word, synset_id_offset, pos):
    cache_key = (word.casefold(), synset_id_offset, pos)
    if cache_key in _fr_lemmas_for_synset_cache:
        return _fr_lemmas_for_synset_cache[cache_key]

    translations = []
    EN = get_en_wordnet()
    for wn_synset in EN.synsets(word, pos=pos):
        offset = wn_synset.id.split("-")[-2]
        if offset != synset_id_offset:
            continue
        for sense in wn_synset.senses():
            try:
                sense_lemma = sense.word().lemma().replace("_", " ").casefold()
            except Exception:
                continue
            if sense_lemma != word.casefold():
                continue
            try:
                french_senses = sense.translate(lexicon=config.FR_LEXICON)
            except Exception:
                continue
            for french_sense in french_senses:
                try:
                    lemma = french_sense.word().lemma().replace("_", " ")
                except Exception:
                    continue
                if lemma and lemma not in translations:
                    translations.append(lemma)

    _fr_lemmas_for_synset_cache[cache_key] = translations
    return translations


_all_fr_lemmas_cache: dict[str, list[str]] = {}


def all_fr_lemmas_for_word(word):
    """Mémoïsé : appelé pour chaque mot de contenu de chaque phrase
    (via claimed_fr_stems), et le vocabulaire courant ("have", "look",
    "get"...) revient des centaines de fois dans le livre — sans cache,
    c'était le principal goulot d'étranglement du batch S5 une fois la
    preuve française activée (mesuré : 1.0/s -> 0.18/s)."""

    key = word.casefold()
    if key in _all_fr_lemmas_cache:
        return _all_fr_lemmas_cache[key]

    translations = []
    EN = get_en_wordnet()
    for wn_synset in EN.synsets(word):
        for sense in wn_synset.senses():
            try:
                french_senses = sense.translate(lexicon=config.FR_LEXICON)
            except Exception:
                continue
            for french_sense in french_senses:
                try:
                    lemma = french_sense.word().lemma().replace("_", " ")
                except Exception:
                    continue
                if lemma:
                    translations.append(lemma)

    _all_fr_lemmas_cache[key] = translations
    return translations


def content_words(english, exclude):
    exclude_key = exclude.casefold()
    words = set()
    for token in re.findall(r"[A-Za-z']+", english):
        if len(token) <= 2 or token.casefold() == exclude_key:
            continue
        words.add(token.casefold())
    return words


def claimed_fr_stems(word, english):
    claimed = {fr_stem(word)}
    for other_word in content_words(english, word):
        for lemma in all_fr_lemmas_for_word(other_word):
            claimed.add(fr_stem(lemma))
    return claimed


def load_wonef_precision():
    global _wonef_cache
    if _wonef_cache is not None:
        return _wonef_cache

    import xml.etree.ElementTree as ET

    wonef_by_id = {}
    try:
        tree = ET.parse(config.WONEF_PRECISION_PATH)
    except (FileNotFoundError, ET.ParseError):
        _wonef_cache = {}
        return _wonef_cache

    for synset_element in tree.getroot().iter():
        tag = synset_element.tag.split("}")[-1].upper()
        if tag != "SYNSET":
            continue
        synset_id = None
        literals = []
        for child in synset_element:
            child_tag = child.tag.split("}")[-1].upper()
            if child_tag == "ID" and child.text:
                synset_id = child.text.strip()
            elif child_tag == "SYNONYM":
                for literal_element in child.iter():
                    if literal_element.tag.split("}")[-1].upper() != "LITERAL":
                        continue
                    if not literal_element.text:
                        continue
                    literal = literal_element.text.strip()
                    if literal and literal != "_EMPTY_":
                        literals.append(literal)
        if synset_id:
            wonef_by_id[synset_id] = literals

    _wonef_cache = wonef_by_id
    return _wonef_cache


def wonef_lemmas_for_synset(word, offset, pos):
    wonef_by_id = load_wonef_precision()
    literals = wonef_by_id.get(f"eng-30-{offset}-{pos}", [])
    word_stem = fr_stem(word)
    return [l for l in literals if fr_stem(l) != word_stem]


def fr_lemma_match_key(lemma, fr_word_stems, normalized_french):
    if " " in lemma or "-" in lemma:
        normalized_lemma = normalize_fr_phrase(lemma)
        return normalized_lemma if normalized_lemma and normalized_lemma in normalized_french else None
    stem = fr_stem(lemma)
    return stem if stem in fr_word_stems else None


def compute_fr_scores(word, pos, synsets, french, claimed):
    fr_word_stems = {fr_stem(t) for t in fr_tokens(french)}
    normalized_french = normalize_fr_phrase(french)

    per_synset = {}
    for synset in synsets:
        offset = synset_offset(synset)
        omw_lemmas = fr_lemmas_for_synset(word, offset, pos)
        omw_hit_keys = {
            key for lemma in omw_lemmas
            if (key := fr_lemma_match_key(lemma, fr_word_stems, normalized_french)) is not None
        }
        if omw_hit_keys:
            per_synset[synset.name()] = {"source": "omw-fr", "lemmas": omw_lemmas, "hit_keys": omw_hit_keys}
            continue

        wonef_lemmas = wonef_lemmas_for_synset(word, offset, pos)
        wonef_hit_keys = {
            key for lemma in wonef_lemmas
            if (key := fr_lemma_match_key(lemma, fr_word_stems, normalized_french)) is not None
        }
        per_synset[synset.name()] = {
            "source": "wonef" if wonef_hit_keys else None,
            "lemmas": wonef_lemmas if wonef_hit_keys else omw_lemmas,
            "hit_keys": wonef_hit_keys,
        }

    key_counts = {"omw-fr": {}, "wonef": {}}
    for entry in per_synset.values():
        source = entry["source"]
        if source is None:
            continue
        for key in entry["hit_keys"]:
            key_counts[source][key] = key_counts[source].get(key, 0) + 1

    base_weight = {"omw-fr": config.FR_BASE_OMW, "wonef": config.FR_BASE_WONEF}

    results = {}
    for name, entry in per_synset.items():
        source, hit_keys = entry["source"], entry["hit_keys"]
        if source is None or not hit_keys:
            results[name] = {"score": 0.0, "source": None, "lemmas": entry["lemmas"],
                              "hits": [], "best_key": None, "best_count": None}
            continue

        best_score, best_key = 0.0, None
        for key in hit_keys:
            count = key_counts[source][key]
            informativeness = 1.0 / count
            claim_factor = config.FR_CLAIM_DISCOUNT if stems_of(key) & claimed else 1.0
            score = base_weight[source] * informativeness * claim_factor
            if score > best_score or best_key is None:
                best_score, best_key = score, key

        hits = [
            lemma for lemma in entry["lemmas"]
            if fr_lemma_match_key(lemma, fr_word_stems, normalized_french) in hit_keys
        ]
        results[name] = {
            "score": best_score, "source": source, "lemmas": entry["lemmas"],
            "hits": hits, "best_key": best_key, "best_count": key_counts[source][best_key],
        }
    return results


# ============================================================
# GlossBERT
# ============================================================

def glossbert_scores(word, pos, context_text, restrict, synsets):
    located = locate_target_word(word, pos, context_text, restrict)
    if located is None:
        return {}, None
    start, end, surface = located

    raw_results = get_gloss_model()(context_text, start, end, word)
    wanted_names = {s.name() for s in synsets}
    scores = {s.name(): float(sc) for sc, s in raw_results if s.name() in wanted_names}
    return scores, surface


# ============================================================
# Arbitrage LLM (proposition_1 §5.3)
# ============================================================

ARBITRATION_SYSTEM = (
    "Tu es lexicographe. On te donne un mot anglais, une phrase où il "
    "apparaît, et un inventaire fermé de sens WordNet candidats. Choisis "
    "le sens réellement employé dans cette phrase, ou indique qu'aucun ne "
    "convient. Cite l'indice textuel qui justifie ton choix."
)

ARBITRATION_TEMPLATE = """Mot cible : "{word}" (POS={pos})
Contexte : {context}

Sens candidats :
{candidates}

Réponds en JSON strict :
{{"selected_sense": "<id du sens choisi ou 'aucun_sens_adapte'>", "usage_type": "<litteral|idiomatique|autre>", "contextual_meaning_fr": "<courte paraphrase française>", "evidence": "<indice textuel>", "confidence": <0.0-1.0>}}
"""


def arbitrate(word, pos, context_text, synsets):
    candidates = "\n".join(
        f'- {s.name()}: {s.definition()}' for s in synsets
    )
    prompt = ARBITRATION_TEMPLATE.format(
        word=word, pos=pos, context=context_text, candidates=candidates
    )
    try:
        result = llm.call_json(prompt, system=ARBITRATION_SYSTEM, timeout=120)
    except llm.LLMError as exc:
        return {"selected_sense": None, "confidence": 0.0, "error": str(exc)}
    return result


# ============================================================
# Analyse d'une occurrence
# ============================================================

def analyze_occurrence(
    word: str, pos: str, segments: list[Segment], seg_idx: int, allow_arbitration: bool = True
) -> dict | None:
    synsets = get_synsets(word, pos)
    if not synsets:
        return None

    wide = build_wide_context_from_segments(segments, seg_idx)
    context_text = wide["text"]
    restrict = (wide["sentence_start"], wide["sentence_end"])

    if len(synsets) == 1:
        # Rien à désambiguïser : appeler GlossBERT ici ne changerait
        # jamais le résultat (un seul candidat gagne toujours), donc
        # coûterait un forward BERT pour rien. ~21% des occurrences du
        # livre sont dans ce cas (mesuré sur The Humans).
        located = locate_target_word(word, pos, context_text, restrict)
        surface = located[2] if located else None
        gloss_scores: dict[str, float] = {}
    else:
        gloss_scores, surface = glossbert_scores(word, pos, context_text, restrict, synsets)

    by_idx, _ = _segment_lookup(segments)
    french = by_idx[seg_idx].fr or ""
    if french:
        # claimed_fr_stems interroge omw-fr pour CHAQUE mot de contenu
        # de la phrase (all_fr_lemmas_for_word) : coûteux, donc calculé
        # seulement quand une preuve française existe réellement à
        # pondérer (sans bilingue, fr_score est de toute façon 0).
        claimed = claimed_fr_stems(word, by_idx[seg_idx].en)
        fr_scores = compute_fr_scores(word, pos, synsets, french, claimed)
    else:
        fr_scores = {
            s.name(): {"score": 0.0, "source": None, "lemmas": [], "hits": [],
                        "best_key": None, "best_count": None}
            for s in synsets
        }

    results = []
    for synset in synsets:
        fr_entry = fr_scores[synset.name()]
        gloss_score = gloss_scores.get(synset.name(), 0.0)
        results.append({
            "synset": synset.name(),
            "definition": synset.definition(),
            "synonyms": get_synonyms(synset),
            "gloss_score": gloss_score,
            "fr_score": fr_entry["score"],
            "fr_source": fr_entry["source"],
            "fr_hits": fr_entry["hits"],
            "final_score": gloss_score + fr_entry["score"],
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)

    margin = (results[0]["final_score"] - results[1]["final_score"]) if len(results) >= 2 else 1.0
    all_zero = all(r["final_score"] == 0.0 for r in results)

    gloss_best = max(results, key=lambda r: r["gloss_score"])["synset"] if any(r["gloss_score"] for r in results) else None
    fr_best = max(results, key=lambda r: r["fr_score"])["synset"] if any(r["fr_score"] for r in results) else None
    disagreement = bool(gloss_best and fr_best and gloss_best != fr_best)

    needs_arbitration = margin < MARGIN_THRESHOLD or all_zero or disagreement

    record = {
        "word": word, "pos": pos, "segment_idx": seg_idx,
        "target_surface": surface, "context": context_text,
        "french": french or None,
        "candidates": results,
        "best_sense": results[0]["synset"],
        "margin": margin,
        "needs_review": False,
        "arbitration": None,
    }

    if needs_arbitration and allow_arbitration:
        arb = arbitrate(word, pos, context_text, synsets)
        record["arbitration"] = arb
        selected = arb.get("selected_sense")
        if selected and selected != "aucun_sens_adapte" and any(r["synset"] == selected for r in results):
            record["best_sense"] = selected
            record["needs_review"] = float(arb.get("confidence", 0)) < 0.6
        elif selected == "aucun_sens_adapte":
            record["best_sense"] = "aucun_sens_adapte"
            record["needs_review"] = True
        else:
            record["needs_review"] = True

    return record


def coarse_priority(t: dict) -> float:
    """Score grossier AVANT désambiguïsation, pour ordonner le batch S5
    du plus prometteur au moins prometteur — uniquement à partir de ce
    que S4 sait déjà (Zipf, pas de sens). Sert à deux choses : (1) un
    run interrompu laisse quand même les meilleurs candidats déjà
    traités ; (2) SENSE_TOP_K (si fixé) peut couper la queue sans jeter
    au hasard les mots qui avaient une vraie chance de finir dans le
    classement final — répond à l'objection : pourquoi désambiguïser
    des mots qu'on va de toute façon écarter au tri ?"""

    zipf = t.get("zipf")
    zipf_component = (6.0 - zipf) / 4.0 if zipf is not None else 0.5
    # Léger bonus de fréquence dans le livre : un mot vu 5 fois vaut
    # d'être traité en priorité (il pèsera plus dans le classement).
    return zipf_component + 0.05 * min(t.get("book_count", 1), 5)


def most_frequent_sense_fallback(word: str, wn_pos: str, synsets) -> dict:
    """Repli bon marché pour la queue au-delà de SENSE_TOP_K : le sens
    WordNet avec le plus grand lemma.count() SemCor (sens dominant
    déclaré, pas mesuré sur CE livre) — sans GlossBERT ni LLM. Marqué
    needs_review pour rester visible et corrigeable, jamais silencieux."""

    def semcor_count(synset):
        for l in synset.lemmas():
            if l.name().replace("_", " ").casefold() == word.casefold():
                return l.count()
        return 0

    best = max(synsets, key=semcor_count)
    return {
        "word": word, "pos": wn_pos, "best_sense": best.name(),
        "candidates": [{
            "synset": best.name(), "definition": best.definition(),
            "synonyms": get_synonyms(best), "gloss_score": 0.0,
            "fr_score": 0.0, "fr_source": None, "fr_hits": [], "final_score": 0.0,
        }],
        "target_surface": None, "context": None, "french": None,
        "margin": 1.0, "needs_review": True, "arbitration": None,
        "fallback": "most_frequent_sense_no_gloss",
    }


# ============================================================
# Relecture de pipeline_out/senses.jsonl (+ selected_mwe.jsonl pour les
# clés `mwe:*`) par sense_id — utilisé par S6b (pipeline/sense_fr_frontier.py,
# pipeline/sense_fr.py, pipeline/sense_fr_adjudicate.py, pipeline/score.py)
# pour retrouver les phrases RÉELLES du livre COURANT associées à un
# sense_id. Vit ici (pas dans un module S6b) parce que c'est ce module qui
# écrit senses.jsonl et construit le contexte élargi par segment
# (build_wide_context_from_segments, réutilisé tel quel pour les MWE), et
# parce que sense_fr.py ne peut pas importer sense_fr_frontier.py sans
# créer un cycle (sense_fr_frontier importe déjà sense_fr).
#
# IMPORTANT — ne JAMAIS persister le résultat dans data/sense_fr.jsonl :
# ce magasin est le dictionnaire sens->traduction PERMANENT, réutilisé
# d'un livre à l'autre (voir pipeline/sense_fr.py) ; les phrases qu'il
# renvoie ici sont, elles, propres au livre dont pipeline_out/senses.jsonl
# est le produit du run COURANT — jamais valables tel quel pour un autre
# livre. Recalculer à chaque run/à chaque sortie plutôt que de mettre en
# cache dans une structure partagée entre livres — le cache module-level
# `_mwe_occurrences_cache` ci-dessous ne survit qu'un seul processus (un
# seul run), même principe que `_segment_index_cache` plus haut.
# ============================================================


def load_occurrences_by_sense() -> dict[str, list[dict]]:
    """Index sense_id -> occurrences (context, target_surface, segment_idx)
    depuis pipeline_out/senses.jsonl (le livre du run courant), fusionné
    avec load_mwe_occurrences_by_key() ci-dessous pour les clés `mwe:*`
    (pipeline_out/selected_mwe.jsonl) : les deux sources ont des espaces
    de clés disjoints par construction (sense_id WordNet vs préfixe
    `mwe:`), donc la fusion est une simple mise à jour de dict."""
    by_sense: dict[str, list[dict]] = {}
    decoder = json.JSONDecoder(strict=False)
    n_corrupt = 0
    with config.SENSES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                occ = decoder.decode(line)
            except json.JSONDecodeError:
                n_corrupt += 1
                continue
            best = occ.get("best_sense")
            context = occ.get("context")
            target = occ.get("target_surface") or occ.get("word")
            if not best or best == "aucun_sens_adapte" or not context or not target:
                continue
            by_sense.setdefault(best, []).append({
                "context": context, "target_surface": target,
                "segment_idx": occ.get("segment_idx", 0),
            })
    if n_corrupt:
        print(f"  ({n_corrupt} ligne(s) corrompue(s) ignorée(s) dans {config.SENSES_PATH})")
    by_sense.update(load_mwe_occurrences_by_key())
    return by_sense


_mwe_occurrences_cache: dict[str, list[dict]] | None = None


def load_mwe_occurrences_by_key() -> dict[str, list[dict]]:
    """Clé `mwe:{canonical_form}:{label}` -> occurrences, MÊME forme que
    pour les mots ({context, target_surface, segment_idx}) : la fenêtre de
    contexte est construite par le même build_wide_context_from_segments
    que pour un mot seul (±config.CONTEXT_WINDOW segments), donc une MWE
    et un mot montrent au modèle exactement le même genre d'extrait.

    Source : pipeline_out/selected_mwe.jsonl (occurrence_segment_idxs,
    voir select.py::build_mwe_units) pour la liste des segments, et
    pipeline_out/mwe_confirmed_spans.jsonl pour la forme fléchie
    réellement rencontrée dans chaque segment (repli sur canonical_form
    si le span est introuvable). Renvoie {} si selected_mwe.jsonl
    n'existe pas encore (même garde que score.build_mwe_units)."""
    global _mwe_occurrences_cache
    if _mwe_occurrences_cache is not None:
        return _mwe_occurrences_cache

    if not config.SELECTED_MWE_PATH.exists():
        _mwe_occurrences_cache = {}
        return _mwe_occurrences_cache

    with config.SELECTED_MWE_PATH.open(encoding="utf-8") as f:
        mwe_units = [json.loads(l) for l in f]

    surface_by_segment: dict[tuple[str, int], str] = {}
    if config.MWE_SPANS_PATH.exists():
        with config.MWE_SPANS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for span in row.get("spans", []):
                    surface_by_segment[(span["idiom"], span["segment_idx"])] = span["surface"]

    segments = load_segments()
    by_key: dict[str, list[dict]] = {}
    for u in mwe_units:
        key = f"mwe:{u['canonical_form']}:{u['label']}"
        occs = []
        for seg_idx in u["occurrence_segment_idxs"]:
            wide = build_wide_context_from_segments(segments, seg_idx)
            if not wide["text"]:
                continue
            surface = surface_by_segment.get((u["canonical_form"], seg_idx), u["canonical_form"])
            occs.append({
                "context": wide["text"], "target_surface": surface,
                "segment_idx": seg_idx,
            })
        by_key[key] = occs

    _mwe_occurrences_cache = by_key
    return by_key


def pick_diverse_occurrences(occurrences: list[dict], k: int) -> list[dict]:
    """k occurrences réparties dans le livre (pas les k premières —
    l'ordre de segment_idx couvre le début/milieu/fin) : des contextes
    différents sont ce qui rend un désaccord entre occurrences informatif
    plutôt que redondant."""
    ordered = sorted(occurrences, key=lambda o: o["segment_idx"])
    n = len(ordered)
    if n <= k:
        return ordered
    step = n / k
    picked_idx = sorted({int(i * step) for i in range(k)})
    return [ordered[i] for i in picked_idx]


def run(top_k: int | None = None) -> int:
    config.ensure_out_dir()

    with config.SELECTED_TYPES_PATH.open(encoding="utf-8") as f:
        types = [json.loads(l) for l in f]

    types.sort(key=coarse_priority, reverse=True)

    segments = load_segments()
    total = sum(len(t["occurrence_segment_idxs"]) for t in types)

    import time
    started = time.time()
    n_fallback = 0
    processed = 0
    types_processed_fully = 0
    records: list[dict] = []

    for t in types:
        use_full = top_k is None or types_processed_fully < top_k
        types_processed_fully += 1

        for seg_idx in t["occurrence_segment_idxs"]:
            if use_full:
                record = analyze_occurrence(t["lemma"], t["wn_pos"], segments, seg_idx)
            else:
                synsets = get_synsets(t["lemma"], t["wn_pos"])
                if not synsets:
                    record = None
                else:
                    record = most_frequent_sense_fallback(t["lemma"], t["wn_pos"], synsets)
                    record["segment_idx"] = seg_idx
                    n_fallback += 1

            processed += 1
            if processed % 50 == 0 or processed == total:
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed > 0 else 0
                eta_min = (total - processed) / rate / 60 if rate > 0 else float("inf")
                print(f"  {processed}/{total} ({rate:.1f}/s, ETA {eta_min:.0f} min, "
                      f"{n_fallback} en repli sans GlossBERT)", flush=True)
            if record is None:
                continue
            records.append(record)

    # Lot 0 — écriture atomique (pipeline/atomic.py) : le fichier
    # n'apparaît qu'une fois complet, ce qui élimine la classe de
    # corruption observée (lignes tronquées/entrelacées par deux runs
    # concurrents ayant chacun ouvert senses.jsonl en mode "w" — voir
    # atomic.py pour le diagnostic complet). Accumuler en mémoire avant
    # d'écrire est sans risque ici : quelques milliers de records au plus.
    n = atomic.atomic_write_jsonl(config.SENSES_PATH, records)

    print(f"{n} occurrences désambiguïsées -> {config.SENSES_PATH} "
          f"({n - n_fallback} via GlossBERT, {n_fallback} en repli sens dominant)")
    return 0


def dedupe_senses_file() -> int:
    """Maintenance ponctuelle : nettoie un `senses.jsonl` déjà corrompu par
    d'anciens runs concurrents (voir le diagnostic dans pipeline/atomic.py)
    SANS refaire tourner GlossBERT/l'arbitrage LLM. Ignore les lignes
    illisibles, déduplique sur la clé (word, pos, segment_idx,
    target_surface) en gardant le DERNIER enregistrement rencontré (le
    plus susceptible d'être le résultat d'un run complet plutôt que
    partiel). Écrit atomiquement. Retourne le nombre de lignes retenues.

    Note : cette clé n'est pas un `occurrence_id` complet (pas de
    token_i/offset ici) — deux occurrences distinctes du même mot dans le
    même segment seraient confondues. C'est un cas marginal pour ce livre
    (non observé) ; la correction structurelle passe par
    `lexical_inventory.jsonl` (lot 3 du plan), qui portera les offsets
    exacts jusqu'à cette étape."""

    if not config.SENSES_PATH.exists():
        print(f"{config.SENSES_PATH} n'existe pas, rien à nettoyer.")
        return 0

    seen: dict[tuple, dict] = {}
    n_bad_lines = 0
    n_lines = 0
    with config.SENSES_PATH.open(encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_bad_lines += 1
                continue
            key = (record.get("word"), record.get("pos"), record.get("segment_idx"),
                   record.get("target_surface"))
            seen[key] = record  # le dernier gagne

    n = atomic.atomic_write_jsonl(config.SENSES_PATH, seen.values())
    n_dupes = n_lines - n_bad_lines - n
    print(f"{n_lines} lignes lues, {n_bad_lines} illisibles écartées, "
          f"{n_dupes} doublon(s) écarté(s), {n} enregistrement(s) conservé(s) "
          f"-> {config.SENSES_PATH}")
    return n


if __name__ == "__main__":
    import sys
    if "--dedupe" in sys.argv:
        dedupe_senses_file()
        raise SystemExit(0)
    raise SystemExit(run())
