"""S6b — Traduction française de référence, indexée par sens (pas par
occurrence), construite dans un magasin PERMANENT (data/sense_fr.jsonl,
versionné, réutilisé d'un livre à l'autre) plutôt que dans pipeline_out/
(régénéré à chaque run).

Contexte : `meaning_fr` (export.py/score.py) vient de `fr_hits`, c'est-
à-dire de l'intersection entre les lemmes omw-fr/WoNeF du sens retenu
et la phrase française du livre (elle-même traduite par un LLM local,
potentiellement fausse). Aucune des deux ressources n'est fiable à
100% : omw-fr (WOLF) et WoNeF sont construites automatiquement. La
seule garantie possible est donc une VALIDATION HUMAINE — cette étape
ne fait que réduire son volume au strict nécessaire.

Principe (mesuré par pipeline/sense_fr_probe.py sur "The Humans" —
849 sense_id : 288 concordances omw-fr/WoNeF, 380 couverture par une
seule des deux, 108 sans aucune ressource, 73 divergentes) :

- si omw-fr ET WoNeF proposent au moins une traduction en commun
  (même racine), c'est déjà une concordance entre deux ressources
  INDÉPENDANTES -> accepté automatiquement (`auto_strong`). La rétro-
  traduction LLM (quand le LLM est disponible) est journalisée comme
  confirmation supplémentaire mais n'est pas une condition : la rendre
  bloquante ferait dépendre l'acceptation automatique de la
  disponibilité d'un service tiers, y compris pour les cas déjà les
  mieux corroborés par deux ressources indépendantes ;
- si UNE SEULE des deux ressources a une proposition, elle n'est
  acceptée automatiquement que si le LLM local, interrogé SANS le
  contexte du livre (traduction "de dictionnaire", à partir des
  lemmes + définition WordNet seuls) et sur plusieurs formulations de
  prompt, converge vers la même proposition, ET que la rétro-
  traduction confirme ;
- si aucune ressource lexicale ne couvre le sens, ou si les deux se
  contredisent sans aucun recoupement, ou pour toute unité multi-mots
  (MWE — pas de synset, donc aucune ressource possible par
  construction) : toujours `pending`, jamais d'acceptation
  automatique, quel que soit l'avis du LLM seul. Un LLM interrogé
  plusieurs fois sur lui-même n'est PAS une source indépendante (voir
  le plan §5.5 sur la circularité) — au mieux un filtre de cohérence,
  jamais une preuve.

Seuls `validated` (relu par un humain, via sense_fr_commit.py) et
`auto_strong` sont exportés par export.py/score.py. `pending` n'émet
jamais de texte français.
"""

from __future__ import annotations

import bz2
import difflib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date

from nltk.corpus import wordnet as nwn
from nltk.corpus.reader.wordnet import WordNetError

from pipeline import config, llm, senses

# ============================================================
# Ressources lexicales statiques (omw-fr, WoNeF)
# ============================================================

_wonef_fscore_cache: dict[str, list[str]] | None = None


def load_wonef_fscore() -> dict[str, list[str]]:
    """wonef-precision.xml (config.WONEF_PRECISION_PATH, utilisé par
    senses.load_wonef_precision) est absent du dépôt — seule la
    variante f-score, compressée, est présente. Même format XML,
    parsing identique."""
    global _wonef_fscore_cache
    if _wonef_fscore_cache is not None:
        return _wonef_fscore_cache

    if not config.WONEF_FSCORE_PATH.exists():
        _wonef_fscore_cache = {}
        return _wonef_fscore_cache

    with bz2.open(config.WONEF_FSCORE_PATH, "rb") as f:
        tree = ET.parse(f)

    wonef_by_id: dict[str, list[str]] = {}
    for synset_element in tree.getroot().iter():
        if synset_element.tag.split("}")[-1].upper() != "SYNSET":
            continue
        synset_id = None
        literals: list[str] = []
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

    _wonef_fscore_cache = wonef_by_id
    return _wonef_fscore_cache


def clean_french_candidates(candidates: list[str], english_lemmas: list[str]) -> list[str]:
    """Retire les candidats vides et ceux identiques à un lemme
    anglais du synset (mots invariants comme "score", "duplex")."""
    english_keys = {lemma.casefold() for lemma in english_lemmas}
    cleaned = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate == "_EMPTY_":
            continue
        if candidate.casefold() in english_keys:
            continue
        if candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned


def fr_candidates_omw(offset: str, pos: str, english_lemmas: list[str]) -> list[str]:
    """Traductions omw-fr agrégées sur TOUS les lemmes anglais du
    synset (la traduction d'un SENS ne dépend pas de la forme de
    surface qui y a mené dans ce livre précis). Recherche le synset
    `wn` directement par identifiant omw-en-{offset}-{pos}, avec le
    pos EXACT (satellite "s" inclus) — contrairement à
    senses.fr_lemmas_for_synset, qui reçoit le pos "grossier" du type
    ("a" même pour un sens satellite) et rate donc silencieusement ces
    sens (constaté avec clear.s.11 / "decipherable" : 0 résultat avec
    pos="a", 2 résultats avec le pos réel "s")."""
    EN = senses.get_en_wordnet()
    try:
        wn_synset = EN.synset(f"omw-en-{offset}-{pos}")
    except Exception:
        return []

    candidates: list[str] = []
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
            if lemma and lemma not in candidates:
                candidates.append(lemma)

    return clean_french_candidates(candidates, english_lemmas)


def fr_candidates_wonef(offset: str, pos: str, english_lemmas: list[str]) -> list[str]:
    wonef_by_id = load_wonef_fscore()
    literals = wonef_by_id.get(f"eng-30-{offset}-{pos}", [])
    return clean_french_candidates(literals, english_lemmas)


# ============================================================
# LLM — traduction de dictionnaire (sans contexte de livre) + rétro-traduction
# ============================================================

TRANSLATE_SYSTEM = (
    "Tu es lexicographe bilingue anglais-français. On te donne un sens précis "
    "d'un mot ou d'une expression anglaise (lemmes, catégorie grammaticale, "
    "définition WordNet), SANS phrase d'exemple tirée d'un livre particulier. "
    "Donne la traduction française la plus naturelle de CE SENS, celle qu'on "
    "trouverait dans un dictionnaire bilingue."
)

# Formulations distinctes pour obtenir plusieurs tirages malgré
# LLM_TEMPERATURE=0.0 (le cache de llm.py est indexé sur le texte
# exact du prompt) — voir config.SENSE_FR_LLM_DRAWS. Rappel : même
# modèle, mêmes poids -> un accord entre ces tirages est un indice de
# cohérence interne, pas une source indépendante.
TRANSLATE_INSTRUCTIONS = [
    "Quelle est la meilleure traduction française de ce sens ?",
    "Donne l'équivalent français le plus naturel pour ce sens précis, "
    "hors de toute phrase particulière.",
    "Comment traduirais-tu ce sens dans un dictionnaire bilingue anglais-français ?",
]

TRANSLATE_TEMPLATE = """Mot(s) anglais : {lemmas}
Catégorie grammaticale : {pos_label}
Définition (WordNet) : {definition}
Exemple(s) WordNet : {examples}

{instruction}

Réponds en JSON strict :
{{"fr": "<traduction principale, un mot ou une courte expression>", "fr_alt": ["<variante acceptable>", "..."]}}
"""

BACKTRANSLATE_SYSTEM = (
    "Tu es traducteur français-anglais. On te donne un mot ou une courte "
    "expression français, et le sens anglais précis qu'il est censé traduire "
    "(définition WordNet d'origine). Donne la traduction anglaise la plus "
    "naturelle de ce mot ou cette expression, dans CE sens précis."
)

BACKTRANSLATE_TEMPLATE = """Mot ou expression français : "{fr_candidate}"
Sens visé (définition anglaise d'origine) : {definition}

Réponds en JSON strict :
{{"en": "<traduction anglaise>"}}
"""

POS_LABELS = {"n": "nom", "v": "verbe", "a": "adjectif", "s": "adjectif (satellite)", "r": "adverbe"}

# llm.is_available() fait un aller-retour réseau (timeout 5s) : appelé
# une fois par sense_id ET par tentative de rétro-traduction (jusqu'à
# ~2000 fois sur ce livre), il rendait le run interminable dès que
# l'endpoint est injoignable. Un seul vrai ping par process.
_llm_available: bool | None = None


def llm_is_available() -> bool:
    global _llm_available
    if _llm_available is None:
        _llm_available = llm.is_available()
    return _llm_available


def llm_translate_votes(
    lemmas_en: list[str], pos: str, definition_en: str, examples: list[str]
) -> list[tuple[str, list[str]]]:
    """SENSE_FR_LLM_DRAWS tentatives de traduction, chacune avec une
    formulation différente. Renvoie la liste des (fr, fr_alt) obtenus
    (liste plus courte que SENSE_FR_LLM_DRAWS si le LLM est
    indisponible ou renvoie une erreur sur certains tirages)."""
    if not llm_is_available():
        return []

    votes = []
    body = TRANSLATE_TEMPLATE
    for i in range(config.SENSE_FR_LLM_DRAWS):
        instruction = TRANSLATE_INSTRUCTIONS[i % len(TRANSLATE_INSTRUCTIONS)]
        prompt = body.format(
            lemmas=", ".join(lemmas_en),
            pos_label=POS_LABELS.get(pos, pos),
            definition=definition_en or "?",
            examples="; ".join(examples) if examples else "(aucun)",
            instruction=instruction,
        )
        try:
            result = llm.call_json(prompt, system=TRANSLATE_SYSTEM, timeout=120)
        except llm.LLMError:
            continue
        fr = (result.get("fr") or "").strip()
        if not fr:
            continue
        fr_alt = [a.strip() for a in (result.get("fr_alt") or []) if a and a.strip()]
        votes.append((fr, fr_alt))
    return votes


def llm_backtranslate(fr_candidate: str, definition_en: str) -> str | None:
    if not llm_is_available():
        return None
    prompt = BACKTRANSLATE_TEMPLATE.format(fr_candidate=fr_candidate, definition=definition_en or "?")
    try:
        result = llm.call_json(prompt, system=BACKTRANSLATE_SYSTEM, timeout=120)
    except llm.LLMError:
        return None
    en = (result.get("en") or "").strip()
    return en or None


_BT_PREFIX_RE = re.compile(r"^(to|the|a|an)\s+", re.I)


def _bt_normalize(text: str) -> str:
    text = _BT_PREFIX_RE.sub("", text.strip().casefold())
    return text.strip(" .,;:!?\"'")


def _synset_neighbours(synset) -> set:
    """Voisinage sémantique immédiat d'un synset : lui-même, ses
    `similar_tos()` (aller-retour — indispensable pour les satellites
    "s" : adequate.s.01 -> sufficient.a.01 n'est atteint que dans ce
    sens) et les synsets dérivationnellement apparentés à ses lemmes."""
    out = {synset}
    out.update(synset.similar_tos())
    for other in synset.similar_tos():
        out.update(other.similar_tos())
    for lemma in synset.lemmas():
        for derived in lemma.derivationally_related_forms():
            out.add(derived.synset())
    return out


def backtranslation_matches(en_guess: str, english_lemmas: list[str], synset=None) -> bool:
    """Une rétro-traduction correcte n'est pas forcément un lemme du
    synset d'origine : le LLM répond souvent par un synonyme fidèle au
    SENS ("to behave" pour act.v.02, "sufficient" pour adequate.s.01)
    plutôt que par le mot anglais exact. Comparer les CHAÎNES (même
    tolérant, via difflib) confond donc "synonyme correct" et
    "traduction fausse mais orthographiquement proche" — mesuré sur ce
    magasin : 3 des 4 rétro-traductions correctes déjà en cache étaient
    rejetées par l'ancien test.

    Test principal, sémantique : la rétro-traduction retombe-t-elle,
    via WordNet, dans le voisinage immédiat (_synset_neighbours) du
    synset visé ? Repli orthographique (casefold exact ou similarité
    difflib élevée) pour les mots absents de WordNet (formes fléchies,
    expressions) — comportement d'origine, conservé en secours."""
    guess_key = _bt_normalize(en_guess)

    if synset is not None:
        candidates = nwn.synsets(guess_key.replace(" ", "_"))
        if candidates and (set(candidates) & _synset_neighbours(synset)):
            return True

    for lemma in english_lemmas:
        lemma_key = lemma.casefold()
        if guess_key == lemma_key:
            return True
        if difflib.SequenceMatcher(None, guess_key, lemma_key).ratio() >= 0.82:
            return True
    return False


# ============================================================
# Collecte des clés à traiter (mots + MWE)
# ============================================================

def collect_targets() -> dict[str, dict]:
    """Une entrée par clé du magasin. Les unités "word" partageant le
    même sense_id (deux lemmes désambiguïsés vers le même synset) sont
    fusionnées : la traduction d'un SENS ne dépend pas du mot du livre
    qui y a mené.

    Recalcule directement depuis senses.jsonl/selected_mwe.jsonl via
    score.build_records/aggregate_and_score/build_mwe_units — PAS
    depuis vocab.jsonl, qui n'existe pas encore à ce point du pipeline
    (cette étape tourne entre S5 "senses" et S7 "export" ; export.py
    est ce qui écrit vocab.jsonl, et lit en retour le magasin que
    cette étape vient de remplir, pour fr_opacity et
    meaning_fr_official). Le recalcul est pur CPU, sans LLM : peu
    coûteux comparé au reste du run."""
    from pipeline.score import aggregate_and_score, build_mwe_units, build_records

    targets: dict[str, dict] = {}

    for u in aggregate_and_score(build_records()):
        key = u["sense_id"]
        entry = targets.setdefault(key, {
            "key": key, "kind": "synset", "lemmas_en": [], "occurrences": 0,
        })
        if u["canonical_form"] not in entry["lemmas_en"]:
            entry["lemmas_en"].append(u["canonical_form"])
        entry["occurrences"] += u["occurrences"]

    for u in build_mwe_units():
        key = f"mwe:{u['canonical_form']}:{u['sense_id']}"
        targets[key] = {
            "key": key, "kind": "mwe", "lemmas_en": [u["canonical_form"]],
            "occurrences": u["occurrences"], "definition_en": u["definition_en"],
            "mwe_label": u["sense_id"],
        }

    return targets


# ============================================================
# Classification par sens (word) — omw-fr / WoNeF / LLM / rétro-traduction
# ============================================================

def classify_synset_key(target: dict, diag: Counter | None = None) -> dict:
    """Construit l'entrée complète du magasin pour un sense_id
    (synset), avec ses preuves et son statut (`auto_strong` ou
    `pending`).

    `diag`, si fourni, reçoit un décompte par "porte" franchie ou non
    pour les entrées `source_unique` — sert uniquement au diagnostic
    de --retry-pending (voir run()), jamais stocké dans le magasin."""
    sense_id = target["key"]
    entry_base = {
        "key": sense_id, "kind": "synset", "lemmas_en": target["lemmas_en"],
        "occurrences": target["occurrences"],
    }

    try:
        synset = nwn.synset(sense_id)
    except (WordNetError, ValueError):
        entry_base.update({
            "pos": None, "definition_en": None, "fr": None, "fr_alt": [],
            "status": "pending", "agreement": "sense_id_non_resolu",
            "evidence": {}, "decided_at": None, "decided_by": None, "note": "",
        })
        return entry_base

    offset = f"{synset.offset():08d}"
    pos = synset.pos()
    english_lemmas = [l.name().replace("_", " ") for l in synset.lemmas()]
    definition_en = synset.definition()
    examples = synset.examples()

    omw = fr_candidates_omw(offset, pos, english_lemmas)
    wonef = fr_candidates_wonef(offset, pos, english_lemmas)
    omw_stems = {senses.fr_stem(c) for c in omw}
    wonef_stems = {senses.fr_stem(c) for c in wonef}

    llm_votes_raw = llm_translate_votes(english_lemmas, pos, definition_en, examples)
    llm_vote_counts = Counter(senses.fr_stem(fr) for fr, _ in llm_votes_raw)
    llm_by_stem = {senses.fr_stem(fr): fr for fr, _ in llm_votes_raw}
    llm_alt_by_stem: dict[str, list[str]] = {}
    for fr, alt in llm_votes_raw:
        llm_alt_by_stem.setdefault(senses.fr_stem(fr), []).extend(alt)

    llm_consensus_stem = None
    if llm_vote_counts:
        stem, count = llm_vote_counts.most_common(1)[0]
        if count >= config.SENSE_FR_LLM_MIN_AGREE:
            llm_consensus_stem = stem

    status = "pending"
    agreement = "aucune_source"
    fr, fr_alt = None, []
    backtranslation_ok = False

    overlap = omw_stems & wonef_stems
    if overlap:
        # Deux ressources construites INDÉPENDAMMENT (omw-fr/WOLF et
        # WoNeF) recoupent déjà leurs traductions : c'est la
        # concordance elle-même, entre deux sources distinctes, qui
        # vaut acceptation automatique — la rétro-traduction LLM
        # (quand disponible) n'est qu'une confirmation supplémentaire
        # journalisée dans `evidence`, jamais une condition bloquante.
        # La rendre bloquante ferait dépendre TOUT le mécanisme
        # d'acceptation automatique de la disponibilité du LLM local,
        # y compris pour les cas déjà les mieux corroborés.
        agreement = "concordantes"
        status = "auto_strong"
        best_stem = next(iter(overlap))
        fr = next((c for c in omw if senses.fr_stem(c) == best_stem), None) or \
            next((c for c in wonef if senses.fr_stem(c) == best_stem), None)
        fr_alt = sorted({c for c in omw + wonef if c != fr}, key=len)
        en_guess = llm_backtranslate(fr, definition_en) if fr else None
        backtranslation_ok = bool(
            en_guess and backtranslation_matches(en_guess, english_lemmas, synset)
        )
    elif omw and wonef:
        # "divergentes" : les DEUX ressources ont un avis, et elles se
        # CONTREDISENT (sinon on serait dans la branche `overlap`
        # ci-dessus). Toujours `pending`, quel que soit l'avis du LLM —
        # un vrai désaccord entre deux références se tranche par un
        # humain, pas par un troisième avis qui prend parti pour l'une
        # des deux (voir la docstring du module). Seule la suggestion
        # pré-remplie profite du LLM, jamais le statut.
        agreement = "divergentes"
        pool = omw + wonef
        fr = llm_by_stem.get(llm_consensus_stem) if llm_consensus_stem else pool[0]
        fr_alt = sorted(set(pool) - {fr}, key=len)
    elif omw or wonef:
        # "source_unique" : une seule ressource a un avis, l'autre est
        # muette (pas de contradiction active). Ici, un consensus LLM
        # indépendant qui confirme CETTE ressource, plus une rétro-
        # traduction qui retombe sur le sens anglais d'origine, élève
        # la confiance à un niveau comparable à `concordantes`.
        single_source = omw or wonef
        single_stems = omw_stems or wonef_stems
        agreement = "source_unique"
        # Le stem qui compte est celui du fr PRINCIPAL majoritaire
        # (llm_consensus_stem), jamais une simple mention en fr_alt.
        # Une variante `fr_alt` est, par construction du prompt, une
        # "variante acceptable" plus faible que la réponse principale
        # — l'accepter comme preuve d'accord avec la ressource laisse
        # une mention secondaire écraser le vrai choix du LLM. Mesuré
        # sur ce magasin : privacy.n.01 promu vers "solitude" (le seul
        # mot de WoNeF) alors que le LLM répondait "intimité" à
        # l'unanimité en fr principal, "solitude" n'apparaissant qu'en
        # fr_alt à chaque tirage — une contresens promu automatiquement.
        matching_stem = (
            llm_consensus_stem if llm_consensus_stem in single_stems else None
        )
        if matching_stem:
            fr = next((c for c in single_source if senses.fr_stem(c) == matching_stem), None)
            fr_alt = sorted({c for c in single_source if c != fr}, key=len)
            en_guess = llm_backtranslate(fr, definition_en) if fr else None
            backtranslation_ok = bool(
                en_guess and backtranslation_matches(en_guess, english_lemmas, synset)
            )
            if backtranslation_ok:
                status = "auto_strong"
        if status != "auto_strong":
            # Rien d'accepté automatiquement : on pré-remplit quand même
            # la meilleure suggestion pour la relecture humaine.
            fr = llm_by_stem.get(llm_consensus_stem) if llm_consensus_stem else single_source[0]
            fr_alt = sorted(set(single_source) - {fr}, key=len)
    else:
        agreement = "aucune_source"
        if llm_consensus_stem:
            fr = llm_by_stem[llm_consensus_stem]
            fr_alt = sorted(set(llm_alt_by_stem.get(llm_consensus_stem, [])) - {fr}, key=len)
        # Jamais auto_strong ici : pas de ressource lexicale du tout,
        # donc le LLM (même unanime sur lui-même) reste une hypothèse,
        # pas une preuve (plan §5.5).

    if diag is not None and agreement == "source_unique":
        if not llm_votes_raw:
            diag["source_unique/pas_de_votes_llm"] += 1
        elif matching_stem is None:
            diag["source_unique/consensus_hors_ressource"] += 1
        elif status != "auto_strong":
            diag["source_unique/retrotraduction_echouee"] += 1
        else:
            diag["source_unique/promue"] += 1

    entry_base.update({
        "pos": pos, "definition_en": definition_en,
        "fr": fr, "fr_alt": fr_alt,
        "status": status, "agreement": agreement,
        "evidence": {
            "omw_fr": omw, "wonef": wonef,
            "llm_votes": {llm_by_stem.get(s, s): c for s, c in llm_vote_counts.items()},
            "backtranslation_ok": backtranslation_ok,
        },
        "decided_at": date.today().isoformat() if status == "auto_strong" else None,
        "decided_by": "auto" if status == "auto_strong" else None,
        "note": "",
    })
    return entry_base


def classify_mwe_key(target: dict) -> dict:
    """Les MWE n'ont pas de synset : aucune ressource lexicale
    possible par construction. Toujours `pending` — un idiome ne peut
    être validé que par un humain (ou rester sans équivalent)."""
    lemmas_en = target["lemmas_en"]
    definition_en = target.get("definition_en")
    llm_votes_raw = llm_translate_votes(lemmas_en, "mwe", definition_en, [])
    llm_vote_counts = Counter(senses.fr_stem(fr) for fr, _ in llm_votes_raw)
    fr, fr_alt = None, []
    if llm_votes_raw:
        stem, _ = llm_vote_counts.most_common(1)[0]
        fr = next(f for f, _ in llm_votes_raw if senses.fr_stem(f) == stem)
        fr_alt = sorted({f for f, _ in llm_votes_raw if f != fr}, key=len)

    return {
        "key": target["key"], "kind": "mwe", "lemmas_en": lemmas_en,
        "occurrences": target["occurrences"],
        "pos": None, "definition_en": definition_en,
        "fr": fr, "fr_alt": fr_alt,
        "status": "pending", "agreement": "mwe_sans_ressource",
        "evidence": {"llm_votes": {f: c for f, c in llm_vote_counts.items()}},
        "decided_at": None, "decided_by": None, "note": "",
    }


# ============================================================
# Magasin (data/sense_fr.jsonl)
# ============================================================

def load_store() -> dict[str, dict]:
    if not config.SENSE_FR_STORE_PATH.exists():
        return {}
    store = {}
    with config.SENSE_FR_STORE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            store[entry["key"]] = entry
    return store


def write_store(store: dict[str, dict]) -> None:
    config.ensure_data_dir()
    with config.SENSE_FR_STORE_PATH.open("w", encoding="utf-8") as f:
        for key in sorted(store):
            f.write(json.dumps(store[key], ensure_ascii=False) + "\n")


def format_occurrences_en(
    occurrences: list[dict], limit: int = config.SENSE_FR_FRONTIER_MAX_OCCURRENCES
) -> str:
    """Formatage ("phrase1 || phrase2") des phrases du livre COURANT sur
    lesquelles une traduction s'appuie — `occurrences` vient TOUJOURS de
    senses.load_occurrences_by_sense() sur le run en cours (TOUTES les
    occurrences de ce sens dans le livre, potentiellement des dizaines —
    "beat" apparaît 30 fois dans "The Humans"), jamais du magasin
    data/sense_fr.jsonl (qui est le dictionnaire sens->traduction
    PERMANENT, réutilisé d'un livre à l'autre : il ne doit jamais porter
    de texte propre à un seul livre — voir la docstring de
    sense_fr_frontier.build_entry).

    `limit` : replafonne ICI, au moment du formatage, via
    senses.pick_diverse_occurrences — sans ce plafond une cellule CSV
    peut atteindre plusieurs milliers de caractères (mesuré : 30
    occurrences de "beat" jointes = ~8000 caractères), illisible et
    cassant l'affichage Excel. Même plafond que celui montré au modèle
    au moment de la décision (config.SENSE_FR_FRONTIER_MAX_OCCURRENCES)
    par défaut, pour que la phrase affichée corresponde à celle
    réellement vue par le modèle quand cette entrée vient d'être décidée.

    Même format partout où une traduction est exposée (sense_fr_review.csv,
    sense_id_suspects.csv, vocab.csv/vocab.jsonl, sense_fr_adjudication.csv),
    pour que pipeline_out/ (hors cache/) soit auto-suffisant pour l'audit
    d'une traduction SUR LE LIVRE COURANT."""
    picked = senses.pick_diverse_occurrences(occurrences, limit) if occurrences else []
    return " || ".join(o["context"] for o in picked)


# ============================================================
# File de relecture
# ============================================================

AGREEMENT_RANK = {
    "sense_id_non_resolu": 0,
    "aucune_source": 0,
    "mwe_sans_ressource": 0,
    "divergentes": 1,
    "source_unique": 2,
}

# `reassigner_vers` (colonne optionnelle, remplie à la main comme
# fr_final/decision/note — normalement produite par pipeline/review_ui.py,
# une petite page HTML locale qui propose la liste des sense_id WordNet du
# mot avec leur définition, pour ne jamais avoir à le taper soi-même) :
# quand `contexte_en` montre que les occurrences de `key` n'appartiennent
# PAS au sens affiché (ex. "smart-ass" coupé en "ass" seul par la
# tokenisation, "e-mail" coupé en "mail" seul — voir le plan du 2026-08-27
# "Correction manuelle smart-ass / e-mail sans re-run complet"), le
# relecteur écrit ici la VRAIE clé : soit un sense_id WordNet existant
# ("e-mail.v.01"), soit une clé "mwe:<expression>:<idiome|phrasal_verb|
# semi_fige>" pour promouvoir une expression composée absente de WordNet
# (ex. "mwe:smart ass:idiome"). `fr_final`/`decision=ok` restent
# obligatoires, comme pour une validation normale, mais portent alors sur
# la NOUVELLE clé — voir pipeline/sense_fr_commit.py, qui écrit
# data/manual_corrections.jsonl en plus du magasin. Ne s'applique qu'aux
# entrées `kind == "synset"` (une occurrence de MOT mal groupée par S5) ;
# sans effet sur une entrée `kind == "mwe"` (rien à re-clé, uniquement une
# traduction à corriger via fr_final comme d'habitude).
# `definition_en_perso` (optionnelle, utile seulement avec une clé
# "mwe:...") : glose anglaise tapée à la main pour une expression toute
# neuve, jamais vue par pipeline/mwe.py::CUSTOM_IDIOMS — sinon la glose
# existante de CUSTOM_IDIOMS est reprise automatiquement, voir
# sense_fr_commit.py::derive_reassignment.
REVIEW_FIELDS = [
    "key", "kind", "lemmas_en", "pos", "definition_en", "occurrences", "agreement",
    "suggested_fr", "suggested_fr_alt", "omw_fr", "wonef", "frontier_confidence",
    "contexte_en", "sense_fit", "sense_fit_note",
    "definition_en_perso", "reassigner_vers", "fr_final", "fr_alt_final", "decision", "note",
]


def build_review_row(e: dict, occurrences_by_sense: dict[str, list[dict]]) -> dict:
    """Une entrée du magasin -> une ligne de relecture (REVIEW_FIELDS,
    colonnes de décision vides — à remplir par le relecteur), + quelques
    champs informatifs hors REVIEW_FIELDS (`surface_forms`, `status`,
    `decided_by` — ignorés par write_review_csv, DictWriter
    extrasaction="ignore", consommés par pipeline/review_ui.py).

    Factorisé pour construire une ligne à partir de N'IMPORTE QUELLE
    entrée du magasin, pas seulement `pending` — voir pending_review_rows
    ci-dessous (status == "pending") et pipeline/review_ui.py
    build_flagged_payload (déjà verrouillée mais `sense_fit == "mismatch"`,
    voir le plan du 2026-08-27 "Étendre l'IHM aux entrées verrouillées
    incohérentes")."""
    ev = e.get("evidence", {})
    occs = occurrences_by_sense.get(e["key"], [])
    return {
        "key": e["key"], "kind": e["kind"],
        "lemmas_en": "/".join(e["lemmas_en"]), "pos": e.get("pos") or "",
        "definition_en": e.get("definition_en") or "",
        "occurrences": e.get("occurrences", ""),
        "agreement": e["agreement"],
        "suggested_fr": e.get("fr") or "",
        "suggested_fr_alt": "; ".join(e.get("fr_alt") or []),
        "omw_fr": "; ".join(ev.get("omw_fr", [])),
        "wonef": "; ".join(ev.get("wonef", [])),
        "frontier_confidence": ev.get("frontier_confidence", ""),
        "contexte_en": format_occurrences_en(occs),
        # Forme(s) EXACTES du mot telles qu'elles apparaissent dans LE
        # LIVRE (occ["target_surface"], voir senses.load_occurrences_by_sense)
        # — pas les lemmes WordNet du synset (`lemmas_en` ci-dessus, qui
        # peut lister des synonymes jamais employés par cet auteur, ex.
        # feeble.s.01 -> "feeble/lame" alors que seul "lame" est dans le
        # texte). Triées par fréquence d'usage.
        "surface_forms": [
            s for s, _ in Counter(o["target_surface"] for o in occs if o.get("target_surface")).most_common(6)
        ],
        "sense_fit": e.get("sense_fit") or "",
        "sense_fit_note": e.get("sense_fit_note") or "",
        "status": e.get("status"), "decided_by": e.get("decided_by"),
        "definition_en_perso": "", "reassigner_vers": "",
        "fr_final": "", "fr_alt_final": "", "decision": "", "note": "",
    }


def pending_review_rows(
    store: dict[str, dict], occurrences_by_sense: dict[str, list[dict]] | None = None
) -> list[dict]:
    """Une ligne par entrée `pending` du magasin, triées comme
    write_review_csv ci-dessous. Factorisé pour servir DEUX sorties : le
    CSV par lot (write_review_csv) et l'API du serveur local
    `GET /api/pending` (pipeline/review_ui.py) — même contenu, même tri,
    un seul endroit qui les construit.

    `occurrences_by_sense` : voir write_review_csv."""
    occurrences_by_sense = occurrences_by_sense or {}

    pending = [e for e in store.values() if e["status"] == "pending"]
    pending.sort(key=lambda e: (
        AGREEMENT_RANK.get(e["agreement"], 1),
        -e.get("occurrences", 0),
    ))
    return [build_review_row(e, occurrences_by_sense) for e in pending]


def write_review_csv(
    store: dict[str, dict], occurrences_by_sense: dict[str, list[dict]] | None = None
) -> int:
    """Régénère la file de relecture à partir de TOUTES les entrées
    `pending` actuellement dans le magasin (pas seulement celles
    ajoutées à ce run) : la relecture peut se faire en plusieurs
    séances. `occurrences` est celui figé au moment où l'entrée a été
    créée (informatif — sert au tri, pas recalculé d'un livre à
    l'autre).

    `occurrences_by_sense` : phrases du LIVRE COURANT
    (senses.load_occurrences_by_sense()), jamais lues depuis le magasin —
    voir la docstring de format_occurrences_en. Optionnel (défaut : aucune
    phrase, `contexte_en` reste vide) pour les appelants qui n'ont pas de
    livre courant en contexte."""
    import csv

    rows = pending_review_rows(store, occurrences_by_sense)

    config.ensure_out_dir()
    with config.SENSE_FR_REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ============================================================
# Orchestration
# ============================================================

def _target_from_entry(entry: dict) -> dict:
    """Reconstruit un `target` (voir collect_targets) à partir d'une
    entrée déjà en magasin — sert à reclasser un `pending` existant
    sans dépendre du livre actuellement traité (le magasin est partagé
    entre livres ; --retry-pending doit pouvoir rebalayer TOUT ce qui
    est en attente, pas seulement les clés du run courant)."""
    return {
        "key": entry["key"], "kind": entry["kind"], "lemmas_en": entry["lemmas_en"],
        "occurrences": entry.get("occurrences", 0), "definition_en": entry.get("definition_en"),
    }


def _stratified_sample(keys_by_agreement: dict[str, list[str]], limit: int) -> list[str]:
    """Prend un peu de chaque catégorie d'agreement plutôt que les N
    premières clés (ordre alphabétique) — pour qu'un --retry-limit
    serve vraiment de test représentatif, pas juste des mots en `a`."""
    buckets = {k: list(v) for k, v in keys_by_agreement.items()}
    sample = []
    while len(sample) < limit and any(buckets.values()):
        for agreement in list(buckets):
            if len(sample) >= limit:
                break
            if buckets[agreement]:
                sample.append(buckets[agreement].pop(0))
    return sample


def run(
    retry_pending: bool = False,
    retry_limit: int | None = None,
    retry_agreement: str | None = None,
) -> int:
    config.ensure_data_dir()
    config.ensure_out_dir()

    if not llm_is_available():
        print(f"  (LLM local injoignable à {config.OLLAMA_URL} : les acceptations "
              f"automatiques nécessitant un consensus LLM seront sautées ; "
              f"les concordances omw-fr/WoNeF pures restent possibles.)")

    store = load_store()
    targets = collect_targets()

    new_keys = [k for k in targets if k not in store]
    print(f"{len(targets)} clés (sens + MWE) retenues par ce livre, "
          f"{len(store)} déjà dans le magasin, {len(new_keys)} nouvelles à classer.")

    # IMPORTANT : le magasin est incrémental — une clé déjà présente
    # (même `pending`) n'est JAMAIS reclassée automatiquement, y
    # compris si le LLM est devenu joignable entre-temps. Sans
    # --retry-pending, relancer ce script quand rien n'a changé dans
    # le livre est un no-op, par construction.
    retry_keys = []
    if retry_pending:
        by_agreement: dict[str, list[str]] = {}
        for k, e in store.items():
            if e["status"] == "pending":
                by_agreement.setdefault(e["agreement"], []).append(k)
        if retry_agreement is not None:
            # Restreint à une seule catégorie : 3 des 4 catégories
            # d'agreement (mwe_sans_ressource, aucune_source,
            # divergentes) ne peuvent JAMAIS devenir auto_strong (voir
            # la docstring du module) — sans ce filtre, un échantillon
            # non ciblé dépense la plupart de ses appels LLM sur des
            # entrées qui resteront `pending` par construction.
            by_agreement = {retry_agreement: by_agreement.get(retry_agreement, [])}
        if retry_limit is not None:
            retry_keys = _stratified_sample(by_agreement, retry_limit)
        else:
            retry_keys = [k for keys in by_agreement.values() for k in keys]
        print(f"--retry-pending : {len(retry_keys)} entrée(s) `pending` du magasin "
              f"(tous livres confondus) vont être reclassées"
              + (f", catégorie '{retry_agreement}'" if retry_agreement is not None else "")
              + (f" (échantillon limité à {retry_limit})." if retry_limit is not None else "."))

    to_process = [(k, targets[k]) for k in new_keys] + \
                 [(k, _target_from_entry(store[k])) for k in retry_keys]

    n_auto = 0
    n_promoted = 0
    diag: Counter = Counter()
    for i, (key, target) in enumerate(to_process, start=1):
        was_pending = key in store and store[key]["status"] == "pending"
        if target["kind"] == "synset":
            entry = classify_synset_key(target, diag=diag)
        else:
            entry = classify_mwe_key(target)
        store[key] = entry
        if entry["status"] == "auto_strong":
            n_auto += 1
            if was_pending:
                n_promoted += 1
        if i % 50 == 0 or i == len(to_process):
            print(f"  {i}/{len(to_process)} classées ({n_auto} auto_strong pour l'instant, "
                  f"dont {n_promoted} promues depuis `pending`)")
            # Sauvegarde intermédiaire : avec --retry-pending, ce run peut
            # porter sur plusieurs centaines d'entrées et prendre des
            # heures (jusqu'à ~4 appels LLM chacune) — écrire seulement à
            # la fin ferait tout perdre en cas d'interruption. Les appels
            # LLM eux-mêmes restent cachés sur disque (pipeline/llm.py),
            # donc une reprise après coupure ne repaie que le travail non
            # encore classé, pas les appels déjà faits.
            write_store(store)

    write_store(store)
    # Livre courant uniquement (voir format_occurrences_en) — ce chemin
    # ollama historique ne montre jamais de phrase au modèle (glose
    # seule, cf. docstring du module), mais un humain qui relit
    # sense_fr_review.csv en profite quand même.
    n_pending = write_review_csv(store, senses.load_occurrences_by_sense())

    n_validated = sum(1 for e in store.values() if e["status"] == "validated")
    n_auto_total = sum(1 for e in store.values() if e["status"] == "auto_strong")
    print(f"Magasin : {len(store)} entrées ({n_validated} validées, {n_auto_total} auto_strong, "
          f"{n_pending} en attente -> {config.SENSE_FR_REVIEW_PATH}).")
    if retry_pending:
        print(f"{n_promoted} entrée(s) promue(s) de `pending` à `auto_strong` grâce au LLM.")
    if diag:
        print("Ventilation par porte (entrées `source_unique` traitées ce run) :")
        for k, v in sorted(diag.items()):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retry-pending", action="store_true",
        help="Reclasse aussi toutes les entrées `pending` déjà en magasin (tous livres "
             "confondus) — utile après avoir retrouvé l'accès au LLM local. N'a "
             "aucun effet sur `validated`/`rejected`/`no_equivalent`.",
    )
    parser.add_argument(
        "--retry-limit", type=int, default=None,
        help="Avec --retry-pending, limite le nombre d'entrées reclassées (échantillon "
             "réparti entre les catégories d'agreement, ou une seule si --retry-agreement "
             "est aussi fourni) — pour tester avant de lancer le retry complet, "
             "potentiellement long (jusqu'à ~4 appels LLM par entrée).",
    )
    parser.add_argument(
        "--retry-agreement", choices=sorted(AGREEMENT_RANK), default=None,
        help="Avec --retry-pending, restreint le reclassement à une seule catégorie "
             "d'agreement — utile pour cibler 'source_unique' (seule catégorie où le "
             "LLM+rétro-traduction peuvent promouvoir en auto_strong ; les 3 autres "
             "restent toujours `pending` par construction).",
    )
    args = parser.parse_args()
    raise SystemExit(run(
        retry_pending=args.retry_pending,
        retry_limit=args.retry_limit,
        retry_agreement=args.retry_agreement,
    ))
