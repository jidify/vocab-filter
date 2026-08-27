"""Arbitrage final SANS relecture humaine — étape 6 du dispositif (voir le
plan "Valider / corriger suggested_fr et suggested_fr_alt"). Recalcule le
statut des entrées `pending`/`auto_llm` du magasin à partir d'un FAISCEAU
de signaux, à la place de l'unique corroboration bruitée (omw-fr/WoNeF)
qui laissait 210 lignes en désaccord (voir pipeline/sense_fr_frontier.py).

Depuis que pipeline/sense_fr_frontier.py est devenu la passe PRIMAIRE et
CONTEXTUELLE (phrases réelles + candidats mélangés, un seul appel), la
fidélité du sens (`translation_type`, `sense_fit`) est déjà tranchée EN
AMONT de ce module — une reformulation ou un sense_id douteux ne
parviennent jamais jusqu'ici en `pending`/`auto_llm` "corroborable" sans
avoir déjà été signalés. ex-`context_match` (comparer `fr` à la lecture
contextuelle d'un second appel) a donc disparu : les deux venaient
désormais du MÊME appel, la corroboration aurait été tautologique. Les
signaux comptés ici sont exclusivement des sources HORS LIGNE et
indépendantes du modèle — cf. le plan §5.5 sur la circularité.

Trois passes, DÉLIBÉRÉMENT séparées car seule la première est utilisable
sans clé API :

- **Stage A** (aucun appel LLM — normalisation, dictionnaires humains
  hors-ligne déjà extraits par pipeline/lex_bilingual.py, LaBSE local,
  wordfreq) : calcule tous les signaux, promeut en `auto_corroborated`
  ce que >=2 signaux indépendants corroborent avec tests déterministes
  au vert, et RÉTROGRADE en `pending` les `auto_llm` existants qui
  échouent ces mêmes tests déterministes (ils avaient été acceptés SANS
  aucun test — voir sense_fr_frontier.py). Entièrement testable hors
  ligne : c'est ce que ce module exécute par défaut.
- **Stage B** (--with-backtranslation, nécessite une clé API) :
  rétro-traduction LiteLLM + pipeline.sense_fr.backtranslation_matches
  (test sémantique par voisinage WordNet, pas de ressemblance de
  chaînes — déjà écrit, réutilisé tel quel). Renforce un signal unique
  de Stage A en `auto_corroborated`, ou re-certifie un `auto_llm` qui
  n'avait alors QUE les tests déterministes.
- **Stage C** (--with-judge, nécessite une clé API) : juge sur dossier,
  candidats mélangés et non étiquetés, PLUS les phrases réelles du livre
  courant (recalculées via senses.load_occurrences_by_sense(), jamais
  lues depuis le magasin — voir sa docstring), sur le résidu de A+B. Seul
  composant du dispositif autorisé à RÉÉCRIRE `fr`/`fr_alt` plutôt que de
  simplement corroborer ce qui
  existe déjà.

Le résultat, quel que soit le stage exécuté, est journalisé ligne à ligne
dans pipeline_out/sense_fr_adjudication.csv (une colonne par signal) —
c'est ce fichier qui rend un contrôle par sondage possible sans rouvrir
le magasin, et qui permet de comprendre POURQUOI une entrée a été
promue.

IMPORTANT — ce module ne retouche JAMAIS `auto_strong` (déjà la
concordance la plus forte du dispositif) et ne modifie `fr` d'une entrée
déjà verrouillée (validated/auto_strong/auto_llm/auto_corroborated/
auto_judged) que par le canal désigné de la promotion elle-même — voir
pipeline/verify_fr_lock.py, dont LOCKED_STATUSES doit inclure les
nouveaux statuts (voir pipeline/score.py et pipeline/verify_fr_lock.py).

Usage :
    uv run python -m pipeline.sense_fr_adjudicate --dry-run --limit 50
    uv run python -m pipeline.sense_fr_adjudicate
    uv run python -m pipeline.sense_fr_adjudicate --with-backtranslation --with-judge
"""

from __future__ import annotations

import csv
from datetime import date
from typing import Literal

from nltk.corpus import wordnet as nwn
from nltk.corpus.reader.wordnet import WordNetError
from wordfreq import zipf_frequency

from pipeline import config, fr_norm, inventory, lex_bilingual, senses, sense_fr

ADJUDICATION_CSV_PATH = config.OUT_DIR / "sense_fr_adjudication.csv"

# Zipf bas : l'essentiel du lexique français, même rare, dépasse ce seuil
# -- sert à attraper les coquilles réelles (knock.v.06 "critquer" au lieu
# de "critiquer") et l'anglais laissé tel quel, pas les mots peu fréquents
# mais légitimes.
WORDFREQ_MIN_ZIPF = 1.0

# Statuts déjà décidés par une source indépendante d'ARBITRAGE (pas
# seulement de proposition) — sert au test de collision polysémique :
# une entrée `pending`/`auto_llm` non encore arbitrée ne compte pas.
DECIDED_STATUSES = {"validated", "auto_strong", "auto_corroborated", "auto_judged"}


# ============================================================
# Stage A — signaux offline
# ============================================================


def _candidates_of(entry: dict) -> list[str]:
    return [c for c in [entry.get("fr")] + list(entry.get("fr_alt") or []) if c]


def resource_match(entry: dict) -> bool:
    evidence = entry.get("evidence", {})
    resources = list(evidence.get("omw_fr") or []) + list(evidence.get("wonef") or [])
    return fr_norm.any_match(_candidates_of(entry), resources)


def dbnary_match(entry: dict) -> tuple[bool, list[str] | None, float | None]:
    # Lot 4 (point 22) : plus de garde `kind != "synset"` — DBnary
    # contient déjà 136 entrées multi-mots ; `definition_en`/`lemmas_en`
    # suffisent (une MWE sans glose idioms.yml a `definition_en=None`,
    # exclue par la condition ci-dessous comme n'importe quelle entrée
    # "synset" sans glose résolue — pas un cas spécial MWE).
    if not entry.get("definition_en") or not entry.get("lemmas_en"):
        return False, None, None
    best_candidates: list[str] | None = None
    best_score = 0.0
    for lemma in entry["lemmas_en"]:
        result = lex_bilingual.best_dbnary_match(entry["definition_en"], lemma, entry.get("pos"))
        if result is None:
            continue
        candidates, score = result
        if score > best_score:
            best_candidates, best_score = candidates, score
    if best_candidates is None:
        return False, None, None
    match = fr_norm.any_match(_candidates_of(entry), best_candidates)
    return match, best_candidates, best_score


def apertium_match(entry: dict) -> bool:
    for lemma in entry.get("lemmas_en", []):
        for candidate in _candidates_of(entry):
            if lex_bilingual.apertium_attests(lemma, candidate):
                return True
    return False


def wordfreq_ok(fr: str | None, threshold: float = WORDFREQ_MIN_ZIPF) -> bool:
    if not fr:
        return False
    words = fr_norm.readable_content_words(fr)
    if not words:
        return True  # rien à vérifier (ex. glose vide après nettoyage) -> ne bloque pas
    return all(zipf_frequency(w, "fr") > threshold for w in words)


def _resolve_synset(entry: dict):
    if entry.get("kind") != "synset":
        return None
    try:
        return nwn.synset(entry["key"])
    except (WordNetError, ValueError):
        return None


def sibling_synsets(synset) -> list:
    """Synsets partageant au moins un lemme avec `synset`, même grande
    catégorie (adjectif satellite "s" regroupé avec "a"), lui-même
    exclu — voisinage de polysémie à discriminer."""
    pos_group = {"a", "s"} if synset.pos() in ("a", "s") else {synset.pos()}
    seen = {synset}
    siblings = []
    for lemma in synset.lemmas():
        for other in nwn.synsets(lemma.name()):
            if other.pos() not in pos_group or other in seen:
                continue
            seen.add(other)
            siblings.append(other)
    return siblings


def discrimination_rank(fr_candidate: str, synset) -> int | None:
    """Rang (1 = meilleur) du synset CIBLE parmi lui-même et ses synsets
    frères, en classant leurs définitions anglaises par similarité LaBSE
    au candidat français seul.

    INFORMATIF SEULEMENT — PAS un test bloquant (voir compute_signals) :
    mesuré en développant ce module, ce test échoue à tort sur des
    candidats pourtant CORRECTS dès que le lemme cible est lui-même
    polysémique en anglais. Exemple mesuré : "angle" (angle.n.01,
    proposition correcte) classé 2e derrière slant.n.01 ("a biased way
    of looking at or presenting something") — un mot isolé, sans la
    phrase qui le désambiguïserait, n'apporte pas assez de signal à
    LaBSE face à des définitions WordNet complètes ; le taux de faux
    positifs mesuré sur un premier échantillon de 20 entrées était de
    4/4. Conservé dans l'audit comme indice pour un juge humain ou
    Stage C, jamais comme condition de promotion/rétrogradation."""
    siblings = sibling_synsets(synset)
    if not siblings:
        return None
    model = lex_bilingual.get_labse_model()
    targets = [synset] + siblings
    texts = [fr_candidate] + [s.definition() for s in targets]
    embeddings = model.encode(texts, normalize_embeddings=True)
    sims = embeddings[1:] @ embeddings[0]
    order = sims.argsort()[::-1]
    return int(list(order).index(0)) + 1


def polysemy_collision(decided_snapshot: dict[str, dict], entry: dict, candidate_fr: str | None) -> str | None:
    """Clé d'une AUTRE entrée du magasin, même lemme(s)+POS, déjà
    ARBITRÉE (DECIDED_STATUSES), dont le MOT-TÊTE concorde avec ce
    candidat — signal d'alerte (redite légitime entre sens proches, ou
    confusion) : n'empêche jamais seul une promotion, mais est journalisé
    et retiré du calcul des tests déterministes réussis pour rester
    prudent (voir la classe d'erreur china.n.02 "chine"->"porcelaine"
    corrigée à la main dans le dernier commit).

    `decided_snapshot` : instantané des entrées déjà arbitrées, figé
    AVANT le début de la passe (voir run()) — PAS le magasin vivant en
    cours de mutation. Comparer contre le magasin vivant rendrait le
    résultat dépendant de l'ORDRE de traitement au sein d'une même passe
    (une entrée promue tôt influencerait les collisions calculées pour
    les suivantes) et ferait diverger --dry-run (qui ne mute rien) du
    run réel — mesuré en développant ce module : 31 promotions en
    dry-run contre 30 en run réel avant ce figeage.

    Comparaison STRICTE (mot-tête identique), PAS `fr_norm.candidates_
    match` (trop permissif ici — tout recouvrement de mot de contenu
    suffit à `candidates_match`, ce qui convient pour corroborer un SEUL
    sens mais déclenche des faux positifs entre deux sens VRAIMENT
    différents qui partagent juste un mot : mesuré sur ce magasin,
    peppermint.n.03 "bonbon à la menthe" vs peppermint.n.01 "menthe
    poivrée" partagent "menthe" sans confusion de sens — un bonbon
    aromatisé porte trivialement le nom de la plante)."""
    if entry.get("kind") != "synset" or not candidate_fr:
        return None
    candidate_head = fr_norm.head_stem(candidate_fr)
    if candidate_head is None:
        return None
    lemmas = {l.casefold() for l in entry.get("lemmas_en", [])}
    pos = entry.get("pos")
    for other_key, other in decided_snapshot.items():
        if other_key == entry["key"] or other.get("kind") != "synset" or other.get("pos") != pos:
            continue
        if not other.get("fr") or not (lemmas & {l.casefold() for l in other.get("lemmas_en", [])}):
            continue
        if fr_norm.head_stem(other["fr"]) == candidate_head:
            return other_key
    return None


def compute_signals(decided_snapshot: dict[str, dict], entry: dict, occurrences_by_sense: dict[str, list[dict]]) -> dict:
    """Tous les signaux Stage A pour UNE entrée — ne modifie rien,
    utilisé à la fois pour la décision et pour la ligne d'audit.
    `decided_snapshot` : voir polysemy_collision (instantané figé, pas
    le magasin en cours de mutation). `occurrences_by_sense` : phrases du
    LIVRE COURANT (senses.load_occurrences_by_sense()), jamais lues
    depuis le magasin — voir sense_fr.format_occurrences_en."""
    res_match = resource_match(entry)
    db_match, db_candidates, db_score = dbnary_match(entry)
    ap_match = apertium_match(entry)
    wf_ok = wordfreq_ok(entry.get("fr"))

    synset = _resolve_synset(entry)
    disc_rank = discrimination_rank(entry["fr"], synset) if synset is not None and entry.get("fr") else None
    disc_ok = disc_rank is None or disc_rank == 1  # informatif seulement, voir discrimination_rank

    collision_key = polysemy_collision(decided_snapshot, entry, entry.get("fr"))

    # Trois signaux comptés, tous hors ligne et indépendants du modèle
    # (voir la docstring du module — ex-context_match retiré : il
    # comparait `fr` à un second appel du même modèle sur la même
    # question, devenu tautologique depuis la fusion dans
    # sense_fr_frontier.py).
    n_signals = sum([res_match, db_match, ap_match])
    # `disc_ok` volontairement EXCLU des tests bloquants (voir
    # discrimination_rank) : mesuré à 100% de faux positifs sur un
    # premier échantillon, il resterait dans `deterministic_ok` un
    # signal qui dégraderait le dispositif plutôt que de l'améliorer.
    deterministic_ok = wf_ok and not collision_key

    return {
        "key": entry["key"],
        "resource_match": res_match,
        # Informatifs seulement (déjà tranchés en amont par
        # sense_fr_frontier.py, avant même que l'entrée puisse arriver
        # ici en pending/auto_llm — voir la docstring du module) :
        "translation_type": entry.get("translation_type") or "",
        "sense_fit": entry.get("sense_fit") or "",
        # Mêmes phrases que celles transmises au juge Stage C
        # (run_stage_c) — journalisées ici pour que
        # pipeline_out/sense_fr_adjudication.csv soit auto-suffisant à
        # l'audit sans rouvrir data/sense_fr.jsonl.
        "contexte_en": sense_fr.format_occurrences_en(occurrences_by_sense.get(entry["key"], [])),
        "dbnary_fr": "; ".join(db_candidates or []),
        "dbnary_score": round(db_score, 3) if db_score is not None else "",
        "dbnary_match": db_match,
        "apertium_match": ap_match,
        "wordfreq_ok": wf_ok,
        "discrimination_rank": disc_rank if disc_rank is not None else "",
        "discrimination_ok": disc_ok,
        "polysemy_collision_with": collision_key or "",
        "n_corroborating_signals": n_signals,
        "deterministic_ok": deterministic_ok,
        "backtranslation_en": "", "backtranslation_ok": "",
        "judge_fr": "", "judge_confidence": "", "judge_reason": "",
        "decision": "", "decided_by": "",
    }


def decide_stage_a(entry: dict, signals: dict) -> tuple[str, str] | None:
    """(nouveau_statut, raison) si Stage A suffit à trancher, sinon None
    (laissé à Stage B/C)."""
    if entry["status"] not in ("pending", "auto_llm"):
        return None
    if not entry.get("fr"):
        return None

    if signals["n_corroborating_signals"] >= 2 and signals["deterministic_ok"]:
        matched = [s for s, ok in (
            ("resource", signals["resource_match"]),
            ("dbnary", signals["dbnary_match"]),
            ("apertium", signals["apertium_match"]),
        ) if ok]
        return "auto_corroborated", "stage_a:" + "+".join(matched)

    if entry["status"] == "auto_llm" and not signals["deterministic_ok"]:
        # N'énumère que les tests réellement BLOQUANTS (voir
        # deterministic_ok / discrimination_rank) : la discrimination
        # reste dans l'audit mais ne doit jamais apparaître ici, sous
        # peine de blâmer un signal qui n'a pesé sur aucune décision.
        reasons = [name for name, ok in (
            ("wordfreq", signals["wordfreq_ok"]),
            ("collision", not signals["polysemy_collision_with"]),
        ) if not ok]
        return "pending", "stage_a_echec:" + "+".join(reasons)

    return None


# ============================================================
# Stage B — rétro-traduction (nécessite une clé API)
# ============================================================


class _BacktranslationResult:
    __slots__ = ("key", "en")

    def __init__(self, key: str, en: str):
        self.key, self.en = key, en


def _backtranslate_batch(entries: list[dict], model: str) -> dict[str, str]:
    """Un appel par lot ; réutilise pipeline.sense_fr.backtranslation_matches
    pour la comparaison (voisinage WordNet, pas de chaînes) — voir sa
    docstring. Renvoie {key: traduction_anglaise_devinee}."""
    import hashlib
    import json as _json

    import litellm
    from pydantic import BaseModel as _BaseModel

    class _Guess(_BaseModel):
        key: str
        en: str

    class _BatchGuesses(_BaseModel):
        guesses: list[_Guess]

    system = (
        "Tu es traducteur français-anglais. Pour CHAQUE entrée, on te donne un "
        "mot ou une courte expression française, et le sens anglais précis "
        "qu'il est censé traduire (définition WordNet d'origine). Donne, pour "
        "chaque entrée, la traduction anglaise la plus naturelle de ce mot ou "
        "cette expression DANS CE SENS précis. Recopie la clé à l'identique."
    )
    lines = [
        f"- {e['key']} | français : \"{e.get('fr')}\" | sens visé : {e.get('definition_en') or '?'}"
        for e in entries
    ]
    user = "Entrées (" + str(len(entries)) + ") :\n" + "\n".join(lines)

    cache_key = _json.dumps({"model": model, "system": system, "user": user}, sort_keys=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    config.ensure_out_dir()
    cache_file = config.CACHE_DIR / f"backtranslate_{digest}.json"
    if cache_file.exists():
        parsed = _BatchGuesses.model_validate_json(cache_file.read_text(encoding="utf-8"))
    else:
        response = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=_BatchGuesses,
            reasoning_effort="low",
            max_tokens=8000,
        )
        content = response.choices[0].message.content
        parsed = _BatchGuesses.model_validate_json(content)
        cache_file.write_text(parsed.model_dump_json(), encoding="utf-8")
    return {g.key: g.en for g in parsed.guesses}


def run_stage_b(store: dict, targets: list[dict], model: str, batch_size: int = 40) -> dict[str, dict]:
    """Rétro-traduit `targets` (entrées avec exactement 1 signal Stage A,
    ou `auto_llm` ayant passé les tests déterministes) et renvoie
    {key: {"en": ..., "ok": bool}}. Nécessite une clé API LiteLLM."""
    results: dict[str, dict] = {}
    batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
    for batch in batches:
        guesses = _backtranslate_batch(batch, model)
        for entry in batch:
            en_guess = guesses.get(entry["key"])
            if not en_guess:
                continue
            synset = _resolve_synset(entry)
            ok = sense_fr.backtranslation_matches(en_guess, entry.get("lemmas_en", []), synset)
            results[entry["key"]] = {"en": en_guess, "ok": ok}
    return results


# ============================================================
# Stage C — juge sur dossier (nécessite une clé API)
# ============================================================


def run_stage_c(
    store: dict, targets: list[dict], audits: dict[str, dict], model: str,
    occurrences_by_sense: dict[str, list[dict]], batch_size: int = 20,
) -> dict[str, dict]:
    """Juge sur dossier, candidats MÉLANGÉS et NON ÉTIQUETÉS (pas de "le
    modèle a dit", pas de "omw a dit") — seul composant du dispositif
    autorisé à RÉÉCRIRE fr/fr_alt. Nécessite une clé API LiteLLM.

    Reçoit aussi les phrases réelles du LIVRE COURANT
    (`occurrences_by_sense`, senses.load_occurrences_by_sense() — jamais
    lues depuis le magasin, voir sense_fr.format_occurrences_en) : sans
    elles, le juge travaillerait à l'aveugle sur des candidats français
    hors sol alors qu'il est le dernier recours pour un sens encore en
    désaccord."""
    import hashlib
    import json as _json
    import random

    import litellm
    from pydantic import BaseModel as _BaseModel
    from typing import Literal as _Literal

    class _Verdict(_BaseModel):
        key: str
        fr: str
        fr_alt: list[str]
        confidence: _Literal["high", "medium", "low"]
        reason: str
        no_equivalent: bool = False

    class _BatchVerdicts(_BaseModel):
        verdicts: list[_Verdict]

    system = (
        "Tu es lexicographe bilingue anglais-français. Pour CHAQUE sens, on te "
        "donne sa définition WordNet, éventuellement une ou deux phrases RÉELLES "
        "d'un livre où le mot apparaît dans ce sens (à privilégier — elles "
        "montrent l'usage réel), et une liste de candidats de traduction "
        "française DÉJÀ PROPOSÉS par différentes sources (mélangés, sans "
        "indiquer leur origine). Choisis le meilleur candidat (ou réécris-en "
        "un meilleur si aucun ne convient vraiment), donne des variantes "
        "acceptables (fr_alt), ta confiance, et une justification courte. Si "
        "aucun équivalent français satisfaisant n'existe (expression trop "
        "culturellement située, etc.), mets no_equivalent à true."
    )

    rng = random.Random(42)  # ordre de présentation déterministe, pas d'indice de source
    system_prompt_lines = []
    for entry in targets:
        audit = audits.get(entry["key"], {})
        candidates = list({
            entry.get("fr"), *(entry.get("fr_alt") or []),
            *((audit.get("dbnary_fr") or "").split("; ") if audit.get("dbnary_fr") else []),
        } - {None, ""})
        rng.shuffle(candidates)
        occs_all = occurrences_by_sense.get(entry["key"]) or []
        # Mêmes phrases, dans le même ordre, que la colonne contexte_en de
        # l'audit (voir compute_signals ci-dessus et sense_fr.format_occurrences_en) :
        # pick_diverse_occurrences, pas les 2 premières dans l'ordre du
        # fichier — sinon le juge peut voir des phrases différentes de
        # celles que pipeline_out/sense_fr_adjudication.csv prétend lui
        # avoir montrées.
        occs = senses.pick_diverse_occurrences(occs_all, config.SENSE_FR_FRONTIER_MAX_OCCURRENCES) if occs_all else []
        sentences = " || ".join(
            f'"{o["context"]}" (mot cible : {o["target_surface"]})' for o in occs
        )
        system_prompt_lines.append(
            f"- {entry['key']} | {entry.get('pos') or 'mwe'} | {'/'.join(entry.get('lemmas_en', []))} | "
            f"définition : {entry.get('definition_en') or '?'} | phrase(s) : {sentences or '(aucune)'} | "
            f"candidats : {' ; '.join(candidates) or '(aucun)'}"
        )
    user = f"Sens à trancher ({len(targets)}) :\n" + "\n".join(system_prompt_lines)

    cache_key = _json.dumps({"model": model, "system": system, "user": user}, sort_keys=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    config.ensure_out_dir()
    cache_file = config.CACHE_DIR / f"judge_{digest}.json"
    if cache_file.exists():
        parsed = _BatchVerdicts.model_validate_json(cache_file.read_text(encoding="utf-8"))
    else:
        response = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=_BatchVerdicts,
            reasoning_effort="medium",
            max_tokens=16000,
        )
        content = response.choices[0].message.content
        parsed = _BatchVerdicts.model_validate_json(content)
        cache_file.write_text(parsed.model_dump_json(), encoding="utf-8")

    return {v.key: v.model_dump() for v in parsed.verdicts}


# ============================================================
# Orchestration
# ============================================================

AUDIT_FIELDS = [
    "key", "status_before", "kind", "lemmas_en", "pos", "suggested_fr", "suggested_fr_alt",
    "translation_type", "sense_fit", "contexte_en",
    "resource_match", "dbnary_fr", "dbnary_score", "dbnary_match",
    "apertium_match", "wordfreq_ok", "discrimination_rank", "discrimination_ok",
    "polysemy_collision_with", "n_corroborating_signals", "deterministic_ok",
    "backtranslation_en", "backtranslation_ok", "judge_fr", "judge_confidence", "judge_reason",
    "decision", "decided_by",
]


def run(
    limit: int | None = None,
    dry_run: bool = False,
    with_backtranslation: bool = False,
    with_judge: bool = False,
    backtranslation_model: str = config.SENSE_FR_FRONTIER_MODEL,
    judge_model: str = config.SENSE_FR_FRONTIER_MODEL,
) -> int:
    # Lot 3 (point E) — voir sense_fr_frontier.py::run().
    inventory.verify_consumer(config.SENSES_INVENTORY_HASH_PATH, "sense_fr_adjudicate")
    store = sense_fr.load_store()
    # Phrases du LIVRE COURANT, jamais lues depuis le magasin permanent —
    # voir sense_fr.format_occurrences_en et la docstring de compute_signals.
    occurrences_by_sense = senses.load_occurrences_by_sense()
    candidates = [e for e in store.values() if e["status"] in ("pending", "auto_llm")]
    if limit is not None:
        candidates = candidates[:limit]
    print(f"{len(candidates)} entrée(s) `pending`/`auto_llm` à arbitrer (Stage A).")

    # Instantané figé AVANT toute mutation (voir polysemy_collision) —
    # garantit que --dry-run et un run réel calculent EXACTEMENT les
    # mêmes collisions, indépendamment de l'ordre de traitement.
    decided_snapshot = {k: v for k, v in store.items() if v["status"] in DECIDED_STATUSES}

    audits: dict[str, dict] = {}
    n_promoted = n_downgraded = 0
    for entry in candidates:
        signals = compute_signals(decided_snapshot, entry, occurrences_by_sense)
        decision = decide_stage_a(entry, signals)
        audit_row = {**signals, "status_before": entry["status"]}
        if decision is not None:
            new_status, reason = decision
            audit_row["decision"] = new_status
            audit_row["decided_by"] = f"auto_adjudicate/{reason}"
            if not dry_run:
                if new_status == "auto_corroborated":
                    main, gloss = fr_norm.split_gloss(entry["fr"])
                    entry["fr"] = main
                    if gloss:
                        entry["fr_gloss"] = gloss
                entry["status"] = new_status
                entry["decided_at"] = date.today().isoformat()
                entry["decided_by"] = f"auto_adjudicate/{reason}"
            if new_status == "auto_corroborated":
                n_promoted += 1
            else:
                n_downgraded += 1
        audits[entry["key"]] = audit_row

    print(f"Stage A : {n_promoted} promue(s) en auto_corroborated, "
          f"{n_downgraded} auto_llm rétrogradée(s) en pending (tests déterministes échoués).")

    residual = [e for e in candidates if audits[e["key"]]["decision"] == ""]
    print(f"{len(residual)} entrée(s) encore sans décision après Stage A.")

    if with_backtranslation and residual:
        stage_b_targets = [
            e for e in residual
            if e.get("definition_en") and (
                audits[e["key"]]["n_corroborating_signals"] == 1
                or (e["status"] == "auto_llm" and audits[e["key"]]["deterministic_ok"])
            )
        ]
        print(f"Stage B (rétro-traduction) : {len(stage_b_targets)} candidat(s).")
        bt_results = run_stage_b(store, stage_b_targets, backtranslation_model)
        for entry in stage_b_targets:
            result = bt_results.get(entry["key"])
            if result is None:
                continue
            audits[entry["key"]]["backtranslation_en"] = result["en"]
            audits[entry["key"]]["backtranslation_ok"] = result["ok"]
            if not result["ok"]:
                continue
            new_status = "auto_corroborated"
            reason = "stage_b:backtranslation"
            audits[entry["key"]]["decision"] = new_status
            audits[entry["key"]]["decided_by"] = f"auto_adjudicate/{reason}"
            if not dry_run:
                main, gloss = fr_norm.split_gloss(entry["fr"])
                entry["fr"] = main
                if gloss:
                    entry["fr_gloss"] = gloss
                entry["status"] = new_status
                entry["decided_at"] = date.today().isoformat()
                entry["decided_by"] = f"auto_adjudicate/{reason}"
        residual = [e for e in residual if audits[e["key"]]["decision"] == ""]
        print(f"{len(residual)} entrée(s) encore sans décision après Stage B.")

    if with_judge and residual:
        print(f"Stage C (juge sur dossier) : {len(residual)} candidat(s).")
        verdicts = run_stage_c(store, residual, audits, judge_model, occurrences_by_sense)
        for entry in residual:
            verdict = verdicts.get(entry["key"])
            if verdict is None:
                continue
            audits[entry["key"]]["judge_fr"] = verdict["fr"]
            audits[entry["key"]]["judge_confidence"] = verdict["confidence"]
            audits[entry["key"]]["judge_reason"] = verdict["reason"]
            if verdict["no_equivalent"]:
                new_status, reason = "no_equivalent", "stage_c:no_equivalent"
            elif verdict["confidence"] == "high":
                new_status, reason = "auto_judged", "stage_c:judge_high_confidence"
            else:
                new_status, reason = "pending", "stage_c:judge_low_confidence"
            audits[entry["key"]]["decision"] = new_status
            audits[entry["key"]]["decided_by"] = f"auto_adjudicate/{reason}"
            if not dry_run:
                if new_status == "auto_judged":
                    main, gloss = fr_norm.split_gloss(verdict["fr"])
                    entry["fr"] = main
                    if gloss:
                        entry["fr_gloss"] = gloss
                    entry["fr_alt"] = verdict["fr_alt"]
                entry["status"] = new_status
                entry["decided_at"] = date.today().isoformat()
                entry["decided_by"] = f"auto_adjudicate/{reason}"

    config.ensure_out_dir()
    with ADJUDICATION_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in candidates:
            row = {
                **audits[entry["key"]],
                "kind": entry["kind"], "lemmas_en": "/".join(entry.get("lemmas_en", [])),
                "pos": entry.get("pos") or "", "suggested_fr": entry.get("fr") or "",
                "suggested_fr_alt": "; ".join(entry.get("fr_alt") or []),
            }
            writer.writerow(row)
    print(f"Audit détaillé -> {ADJUDICATION_CSV_PATH}")

    if dry_run:
        print("--dry-run : rien n'est écrit dans le magasin.")
        return 0

    sense_fr.write_store(store)
    n_pending = sense_fr.write_review_csv(store, occurrences_by_sense)
    print(f"{n_pending} entrée(s) encore `pending` -> {config.SENSE_FR_REVIEW_PATH}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-backtranslation", action="store_true",
                         help="Stage B : nécessite une clé API (LiteLLM).")
    parser.add_argument("--with-judge", action="store_true",
                         help="Stage C : nécessite une clé API (LiteLLM).")
    parser.add_argument("--backtranslation-model", default=config.SENSE_FR_FRONTIER_MODEL)
    parser.add_argument("--judge-model", default=config.SENSE_FR_FRONTIER_MODEL)
    args = parser.parse_args()
    raise SystemExit(run(
        limit=args.limit, dry_run=args.dry_run,
        with_backtranslation=args.with_backtranslation, with_judge=args.with_judge,
        backtranslation_model=args.backtranslation_model, judge_model=args.judge_model,
    ))
