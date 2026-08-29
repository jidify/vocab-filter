"""S6b — Traduction française de référence par MODÈLE FRONTIÈRE (via
LiteLLM), passe PRIMAIRE et CONTEXTUELLE, en remplacement du chemin ollama
local de pipeline/sense_fr.py pour la décision de statut.

Constat qui motive ce module (voir data/sense_fr.jsonl et le fil de
travail du 2026-08-25/26) : omw-fr (WOLF) et WoNeF sont construites
automatiquement et proposent régulièrement des contresens purs —
able.a.01 ("having the necessary means or skill") -> omw "comptable" ;
applaud.v.01 -> omw "guêtre" ; privacy.n.01 -> WoNeF "solitude" alors
que le sens réellement employé est "intimité". Le principe de
sense_fr.py — deux ressources automatiques bruitées qui se recoupent
valent acceptation automatique — a produit lui-même de tels contresens.
Ici, c'est l'INVERSE : le modèle frontière est la source PRIMAIRE, et
omw-fr/WoNeF ne servent plus qu'à repérer un DÉSACCORD à faire trancher
par un humain — elles ne peuvent plus, seules, faire remonter une
traduction que le modèle n'a pas proposée.

DEUXIÈME constat, mesuré après la mise en service de ce qui précède
(2026-08-26) : le modèle qui décidait ne voyait jamais le livre — glose
WordNet seule, sans la ou les phrases où le mot apparaît réellement. Une
passe séparée, contextuelle (l'ancien pipeline/sense_fr_context.py), avait
été ajoutée pour relire avec les phrases réelles, mais n'avait le droit
d'écrire ni `fr` ni `status`, et arrivait après coup sur un échantillon
seulement. Mesuré sur le groupe témoin des `auto_strong` déjà décidés SANS
contexte : 38/100 divergeaient de la lecture contextuelle, dont de vrais
contresens (alien.s.02 "exotique" alors que le livre dit "extraterrestre").
Ce module fusionne donc les deux passes : la lecture contextuelle DEVIENT
la décision primaire, en un seul appel par sens, avec accès :
- aux phrases RÉELLES du livre (quand `pipeline_out/senses.jsonl` en
  fournit — sinon repli sur la glose WordNet seule, comme avant) ;
- à des candidats de traduction MÉLANGÉS ET NON ÉTIQUETÉS provenant de
  omw-fr, WoNeF, DBnary (glose humaine, voir pipeline/lex_bilingual.py) et
  Apertium (dictionnaire humain, lemme seul) — pour que le modèle puisse
  s'appuyer sur ces ressources sans qu'aucune ne lui soit présentée comme
  fiable par défaut.

Statuts produits (en plus de ceux de sense_fr.py, inchangés) :
- `auto_strong`  : le modèle propose une traduction fidèle (equivalence_
  directe, sense_fit ok, confiance non basse), ET au moins une des deux
  ressources (omw-fr, WoNeF) la corrobore (même racine) ;
- `auto_llm`     : idem, mais aucune des deux ressources ne couvre le sens
  (pas de synset côté omw-fr/WoNeF, ou MWE — jamais de ressource par
  construction) ; seul le modèle a un avis ;
- `pending`      : sense_fit "mismatch"/"doubtful" (le modèle signale que
  le sense_id imposé par S5 ne colle pas à l'usage — voir
  write_sense_id_suspects_csv), OU translation_type != equivalence_directe
  (une reformulation/explicitation signale souvent, elle aussi, un
  sense_id douteux), OU confiance auto-déclarée "low", OU désaccord entre
  le modèle et une ressource qui a un avis. Toujours relu par un humain,
  quel que soit ce que dit le modèle seul.

Le magasin `data/sense_fr.jsonl` reste l'unique référentiel permanent,
indexé par sense_id, réutilisé d'un livre à l'autre — ce module n'introduit
aucune nouvelle structure de fichier, il enrichit les entrées existantes en
place. `pipeline_out/cache/` reste un pur cache de réponses brutes (clé =
hash du prompt exact, qui inclut désormais les phrases et les candidats),
régénérable, distinct du magasin.

Usage :
    uv run python -m pipeline.sense_fr_frontier
    uv run python -m pipeline.sense_fr_frontier --limit 50 --dry-run
    uv run python -m pipeline.sense_fr_frontier --model anthropic/claude-haiku-4-5
"""

from __future__ import annotations

import csv
import random
from datetime import date
from pathlib import Path
from typing import Literal

import litellm
from nltk.corpus import wordnet as nwn
from nltk.corpus.reader.wordnet import WordNetError
from pydantic import BaseModel

from pipeline import config, inventory, lex_bilingual, llm_client, sense_fr, senses
from pipeline.llm_tasks import effective_batch_size, task_config, use_batch_prompt

# ============================================================
# Schéma de sortie structurée
# ============================================================


class SenseTranslation(BaseModel):
    sense_id: str
    fr: list[str]  # 1 à 3 propositions, triées par fréquence d'usage RÉELLE
    translation_type: Literal["equivalence_directe", "reformulation", "explicitation"]
    sense_fit: Literal["ok", "doubtful", "mismatch"]
    sense_fit_note: str
    source: Literal["choisi", "reecrit"]
    confidence: Literal["high", "medium", "low"]


class BatchTranslations(BaseModel):
    translations: list[SenseTranslation]


class UnitTranslation(SenseTranslation):
    """Réponse d'un prompt unitaire (un sens, objet scalaire)."""


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = (
    "Tu es lexicographe bilingue anglais-français, spécialisé en traduction "
    "littéraire. On te donne une liste de SENS PRÉCIS de mots ou d'expressions "
    "anglaises, chacun identifié par un sense_id WordNet (ou une clé mwe:... "
    "pour les expressions figées/verbes à particule sans entrée WordNet), avec "
    "ses lemmes anglais, sa catégorie grammaticale et sa définition. Pour "
    "certains sens, tu recevras EN PLUS une ou plusieurs phrases RÉELLES d'un "
    "livre où le mot cible apparaît (mot cible indiqué séparément) — utilise-les "
    "en priorité : elles montrent l'usage réel, la glose seule ne le montre pas. "
    "Pour certains sens, tu recevras aussi une liste de candidats de traduction "
    "MÉLANGÉS ET NON ÉTIQUETÉS provenant de plusieurs sources automatiques et "
    "humaines — aucune n'est fiable à 100%, ne fais confiance à aucune par "
    "défaut, vérifie chaque candidat contre la définition et les phrases.\n\n"
    "Pour CHAQUE sense_id, réponds avec :\n"
    "- fr : 1 à 3 traductions françaises de CE SENS précis (jamais d'un autre "
    "sens du même mot anglais), TRIÉES PAR FRÉQUENCE D'USAGE RÉELLE dans ce "
    "sens — la plus courante d'abord, la plus rare/marginale en dernier (pas "
    "seulement \"la plus naturelle\") ;\n"
    "- translation_type : \"equivalence_directe\" si fr[0] est un vrai "
    "équivalent lexical du mot, substituable dans une traduction soignée ; "
    "\"reformulation\" si les phrases fournies montrent que le passage "
    "condense, déplace ou explicite l'information au point qu'aucun mot ou "
    "groupe isolé ne correspond vraiment au mot anglais ; \"explicitation\" si "
    "la traduction ajoute une précision absente de l'anglais mais nécessaire en "
    "français. Sans phrase fournie pour ce sens, réponds toujours "
    "\"equivalence_directe\" (rien à évaluer sur un usage que tu ne vois pas) ;\n"
    "- sense_fit : le sens WordNet imposé (sa définition) correspond-il "
    "vraiment à l'usage montré par les phrases fournies ? \"ok\" si oui, "
    "\"doubtful\" si ça te semble limite, \"mismatch\" si les phrases montrent "
    "clairement un AUTRE sens que celui décrit par la définition — dans ce "
    "cas NE CHOISIS PAS un autre sens à sa place, traduis quand même du mieux "
    "possible et contente-toi de signaler le problème. Sans phrase fournie pour "
    "ce sens, réponds toujours \"ok\" (rien à vérifier) ;\n"
    "- sense_fit_note : une phrase courte justifiant sense_fit (chaîne vide si "
    "\"ok\" et évident) ;\n"
    "- source : \"choisi\" si fr[0] reprend un des candidats fournis pour ce "
    "sens (même avec une légère variante orthographique ou grammaticale), "
    "\"reecrit\" si tu proposes une traduction absente de la liste fournie, ou "
    "si aucune liste n'était fournie pour ce sens ;\n"
    "- confidence : \"low\" si la définition est ambiguë, si le sense_id est "
    "trop générique/rare pour être sûr, ou si aucune traduction ne te semble "
    "vraiment naturelle en français.\n\n"
    "Renvoie EXACTEMENT un objet par sense_id reçu, avec le sense_id recopié "
    "à l'identique (ne jamais indexer par le mot anglais : plusieurs sens "
    "différents peuvent partager le même mot)."
)

ITEM_HEADER = "- {sense_id} | {pos_label} | {lemmas} | {definition}"
ITEM_OCCURRENCE = '    contexte : "{context}" || mot cible dans ce contexte : "{target_surface}"'
ITEM_CANDIDATES = "    candidats connus (non fiables, mélangés, ordre sans signification) : {candidates}"

POS_LABELS = {"n": "nom", "v": "verbe", "a": "adjectif", "s": "adjectif", "r": "adverbe", "mwe": "expression"}


def _format_item(target: dict, occurrences: list[dict], candidates: list[str]) -> str:
    lines = [ITEM_HEADER.format(
        sense_id=target["key"],
        pos_label=POS_LABELS.get(target.get("pos") or "mwe", target.get("pos") or "?"),
        lemmas="/".join(target["lemmas_en"]),
        definition=target.get("definition_en") or "?",
    )]
    for occ in occurrences:
        lines.append(ITEM_OCCURRENCE.format(context=occ["context"], target_surface=occ["target_surface"]))
    if candidates:
        lines.append(ITEM_CANDIDATES.format(candidates=" ; ".join(candidates)))
    return "\n".join(lines)


def build_unit_user_prompt(item: tuple[dict, list[dict], list[str]]) -> str:
    target, occurrences, candidates = item
    return "Sens à traduire (1) :\n" + _format_item(target, occurrences, candidates) + (
        "\nRéponds avec un objet JSON unique pour ce sense_id (pas de liste)."
    )


def build_user_prompt(batch: list[tuple[dict, list[dict], list[str]]]) -> str:
    items = "\n".join(_format_item(target, occs, cands) for target, occs, cands in batch)
    return f"Sens à traduire ({len(batch)}) :\n{items}"


# ============================================================
# Collecte + résolution des synsets (pos/définition/lemmes WordNet)
# ============================================================


def collect_frontier_targets() -> tuple[list[dict], list[dict]]:
    """Reprend sense_fr.collect_targets() (fusion par sense_id déjà
    correcte, gestion des MWE déjà correcte) et résout, pour chaque
    cible "synset", le synset WordNet réel — pos, définition, lemmes
    anglais canoniques du synset (pas seulement ceux vus dans CE livre).

    Renvoie (targets_traduisibles, targets_sense_id_non_resolu) — les
    seconds sont mis en pending sans jamais appeler le modèle (aucune
    définition fiable à lui donner)."""
    raw_targets = sense_fr.collect_targets()

    resolved: list[dict] = []
    unresolved: list[dict] = []
    for target in raw_targets.values():
        if target["kind"] == "mwe":
            resolved.append(target)
            continue
        try:
            synset = nwn.synset(target["key"])
        except (WordNetError, ValueError):
            unresolved.append(target)
            continue
        offset = f"{synset.offset():08d}"
        pos = synset.pos()
        english_lemmas = [l.name().replace("_", " ") for l in synset.lemmas()]
        resolved.append({
            **target,
            "pos": pos,
            "definition_en": synset.definition(),
            "lemmas_en": english_lemmas,
            "_synset": synset,
            "_offset": offset,
        })
    return resolved, unresolved


# ============================================================
# Candidats de traduction (omw-fr, WoNeF, DBnary, Apertium) — mélangés
# et non étiquetés dans le prompt, pour que le modèle ne fasse confiance
# à aucune source par défaut.
# ============================================================


def _resources_for(target: dict) -> tuple[list[str], list[str]]:
    """omw-fr / WoNeF ne peuvent jamais rien proposer pour une MWE (pas
    de synset, par construction) — cf. sense_fr.classify_mwe_key."""
    if target["kind"] == "mwe":
        return [], []
    omw = sense_fr.fr_candidates_omw(target["_offset"], target["pos"], target["lemmas_en"])
    wonef = sense_fr.fr_candidates_wonef(target["_offset"], target["pos"], target["lemmas_en"])
    return omw, wonef


def _dbnary_candidates(target: dict) -> list[str]:
    # Lot 4 (point 22) : plus de garde `kind == "mwe"` ici — DBnary
    # contient déjà 136 entrées multi-mots (`wake up__Verb__1`…), la
    # seule chose qui manquait était cet arrêt anticipé. Contrairement à
    # `_resources_for` (omw-fr/WoNeF ci-dessus), best_dbnary_match ne
    # dépend pas d'un synset résolu : `definition_en` (glose idioms.yml,
    # peut être None) et `lemmas_en` (= [canonical_form]) suffisent.
    best_candidates: list[str] | None = None
    best_score = 0.0
    for lemma in target["lemmas_en"]:
        result = lex_bilingual.best_dbnary_match(target.get("definition_en") or "", lemma, target.get("pos"))
        if result is None:
            continue
        candidates, score = result
        if score > best_score:
            best_candidates, best_score = candidates, score
    return best_candidates or []


def _apertium_candidates(target: dict) -> list[str]:
    apertium = lex_bilingual.load_extract().get("apertium", {})
    out: list[str] = []
    for lemma in target.get("lemmas_en", []):
        out.extend(apertium.get(lemma.casefold(), []))
    return out


def collect_candidates(target: dict, rng: random.Random) -> list[str]:
    omw, wonef = _resources_for(target)
    pool = list(dict.fromkeys(omw + wonef + _dbnary_candidates(target) + _apertium_candidates(target)))
    rng.shuffle(pool)
    return pool


# ============================================================
# Cache disque — clé et préfixe INCHANGÉS depuis avant l'unification
# (Lot U3, fix_pipeline/multi_models/report_multi_models.md §4bis) : ce
# cache correspond à des appels OpenAI déjà payés, il ne doit pas être
# invalidé par le passage au client LLM unique. Seule la mécanique de
# hash/lecture/écriture déménage dans pipeline/llm_client.py.
# ============================================================


def _cache_key_fields(model: str, system: str, user: str, *, mode_batch: bool, batch_size: int | None) -> dict:
    return {"task_id": "S6-translate-frontier", "model": model,
            "mode_batch": mode_batch, "batch_size": batch_size or (1 if not mode_batch else 0),
            "system": system, "user": user}


def _cache_path(model: str, system: str, user: str, *, mode_batch: bool = True, batch_size: int | None = None) -> Path:
    return llm_client.cache_path_for(
        _cache_key_fields(model, system, user, mode_batch=mode_batch, batch_size=batch_size),
        prefix="frontier_",
    )


def _translate_batches(
    batches: list[list[tuple[dict, list[dict], list[str]]]], model: str,
    *, mode_batch: bool = True, batch_size: int | None = None,
) -> tuple[list[dict[str, SenseTranslation]], float]:
    """Traduit chaque lot (avec cache disque par lot, via
    pipeline.llm_client). Renvoie la liste des {sense_id: SenseTranslation}
    par lot (même ordre que `batches`) et le coût total en USD des SEULS
    appels réellement effectués (un lot servi par le cache ne coûte rien)."""
    items: list[llm_client.BatchItem] = []
    for batch in batches:
        if not mode_batch and len(batch) != 1:
            raise ValueError(
                f"S6-translate-frontier: mode unitaire attend exactement 1 item par lot, "
                f"reçu {len(batch)}"
            )
        user_prompt = build_user_prompt(batch) if mode_batch else build_unit_user_prompt(batch[0])
        items.append(llm_client.BatchItem(
            system=SYSTEM_PROMPT, user=user_prompt,
            cache_key_fields=_cache_key_fields(model, SYSTEM_PROMPT, user_prompt,
                                               mode_batch=mode_batch, batch_size=batch_size),
            cache_prefix="frontier_",
        ))

    def _on_error(i, _item, exc):
        print(f"  lot {i}: échec ({exc!r}), {len(batches[i])} sens laissés de côté.")

    responses, total_cost = llm_client.call_batch_completion(
        items, model=model,
        response_model=BatchTranslations if mode_batch else UnitTranslation,
        reasoning_effort="low",  # recherche lexicale par item, pas du raisonnement long
        max_tokens=16000,
        max_workers=config.SENSE_FR_FRONTIER_MAX_WORKERS,
        on_error=_on_error,
    )
    results = [
        ({t.sense_id: t for t in parsed.translations} if mode_batch else {parsed.sense_id: parsed})
        if parsed is not None else {}
        for parsed in responses
    ]
    return results, total_cost


# ============================================================
# Fusion dans le magasin — triage par sense_fit / fidélité / désaccord
# ============================================================


SUSPECT_AGREEMENTS = {"sense_id_suspect", "sense_id_douteux"}

# Statuts que cette passe ne doit JAMAIS reproduire ni écraser : `validated`
# est une décision humaine (sense_fr_commit.py), `auto_joint` une décision
# conjointe POS/sense_id à inventaire ouvert (sense_fr_reassign.py) que cette
# passe — qui recopie le sense_id imposé à l'identique, voir SYSTEM_PROMPT —
# ne peut par construction pas reproduire. Sans cette garde, `run()` réévalue
# INCONDITIONNELLEMENT toute clé pertinente pour le livre courant (voir la
# boucle plus bas), y compris celles déjà décidées autrement ; verify_fr_lock
# ne fait que détecter ce genre d'écrasement après coup, ne le prévient pas.
PROTECTED_STATUSES = {"validated", "auto_joint"}


def is_protected(existing: dict | None) -> bool:
    return existing is not None and existing.get("status") in PROTECTED_STATUSES


def build_entry(target: dict, translation: SenseTranslation | None, *, model: str | None = None) -> dict:
    """N'écrit JAMAIS les phrases du livre dans l'entrée renvoyée : cette
    entrée est destinée à data/sense_fr.jsonl, le magasin PERMANENT
    réutilisé d'un livre à l'autre (voir pipeline/sense_fr.py) — une
    phrase brute y serait propre au livre du run courant et invalide pour
    tout autre livre qui réutiliserait ce sense_id. Les phrases utilisées
    pour cette décision restent uniquement dans le prompt envoyé au
    modèle (non persisté hors du cache) ; pour les retrouver après coup,
    voir senses.load_occurrences_by_sense() sur le livre concerné — c'est
    ce que font write_sense_id_suspects_csv ci-dessous et
    pipeline/score.py/pipeline/sense_fr_adjudicate.py, à chaque run, sur
    le livre courant."""
    key = target["key"]
    omw, wonef = _resources_for(target)
    entry_base = {
        "key": key, "kind": target["kind"], "lemmas_en": target["lemmas_en"],
        "pos": target.get("pos"), "definition_en": target.get("definition_en"),
        "occurrences": target.get("occurrences", 0),
    }

    if translation is None or not translation.fr:
        entry_base.update({
            "fr": None, "fr_alt": [], "status": "pending",
            "agreement": "frontier_sans_reponse",
            "translation_type": None, "sense_fit": None, "sense_fit_note": "",
            "source": None,
            "evidence": {"omw_fr": omw, "wonef": wonef, "frontier_model": None, "frontier_confidence": None},
            "decided_at": None, "decided_by": None, "note": "",
        })
        return entry_base

    fr, fr_alt = translation.fr[0], translation.fr[1:]
    resource_stems = {senses.fr_stem(c) for c in omw + wonef}
    proposed_stems = {senses.fr_stem(c) for c in translation.fr}
    overlap = bool(resource_stems & proposed_stems)

    # Ordre des portes : d'abord la fidélité du SENS lui-même (sense_fit,
    # translation_type — un sense_id douteux ou une reformulation ne
    # doivent jamais être verrouillés automatiquement, quelle que soit la
    # confiance du modèle ou l'accord des ressources), puis la confiance
    # déclarée, puis enfin la corroboration par ressource — inchangée par
    # rapport à la version précédente de ce module. Porte partagée avec
    # sense_fr_reassign.py et sense_fr_adjudicate.py (plan §6, S6-1) :
    # voir sense_fr.blocks_auto_lock.
    block_reason = sense_fr.blocks_auto_lock(translation.sense_fit, translation.translation_type)
    if block_reason:
        status, agreement = "pending", block_reason
    elif translation.confidence == "low":
        status, agreement = "pending", "frontier_confiance_faible"
    elif not omw and not wonef:
        status, agreement = "auto_llm", "frontier_sans_ressource"
    elif overlap:
        status, agreement = "auto_strong", "frontier_concordant"
    else:
        status, agreement = "pending", "frontier_desaccord"

    entry_base.update({
        "fr": fr, "fr_alt": fr_alt, "status": status, "agreement": agreement,
        "translation_type": translation.translation_type,
        "sense_fit": translation.sense_fit,
        "sense_fit_note": translation.sense_fit_note,
        "source": translation.source,
        "evidence": {
            "omw_fr": omw, "wonef": wonef,
            "frontier_model": model or config.SENSE_FR_FRONTIER_MODEL,
            "frontier_fr": translation.fr,
            "frontier_confidence": translation.confidence,
        },
        "decided_at": date.today().isoformat() if status in ("auto_strong", "auto_llm") else None,
        "decided_by": "auto_frontier" if status in ("auto_strong", "auto_llm") else None,
        "note": "",
    })
    return entry_base


# ============================================================
# sense_id suspects — boucle de retour vers S5 (voir la docstring)
# ============================================================

SUSPECT_FIELDS = ["key", "lemmas_en", "pos", "sense_fit", "definition_en", "sense_fit_note", "contexte_en"]


def write_sense_id_suspects_csv(store: dict[str, dict], occurrences_by_sense: dict[str, list[dict]]) -> int:
    """`occurrences_by_sense` : phrases du LIVRE COURANT (senses.load_occurrences_by_sense()),
    jamais lues depuis le magasin — voir la docstring de build_entry."""
    suspects = [e for e in store.values() if e.get("agreement") in SUSPECT_AGREEMENTS]
    suspects.sort(key=lambda e: e["key"])
    config.ensure_out_dir()
    with config.SENSE_ID_SUSPECTS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUSPECT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for e in suspects:
            writer.writerow({
                "key": e["key"], "lemmas_en": "/".join(e.get("lemmas_en", [])),
                "pos": e.get("pos") or "", "sense_fit": e.get("sense_fit") or "",
                "definition_en": e.get("definition_en") or "",
                "sense_fit_note": e.get("sense_fit_note") or "",
                "contexte_en": sense_fr.format_occurrences_en(occurrences_by_sense.get(e["key"], [])),
            })
    return len(suspects)


# ============================================================
# Orchestration
# ============================================================


def run(model: str | None = None, limit: int | None = None, dry_run: bool = False) -> int:
    task = task_config("S6-translate-frontier")
    model = model or task.model
    config.require_frontier_model(model, "S6-translate-frontier")
    # Lot 3 (point E) : senses.jsonl doit avoir été calculé contre
    # l'inventaire COURANT (pipeline/inventory.py) — sinon les occurrences
    # présentées en contexte au modèle frontière pourraient ne plus
    # correspondre à ce que select.py a réellement retenu.
    inventory.verify_consumer(config.SENSES_INVENTORY_HASH_PATH, "sense_fr_frontier")
    resolved, unresolved = collect_frontier_targets()

    # Lot 6 (Partie 3, point 31) : le filtrage des cibles déjà protégées
    # (`validated`/`auto_joint`) a lieu ICI, AVANT toute construction
    # d'items/lots — pas après coup (l'ancien point d'insertion, dans la
    # boucle qui consomme `translations_by_batch` plus bas), qui laissait
    # ces cibles alourdir les lots ENVOYÉS au modèle (candidats collectés,
    # occurrences choisies, texte de prompt construit) pour un résultat
    # jeté à la réception. C'était l'obstacle principal à toute reprise
    # gratuite d'un run à l'autre : une clé déjà validée à la main
    # continuait de payer sa part de chaque lot où elle tombait.
    store = sense_fr.load_store()
    n_protected = sum(1 for t in resolved if is_protected(store.get(t["key"])))
    resolved = [t for t in resolved if not is_protected(store.get(t["key"]))]

    if limit is not None:
        resolved = resolved[:limit]
    print(f"{len(resolved)} cible(s) à traduire ({len(unresolved)} sense_id non résolu(s) — "
          f"mis en pending sans appel au modèle ; {n_protected} déjà validée(s)/auto_joint "
          f"laissée(s) intacte(s), jamais mises en lot), modèle={model}.")

    occurrences_by_sense = senses.load_occurrences_by_sense()
    rng = random.Random(42)  # ordre de présentation des candidats déterministe

    items: list[tuple[dict, list[dict], list[str]]] = []
    n_no_occurrence = 0
    for target in resolved:
        occs_all = occurrences_by_sense.get(target["key"]) or []
        if occs_all:
            occs = senses.pick_diverse_occurrences(occs_all, config.SENSE_FR_FRONTIER_MAX_OCCURRENCES)
        else:
            occs = []
            n_no_occurrence += 1
        candidates = collect_candidates(target, rng)
        items.append((target, occs, candidates))

    if n_no_occurrence:
        print(f"  {n_no_occurrence} cible(s) sans occurrence exploitable dans "
              f"{config.SENSES_PATH}/{config.SELECTED_MWE_PATH} — traduites glose seule.")

    # Lots découpés PAR `kind` (synset / mwe), jamais mélangés : le cache
    # disque est indexé sur le texte exact du prompt d'un lot entier
    # (_cache_path ci-dessus). Sans ce découpage, ajouter du contexte aux
    # MWE changerait le texte de TOUT lot qui mélange les deux (il y en a
    # un seul, à la frontière synset/mwe de collect_targets — les MWE sont
    # toujours en fin de liste) et forcerait à rejouer ce lot entier,
    # synsets déjà décidés inclus. En les isolant, tout lot 100% synset de
    # taille pleine (batch_size) reproduit un lot déjà vu -> cache disque,
    # gratuit. EXCEPTION inévitable : le lot synset RESTANT (le nombre de
    # synsets n'est pas un multiple de batch_size) n'a jamais existé seul
    # dans le cache — il était fondu dans l'ancien lot mixte — donc lui
    # sera rejoué au modèle comme les lots MWE. verify_fr_lock (voir
    # data/sense_fr.lock.json) reste le filet de sécurité si ce lot
    # change la traduction verrouillée d'un synset qu'il contient.
    batch_size = effective_batch_size(task)
    items_by_kind: dict[str, list[tuple[dict, list[dict], list[str]]]] = {}
    for item in items:
        items_by_kind.setdefault(item[0]["kind"], []).append(item)
    batches: list[list[tuple[dict, list[dict], list[str]]]] = []
    for kind_items in items_by_kind.values():
        batches.extend(kind_items[i:i + batch_size] for i in range(0, len(kind_items), batch_size))
    mode_batch = use_batch_prompt(task, batch_size)
    translations_by_batch, cost = _translate_batches(batches, model, mode_batch=mode_batch, batch_size=batch_size)

    # `store` a déjà été chargé plus haut pour le filtrage protégé — pas de
    # second appel à `sense_fr.load_store()` ici (Lot 6). `resolved` ne
    # contient plus aucune cible protégée à ce point, donc plus besoin de
    # revérifier `is_protected` dans cette boucle.
    n_by_status: dict[str, int] = {}
    n_suspect = 0
    for batch, translations in zip(batches, translations_by_batch):
        for target, _occs, _candidates in batch:
            translation = translations.get(target["key"])
            entry = build_entry(target, translation, model=model)
            store[entry["key"]] = entry
            n_by_status[entry["status"]] = n_by_status.get(entry["status"], 0) + 1
            if entry.get("agreement") in SUSPECT_AGREEMENTS:
                n_suspect += 1

    for target in unresolved:
        store[target["key"]] = {
            "key": target["key"], "kind": "synset", "lemmas_en": target["lemmas_en"],
            "pos": None, "definition_en": None, "occurrences": target.get("occurrences", 0),
            "fr": None, "fr_alt": [], "status": "pending", "agreement": "sense_id_non_resolu",
            "translation_type": None, "sense_fit": None, "sense_fit_note": "",
            "source": None,
            "evidence": {}, "decided_at": None, "decided_by": None, "note": "",
        }
        n_by_status["pending"] = n_by_status.get("pending", 0) + 1

    print(f"Coût constaté (appels non-cachés uniquement) : ${cost:.4f}")
    print("Ventilation des statuts sur ce run :", n_by_status)
    if n_suspect:
        print(f"  dont {n_suspect} sense_id signalé(s) suspect/douteux par le modèle "
              f"(voir {config.SENSE_ID_SUSPECTS_PATH}).")

    if dry_run:
        print("--dry-run : rien n'est écrit dans le magasin ni dans les CSV.")
        return 0

    sense_fr.write_store(store)
    n_pending = sense_fr.write_review_csv(store, occurrences_by_sense)
    n_suspects_written = write_sense_id_suspects_csv(store, occurrences_by_sense)
    n_validated = sum(1 for e in store.values() if e["status"] == "validated")
    n_auto_strong = sum(1 for e in store.values() if e["status"] == "auto_strong")
    n_auto_llm = sum(1 for e in store.values() if e["status"] == "auto_llm")
    print(f"Magasin : {len(store)} entrées ({n_validated} validées, {n_auto_strong} auto_strong, "
          f"{n_auto_llm} auto_llm, {n_pending} en attente -> {config.SENSE_FR_REVIEW_PATH}).")
    if n_suspects_written:
        print(f"{n_suspects_written} sense_id suspect(s)/douteux -> {config.SENSE_ID_SUSPECTS_PATH}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="Modèle LiteLLM — par défaut, celui résolu par task_config('S6-translate-frontier') "
             "(voir pipeline/llm_tasks.py). require_frontier_model refuse tout autre modèle : "
             "pour en utiliser un autre, poser VOCAB_LLM_S6_TRANSLATE_FRONTIER=provider/nom;... "
             "plutôt que de passer --model seul (empêche une frappe de travers de déclencher "
             "silencieusement un modèle plus coûteux).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limite le nombre de cibles traitées (les N premières) — pour un run d'essai.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Appelle le modèle et affiche le résultat/coût, mais n'écrit rien sur disque.",
    )
    args = parser.parse_args()
    raise SystemExit(run(model=args.model, limit=args.limit, dry_run=args.dry_run))
