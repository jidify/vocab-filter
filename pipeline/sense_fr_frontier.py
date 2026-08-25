"""S6b — Traduction française de référence par MODÈLE FRONTIÈRE (via
LiteLLM), en remplacement du chemin ollama local de pipeline/sense_fr.py
pour la décision de statut.

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

Statuts produits (en plus de ceux de sense_fr.py, inchangés) :
- `auto_strong`  : le modèle propose une traduction, ET au moins une
  des deux ressources (omw-fr, WoNeF) la corrobore (même racine) ;
- `auto_llm`     : aucune des deux ressources ne couvre le sens (pas de
  synset côté omw-fr/WoNeF, ou MWE — jamais de ressource par
  construction) ; seul le modèle a un avis. Nouveau statut, exporté au
  même titre que `auto_strong`/`validated` (voir score.py, verify_fr_lock.py) ;
- `pending`      : désaccord entre le modèle et une ressource qui a un
  avis, OU confiance auto-déclarée `low` du modèle. Toujours relu par
  un humain, quel que soit ce que dit le modèle seul — une confiance
  basse ou un vrai désaccord avec une ressource qui a un avis reste un
  signal qu'il ne faut pas trancher automatiquement.

Le magasin `data/sense_fr.jsonl` reste l'unique référentiel permanent,
indexé par sense_id, réutilisé d'un livre à l'autre — ce module
n'introduit AUCUNE nouvelle structure, il enrichit les entrées
existantes en place. `pipeline_out/cache/` reste un pur cache de
réponses brutes (clé = hash du prompt exact), régénérable, distinct du
magasin.

Usage :
    uv run python -m pipeline.sense_fr_frontier
    uv run python -m pipeline.sense_fr_frontier --limit 50 --dry-run
    uv run python -m pipeline.sense_fr_frontier --model openai/gpt-5
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import litellm
from nltk.corpus import wordnet as nwn
from nltk.corpus.reader.wordnet import WordNetError
from pydantic import BaseModel

from pipeline import config, sense_fr, senses

# ============================================================
# Schéma de sortie structurée
# ============================================================


class SenseTranslation(BaseModel):
    sense_id: str
    fr: list[str]  # 1 à 3 propositions, de la plus naturelle à la plus marginale
    confidence: Literal["high", "medium", "low"]


class BatchTranslations(BaseModel):
    translations: list[SenseTranslation]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = (
    "Tu es lexicographe bilingue anglais-français, spécialisé en traduction "
    "littéraire. On te donne une liste de SENS PRÉCIS de mots ou d'expressions "
    "anglaises, chacun identifié par un sense_id WordNet (ou une clé mwe:... "
    "pour les expressions figées/verbes à particule sans entrée WordNet), avec "
    "ses lemmes anglais, sa catégorie grammaticale et sa définition — SANS "
    "phrase d'exemple tirée d'un livre particulier.\n\n"
    "Pour CHAQUE sense_id, donne 1 à 3 traductions françaises de CE SENS précis "
    "(pas des autres sens possibles du même mot anglais), de la plus naturelle "
    "à la plus marginale, telles qu'on les trouverait dans un bon dictionnaire "
    "bilingue ou une traduction littéraire soignée. Deux sense_id différents "
    "pour un même mot anglais polysémique (p.ex. \"breath\" au sens du souffle "
    "vs. au sens d'un répit) DOIVENT recevoir des traductions différentes si "
    "leurs définitions diffèrent.\n\n"
    "Indique aussi ta confiance (\"high\"/\"medium\"/\"low\") : \"low\" si la "
    "définition est ambiguë, si le sense_id est trop générique/rare pour être "
    "sûr, ou si aucune traduction ne te semble vraiment naturelle en français.\n\n"
    "Renvoie EXACTEMENT un objet par sense_id reçu, avec le sense_id recopié "
    "à l'identique (ne jamais indexer par le mot anglais : plusieurs sens "
    "différents peuvent partager le même mot)."
)

ITEM_TEMPLATE = "- {sense_id} | {pos_label} | {lemmas} | {definition}"

POS_LABELS = {"n": "nom", "v": "verbe", "a": "adjectif", "s": "adjectif", "r": "adverbe", "mwe": "expression"}


def _format_item(target: dict) -> str:
    return ITEM_TEMPLATE.format(
        sense_id=target["key"],
        pos_label=POS_LABELS.get(target.get("pos") or "mwe", target.get("pos") or "?"),
        lemmas="/".join(target["lemmas_en"]),
        definition=target.get("definition_en") or "?",
    )


def build_user_prompt(batch: list[dict]) -> str:
    items = "\n".join(_format_item(t) for t in batch)
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
# Cache disque (même principe que pipeline/llm.py, clé dédiée)
# ============================================================


def _cache_path(model: str, system: str, user: str) -> Path:
    cache_key = json.dumps({"model": model, "system": system, "user": user}, sort_keys=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    config.ensure_out_dir()
    return config.CACHE_DIR / f"frontier_{digest}.json"


def _translate_batches(
    batches: list[list[dict]], model: str
) -> tuple[list[dict[str, SenseTranslation]], float]:
    """Traduit chaque lot (avec cache disque par lot). Renvoie la liste
    des {sense_id: SenseTranslation} par lot (même ordre que `batches`)
    et le coût total en USD des SEULS appels réellement effectués (un
    lot servi par le cache ne coûte rien — cohérent avec pipeline/llm.py)."""
    to_call: list[tuple[int, list[dict]]] = []
    results: list[dict[str, SenseTranslation] | None] = [None] * len(batches)
    total_cost = 0.0

    for i, batch in enumerate(batches):
        user_prompt = build_user_prompt(batch)
        cache_file = _cache_path(model, SYSTEM_PROMPT, user_prompt)
        if cache_file.exists():
            parsed = BatchTranslations.model_validate_json(cache_file.read_text(encoding="utf-8"))
            results[i] = {t.sense_id: t for t in parsed.translations}
        else:
            to_call.append((i, batch))

    if to_call:
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(batch)},
            ]
            for _, batch in to_call
        ]
        responses = litellm.batch_completion(
            model=model,
            messages=messages,
            response_format=BatchTranslations,
            reasoning_effort="low",  # recherche lexicale par item, pas du raisonnement long
            max_tokens=16000,
            # Pas de temperature explicite : certains modèles (famille GPT-5)
            # rejettent temperature=0 (litellm.UnsupportedParamsError) et
            # n'acceptent que leur défaut. Sans effet sur la reproductibilité
            # utile ici — c'est le cache disque, indexé sur le texte exact
            # du prompt, qui garantit qu'une relance identique est gratuite.
            max_workers=config.SENSE_FR_FRONTIER_MAX_WORKERS,
        )
        for (i, batch), response in zip(to_call, responses):
            if isinstance(response, Exception):
                print(f"  lot {i}: échec ({response!r}), {len(batch)} sens laissés de côté.")
                results[i] = {}
                continue
            try:
                total_cost += litellm.completion_cost(completion_response=response)
            except Exception:
                pass
            content = response.choices[0].message.content
            parsed = BatchTranslations.model_validate_json(content)
            cache_file = _cache_path(model, SYSTEM_PROMPT, build_user_prompt(batch))
            cache_file.write_text(parsed.model_dump_json(), encoding="utf-8")
            results[i] = {t.sense_id: t for t in parsed.translations}

    return [r or {} for r in results], total_cost


# ============================================================
# Fusion dans le magasin — triage par désaccord
# ============================================================


def _resources_for(target: dict) -> tuple[list[str], list[str]]:
    """omw-fr / WoNeF ne peuvent jamais rien proposer pour une MWE (pas
    de synset, par construction) — cf. sense_fr.classify_mwe_key."""
    if target["kind"] == "mwe":
        return [], []
    synset = target["_synset"]
    omw = sense_fr.fr_candidates_omw(target["_offset"], target["pos"], target["lemmas_en"])
    wonef = sense_fr.fr_candidates_wonef(target["_offset"], target["pos"], target["lemmas_en"])
    return omw, wonef


def build_entry(target: dict, translation: SenseTranslation | None) -> dict:
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
            "evidence": {"omw_fr": omw, "wonef": wonef, "frontier_model": None, "frontier_confidence": None},
            "decided_at": None, "decided_by": None, "note": "",
        })
        return entry_base

    fr, fr_alt = translation.fr[0], translation.fr[1:]
    resource_stems = {senses.fr_stem(c) for c in omw + wonef}
    proposed_stems = {senses.fr_stem(c) for c in translation.fr}
    overlap = bool(resource_stems & proposed_stems)

    if translation.confidence == "low":
        status, agreement = "pending", "frontier_confiance_faible"
    elif not omw and not wonef:
        status, agreement = "auto_llm", "frontier_sans_ressource"
    elif overlap:
        status, agreement = "auto_strong", "frontier_concordant"
    else:
        status, agreement = "pending", "frontier_desaccord"

    entry_base.update({
        "fr": fr, "fr_alt": fr_alt, "status": status, "agreement": agreement,
        "evidence": {
            "omw_fr": omw, "wonef": wonef,
            "frontier_model": config.SENSE_FR_FRONTIER_MODEL,
            "frontier_fr": translation.fr,
            "frontier_confidence": translation.confidence,
        },
        "decided_at": date.today().isoformat() if status in ("auto_strong", "auto_llm") else None,
        "decided_by": "auto_frontier" if status in ("auto_strong", "auto_llm") else None,
        "note": "",
    })
    return entry_base


# ============================================================
# Orchestration
# ============================================================


def run(model: str = config.SENSE_FR_FRONTIER_MODEL, limit: int | None = None, dry_run: bool = False) -> int:
    resolved, unresolved = collect_frontier_targets()
    if limit is not None:
        resolved = resolved[:limit]
    print(f"{len(resolved)} cible(s) à traduire ({len(unresolved)} sense_id non résolu(s) — "
          f"mis en pending sans appel au modèle), modèle={model}.")

    batch_size = config.SENSE_FR_FRONTIER_BATCH_SIZE
    batches = [resolved[i:i + batch_size] for i in range(0, len(resolved), batch_size)]
    translations_by_batch, cost = _translate_batches(batches, model)

    store = sense_fr.load_store()
    n_by_status: dict[str, int] = {}
    for batch, translations in zip(batches, translations_by_batch):
        for target in batch:
            translation = translations.get(target["key"])
            entry = build_entry(target, translation)
            store[entry["key"]] = entry
            n_by_status[entry["status"]] = n_by_status.get(entry["status"], 0) + 1

    for target in unresolved:
        store[target["key"]] = {
            "key": target["key"], "kind": "synset", "lemmas_en": target["lemmas_en"],
            "pos": None, "definition_en": None, "occurrences": target.get("occurrences", 0),
            "fr": None, "fr_alt": [], "status": "pending", "agreement": "sense_id_non_resolu",
            "evidence": {}, "decided_at": None, "decided_by": None, "note": "",
        }
        n_by_status["pending"] = n_by_status.get("pending", 0) + 1

    print(f"Coût constaté (appels non-cachés uniquement) : ${cost:.4f}")
    print("Ventilation des statuts sur ce run :", n_by_status)

    if dry_run:
        print("--dry-run : rien n'est écrit dans le magasin ni dans le CSV de relecture.")
        return 0

    sense_fr.write_store(store)
    n_pending = sense_fr.write_review_csv(store)
    n_validated = sum(1 for e in store.values() if e["status"] == "validated")
    n_auto_strong = sum(1 for e in store.values() if e["status"] == "auto_strong")
    n_auto_llm = sum(1 for e in store.values() if e["status"] == "auto_llm")
    print(f"Magasin : {len(store)} entrées ({n_validated} validées, {n_auto_strong} auto_strong, "
          f"{n_auto_llm} auto_llm, {n_pending} en attente -> {config.SENSE_FR_REVIEW_PATH}).")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=config.SENSE_FR_FRONTIER_MODEL,
        help="Modèle LiteLLM (préfixé par le fournisseur, p.ex. anthropic/claude-opus-5, "
             "openai/gpt-5...). Le fournisseur lit ses identifiants depuis les variables "
             "d'environnement standard (ANTHROPIC_API_KEY, OPENAI_API_KEY...).",
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
