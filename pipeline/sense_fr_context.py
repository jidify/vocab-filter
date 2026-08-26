"""Traduction CONTEXTUELLE, à SENS IMPOSÉ — étape 2 du dispositif
d'arbitrage sans relecture humaine (voir le plan "Valider / corriger
suggested_fr et suggested_fr_alt").

Différence avec pipeline/sense_fr_frontier.py (traduction "de
dictionnaire", glose seule, sens choisi librement par le modèle) : ici on
fournit au modèle la PHRASE ANGLAISE RÉELLE du livre où le mot cible
apparaît, ET le sens déjà tranché par S5 (synset + définition) — le
modèle n'est plus libre de dériver vers un autre sens, il ne répond qu'à
la question lexicale : "comment ce mot, dans CETTE phrase, se traduit-il
en français ?" C'est une tâche différente de celle qui a produit
`suggested_fr`, même si le même modèle peut être employé pour les deux
(voir le plan : un seul fournisseur, rôles non circulaires).

Sortie structurée en TROIS champs distincts (vocab-filter-resume.md §6,
qui distingue l'élément français aligné du sens contextuel réel — exemple
"access" -> aligné "clé", sens réel "avoir accès") :
- traduction_lexicale : le mot/groupe qui traduit CE mot dans CE sens ;
- sens_contextuel      : ce que le passage veut dire (reformulation admise) ;
- translation_type     : equivalence_directe / reformulation / explicitation.

SEULE une `equivalence_directe` peut corroborer `suggested_fr` (voir
pipeline/sense_fr_adjudicate.py) — une reformulation est journalisée mais
ne tranche rien, sans quoi on confondrait "traduction fidèle du sens" et
"réponse plausible à la phrase". C'est cette distinction qui fait défaut
au signal du corpus bilingue (voir la docstring de ce dernier).

IMPORTANT — ce signal N'EST PAS une vérité de terrain : la même famille de
modèle a produit `suggested_fr`. Un accord mesuré ici est un ACCORD entre
deux lectures (contextuelle vs dictionnaire), jamais une précision (voir
pipeline/eval_sense_fr.py).

Portée par défaut de la passe groupée (voir config.py) : tous les
`pending` et `auto_llm` du magasin (ce sont eux qui manquent de
corroboration), plus un échantillon stratifié d'`auto_strong` en groupe
témoin — sans lui, un taux d'accord sur le résidu est ininterprétable.

N'écrit JAMAIS `fr`/`status` : seulement `entry["context_evidence"]`,
laissé à pipeline/sense_fr_adjudicate.py pour la décision finale.

Usage :
    uv run python -m pipeline.sense_fr_context
    uv run python -m pipeline.sense_fr_context --limit 20 --dry-run
    uv run python -m pipeline.sense_fr_context --model openai/gpt-5-mini
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import litellm
from pydantic import BaseModel

from pipeline import config, sense_fr
from pipeline.sense_fr import _stratified_sample
from pipeline.sense_fr_frontier import POS_LABELS, collect_frontier_targets

# ============================================================
# Schéma de sortie structurée
# ============================================================


class SenseContextTranslation(BaseModel):
    sense_id: str
    traduction_lexicale: str
    sens_contextuel: str
    translation_type: Literal["equivalence_directe", "reformulation", "explicitation"]
    confidence: Literal["high", "medium", "low"]


class BatchContextTranslations(BaseModel):
    translations: list[SenseContextTranslation]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = (
    "Tu es lexicographe bilingue anglais-français, spécialisé en traduction "
    "littéraire. Pour CHAQUE sens WordNet identifié par son sense_id (ou une "
    "clé mwe:... pour une expression figée sans entrée WordNet), on te donne : "
    "le sens PRÉCIS déjà retenu (lemmes anglais, catégorie grammaticale, "
    "définition), ET une ou plusieurs phrases RÉELLES d'un livre où le mot "
    "cible apparaît dans CE sens (le mot ou groupe cible de chaque phrase est "
    "indiqué séparément par 'mot cible'). Le sens est déjà tranché en amont : "
    "n'en propose JAMAIS un autre, même si une phrase prise isolément te "
    "semblerait ambiguë.\n\n"
    "Pour chaque sense_id, réponds avec TROIS champs distincts :\n"
    "- traduction_lexicale : le mot ou groupe de mots français qui traduit CE "
    "mot, dans CE sens précis, tel qu'on le trouverait dans un dictionnaire "
    "bilingue soigné — PAS une paraphrase de la phrase entière ;\n"
    "- sens_contextuel : ce que le passage veut dire dans son contexte ; "
    "reformulation autorisée, et même recommandée si une traduction littérale "
    "serait maladroite ou trahirait le registre ;\n"
    "- translation_type : \"equivalence_directe\" si traduction_lexicale est un "
    "vrai équivalent lexical de CE mot dans CE sens (même registre, "
    "substituable dans une traduction soignée) ; \"reformulation\" si le "
    "passage condense, déplace ou explicite l'information au point qu'aucun "
    "mot ou groupe isolé ne correspond vraiment au mot anglais ; "
    "\"explicitation\" si la traduction ajoute une précision absente de "
    "l'anglais mais nécessaire en français.\n\n"
    "Indique aussi ta confiance (\"high\"/\"medium\"/\"low\") sur "
    "traduction_lexicale : \"low\" si les phrases fournies ne permettent pas "
    "vraiment de trancher un mot isolé.\n\n"
    "Renvoie EXACTEMENT un objet par sense_id reçu, avec le sense_id recopié "
    "à l'identique (ne jamais indexer par le mot anglais)."
)

ITEM_HEADER = "- {sense_id} | {pos_label} | {lemmas} | sens retenu : {definition}"
ITEM_OCCURRENCE = '    contexte : "{context}" || mot cible dans ce contexte : "{target_surface}"'


def _format_item(target: dict, occurrences: list[dict]) -> str:
    lines = [ITEM_HEADER.format(
        sense_id=target["key"],
        pos_label=POS_LABELS.get(target.get("pos") or "mwe", target.get("pos") or "?"),
        lemmas="/".join(target["lemmas_en"]),
        definition=target.get("definition_en") or "?",
    )]
    for occ in occurrences:
        lines.append(ITEM_OCCURRENCE.format(
            context=occ["context"], target_surface=occ["target_surface"],
        ))
    return "\n".join(lines)


def build_user_prompt(batch: list[tuple[dict, list[dict]]]) -> str:
    items = "\n".join(_format_item(target, occs) for target, occs in batch)
    return f"Sens à traduire en contexte ({len(batch)}) :\n{items}"


# ============================================================
# Collecte des occurrences (phrases réelles) par sense_id
# ============================================================


def load_occurrences_by_sense() -> dict[str, list[dict]]:
    """Index sense_id -> occurrences (context, target_surface, segment_idx)
    depuis pipeline_out/senses.jsonl. Seules les unités "word" (kind
    "synset") y ont une entrée par occurrence — les MWE (occurrence_
    segment_idxs de selected_mwe.jsonl) n'y figurent pas avec un mot
    cible unique par segment, donc restent hors de cette passe pour
    l'instant (voir la limitation notée dans le plan) : elles n'auront
    simplement aucune occurrence trouvée ici et seront journalisées comme
    telles par collect_context_targets, pas silencieusement ignorées."""
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
    return by_sense


def _pick_diverse_occurrences(occurrences: list[dict], k: int) -> list[dict]:
    """k occurrences réparties dans le livre (pas les k premières —
    l'ordre de segment_idx couvre le début/milieu/fin) : des contextes
    différents sont ce qui rend le vote entre occurrences (passe résiduelle
    par occurrence) informatif plutôt que redondant."""
    ordered = sorted(occurrences, key=lambda o: o["segment_idx"])
    n = len(ordered)
    if n <= k:
        return ordered
    step = n / k
    picked_idx = sorted({int(i * step) for i in range(k)})
    return [ordered[i] for i in picked_idx]


# ============================================================
# Sélection de la portée (pending + auto_llm + échantillon auto_strong)
# ============================================================


def _sample_auto_strong(store: dict[str, dict], n: int) -> list[str]:
    """Échantillon stratifié par catégorie grammaticale (pas les n
    premières clés par ordre alphabétique) — groupe témoin représentatif,
    pas juste des mots en 'a'. Réutilise le tirage rond-robin déjà écrit
    pour --retry-pending (pipeline/sense_fr.py)."""
    buckets: dict[str, list[str]] = {}
    for key, entry in store.items():
        if entry["status"] != "auto_strong":
            continue
        buckets.setdefault(entry.get("pos") or "?", []).append(key)
    for bucket in buckets.values():
        bucket.sort()
    return _stratified_sample(buckets, n)


def determine_scope(store: dict[str, dict], auto_strong_sample: int) -> dict[str, str]:
    """Renvoie {sense_id: raison} pour toutes les clés à soumettre à la
    passe contextuelle groupée. `raison` sert uniquement au journal
    (pending / auto_llm / auto_strong_temoin)."""
    scope: dict[str, str] = {}
    for key, entry in store.items():
        if entry["status"] == "pending":
            scope[key] = "pending"
        elif entry["status"] == "auto_llm":
            scope[key] = "auto_llm"
    for key in _sample_auto_strong(store, auto_strong_sample):
        scope[key] = "auto_strong_temoin"
    return scope


# ============================================================
# Cache disque (même principe que sense_fr_frontier, espace dédié)
# ============================================================


def _cache_path(model: str, system: str, user: str) -> Path:
    cache_key = json.dumps({"model": model, "system": system, "user": user}, sort_keys=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    config.ensure_out_dir()
    return config.CACHE_DIR / f"context_{digest}.json"


def _translate_batches(
    batches: list[list[tuple[dict, list[dict]]]], model: str
) -> tuple[list[dict[str, SenseContextTranslation]], float]:
    to_call: list[tuple[int, list[tuple[dict, list[dict]]]]] = []
    results: list[dict[str, SenseContextTranslation] | None] = [None] * len(batches)
    total_cost = 0.0

    for i, batch in enumerate(batches):
        user_prompt = build_user_prompt(batch)
        cache_file = _cache_path(model, SYSTEM_PROMPT, user_prompt)
        if cache_file.exists():
            parsed = BatchContextTranslations.model_validate_json(cache_file.read_text(encoding="utf-8"))
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
            response_format=BatchContextTranslations,
            reasoning_effort="low",
            max_tokens=16000,
            max_workers=config.SENSE_FR_CONTEXT_MAX_WORKERS,
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
            parsed = BatchContextTranslations.model_validate_json(content)
            cache_file = _cache_path(model, SYSTEM_PROMPT, build_user_prompt(batch))
            cache_file.write_text(parsed.model_dump_json(), encoding="utf-8")
            results[i] = {t.sense_id: t for t in parsed.translations}

    return [r or {} for r in results], total_cost


# ============================================================
# Orchestration
# ============================================================


def run(
    model: str = config.SENSE_FR_CONTEXT_MODEL,
    limit: int | None = None,
    dry_run: bool = False,
    auto_strong_sample: int = config.SENSE_FR_CONTEXT_AUTO_STRONG_SAMPLE,
) -> int:
    store = sense_fr.load_store()
    scope = determine_scope(store, auto_strong_sample)
    print(f"Portée : {len(scope)} sens "
          f"({sum(1 for r in scope.values() if r == 'pending')} pending, "
          f"{sum(1 for r in scope.values() if r == 'auto_llm')} auto_llm, "
          f"{sum(1 for r in scope.values() if r == 'auto_strong_temoin')} auto_strong témoin).")

    resolved, _unresolved = collect_frontier_targets()
    resolved_by_key = {t["key"]: t for t in resolved}

    occurrences_by_sense = load_occurrences_by_sense()

    items: list[tuple[dict, list[dict]]] = []
    n_no_occurrence = 0
    for key in scope:
        target = resolved_by_key.get(key)
        if target is None:
            continue  # sense_id non résolu côté WordNet (déjà pending par ailleurs)
        occs = occurrences_by_sense.get(key)
        if not occs:
            n_no_occurrence += 1
            continue
        items.append((target, _pick_diverse_occurrences(occs, config.SENSE_FR_CONTEXT_MAX_OCCURRENCES)))

    if n_no_occurrence:
        print(f"  {n_no_occurrence} clé(s) de la portée sans occurrence exploitable dans "
              f"{config.SENSES_PATH} (typiquement des MWE — hors de cette passe pour l'instant) "
              f"laissée(s) sans preuve contextuelle.")

    if limit is not None:
        items = items[:limit]
    print(f"{len(items)} sens avec au moins une phrase réelle, modèle={model}.")

    batch_size = config.SENSE_FR_CONTEXT_BATCH_SIZE
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    translations_by_batch, cost = _translate_batches(batches, model)

    n_by_type: dict[str, int] = {}
    n_written = 0
    for batch, translations in zip(batches, translations_by_batch):
        for target, occs in batch:
            translation = translations.get(target["key"])
            if translation is None:
                continue
            n_by_type[translation.translation_type] = n_by_type.get(translation.translation_type, 0) + 1
            if dry_run:
                continue
            entry = store.get(target["key"])
            if entry is None:
                continue
            entry["context_evidence"] = {
                "model": model,
                "traduction_lexicale": translation.traduction_lexicale,
                "sens_contextuel": translation.sens_contextuel,
                "translation_type": translation.translation_type,
                "confidence": translation.confidence,
                "occurrences_used": [
                    {"context": o["context"], "target_surface": o["target_surface"]} for o in occs
                ],
            }
            n_written += 1

    print(f"Coût constaté (appels non-cachés uniquement) : ${cost:.4f}")
    print("Ventilation par translation_type :", n_by_type)
    total = sum(n_by_type.values())
    if total:
        equiv = n_by_type.get("equivalence_directe", 0)
        print(f"  dont equivalence_directe : {equiv}/{total} ({equiv / total:.0%}) — "
              f"seule cette catégorie peut corroborer suggested_fr.")

    if dry_run:
        print("--dry-run : rien n'est écrit dans le magasin.")
        return 0

    sense_fr.write_store(store)
    print(f"{n_written} entrée(s) enrichie(s) de `context_evidence` dans {config.SENSE_FR_STORE_PATH}.")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.SENSE_FR_CONTEXT_MODEL)
    parser.add_argument("--limit", type=int, default=None,
                         help="Limite le nombre de sens traités (test avant la passe complète).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Appelle le modèle et affiche le résultat/coût, mais n'écrit rien sur disque.")
    parser.add_argument("--auto-strong-sample", type=int, default=config.SENSE_FR_CONTEXT_AUTO_STRONG_SAMPLE,
                         help="Taille du groupe témoin tiré parmi les auto_strong.")
    args = parser.parse_args()
    raise SystemExit(run(
        model=args.model, limit=args.limit, dry_run=args.dry_run,
        auto_strong_sample=args.auto_strong_sample,
    ))
