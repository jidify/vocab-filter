"""Client LLM unique, basé sur LiteLLM, pour toutes les tâches du pipeline.

fix_pipeline/multi_models/report_multi_models.md §4bis constatait deux
clients LLM indépendants dans ce dépôt : `pipeline/llm.py` (stdlib
`urllib`, JSON fait main, pour S3/S5/S6-*-local) et LiteLLM appelé
directement dans `sense_fr_frontier.py`/`sense_fr_reassign.py`/
`sense_fr_adjudicate.py` (pour les 4 tâches S6 batchées). Ce module
remplace les deux : toute tâche du registre `pipeline/llm_tasks.py` passe
par ce module, via `call()`/`call_batch_completion()` (S3-judge-type et les
2 tâches `S6-*-local`, cache disque par prompt) ou via `run_units()` (les 7
tâches en lot — S3-judge-occurrence, S3-definition-cluster, S5-arbitrate,
et les 4 tâches S6 batchées — cache unitaire, voir plus bas). Aucune
n'utilise `mode_batch` LiteLLM — voir la mise en garde de
`fix_pipeline/multi_models/prompts_multi_models.md`.

``call()``/``call_batch_completion()`` NE décident PAS de la structure de
leur clé de cache disque : chaque appelant construit son propre dict
`cache_key_fields` (et, pour les 4 tâches historiquement LiteLLM, son propre
préfixe de fichier) — préservé tel quel pour ne pas invalider un cache de
traduction déjà payé (voir Lot U3 du plan d'unification). Ces deux fonctions
ne font que hasher ce dict et gérer lecture/écriture/appel réseau autour ;
elles restent pour tout appel non décomposable en unités.

``run_units()`` (plan "décorréler l'appel en lot du stockage unitaire") EST
la structure de clé, à l'inverse : elle appelle en LOT (vitesse) et stocke
en UNITAIRE dans `pipeline/llm_store.py` (reprise), avec une clé faite de
valeurs métier lisibles (`task_id`, `model`, `protocol`, `unit_id`) — jamais
`batch_size`/`mode_batch`, jamais le texte du prompt rendu. C'est la voie
que suivent désormais les 7 tâches en lot du registre
`pipeline/llm_tasks.py` ; plus aucune d'elles ne passe de prompt de LOT par
``cache_path_for``.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import litellm
from pydantic import BaseModel, ValidationError

from pipeline import config, llm_store
from pipeline import llm_litellm_catgpt


class LLMError(RuntimeError):
    pass


# ============================================================
# Ordre de présentation déterministe — remplace tout random.Random(42) partagé
# ============================================================


def presentation_order(items: list[str], unit_id: str) -> list[str]:
    """Permutation déterministe de ``items``, dérivée de ``unit_id`` — jamais
    un tirage aléatoire. Sert à masquer l'origine d'une liste de candidats
    (omw-fr/WoNeF/DBnary/Apertium, ou toute autre concaténation dont l'ORDRE
    trahirait la source) sans jamais dépendre de la position de l'unité dans
    un lot, ni d'un état partagé (l'ancien `random.Random(42)` consommé
    séquentiellement sur tout un lot, voir TODO/pipeline_logging_to_files.md
    et le plan de décorrélation lot/stockage).

    Propriété qui compte ici : une fonction PURE de ``(unit_id, ensemble des
    items)`` — la même unité produit toujours le même ordre, quel que soit le
    lot où elle atterrit, sans qu'il soit nécessaire de stocker cet ordre ni
    de le faire entrer dans `payload_sig` (pipeline/llm_store.py). Un ordre
    tiré par `random.Random(42).shuffle()` sur tout le lot ne peut pas avoir
    cette propriété : le résultat pour une unité dépend alors de combien
    d'appels `.shuffle()` ont eu lieu avant elle dans CE lot précis — ce qui a
    mesurablement fait basculer un verdict entre deux compositions de lot
    différentes (voir pipeline/config.py, commentaire sur
    SENSE_FR_REASSIGN_BATCH_SIZE, cas `beat.n.08` -> `beat.n.06`)."""
    return sorted(items, key=lambda item: hashlib.sha256(f"{unit_id}\x00{item}".encode("utf-8")).hexdigest())


# ============================================================
# Cache disque — mécanique seule ; la forme de la clé appartient à l'appelant
# ============================================================


def build_cache_key(*, model: str, system: str, prompt: str, extra: dict | None = None) -> dict:
    """Clé de cache générique pour un appel `call()` migré depuis
    ``pipeline/llm.py`` (S3/S5/S6-*-local, Lot U2) — ``extra`` porte les
    champs propres à la tâche (task_id, mode_batch, batch_size,
    prompt_variant, protocole de prompt...). Les 4 tâches historiquement
    LiteLLM (Lot U3) NE passent PAS par cette fonction : elles construisent
    leur propre dict, préservé à l'identique pour ne pas invalider un cache
    de traduction déjà payé."""
    return {"model": model, "system": system, "prompt": prompt,
            "temperature": config.LLM_TEMPERATURE, "extra": extra or {}}


def cache_path_for(cache_key_fields: dict, *, prefix: str = "") -> Path:
    config.ensure_out_dir()
    digest = hashlib.sha256(
        json.dumps(cache_key_fields, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return config.CACHE_DIR / f"{prefix}{digest}.json"


def _cache_read(cache_file: Path, *, response_model: type[BaseModel] | None):
    if not cache_file.exists():
        return None
    content = cache_file.read_text(encoding="utf-8")
    if response_model is not None:
        return response_model.model_validate_json(content)
    return json.loads(content)


def _cache_write(cache_file: Path, parsed: Any, *, response_model: type[BaseModel] | None) -> None:
    content = (
        parsed.model_dump_json() if response_model is not None
        else json.dumps(parsed, ensure_ascii=False)
    )
    cache_file.write_text(content, encoding="utf-8")


# ============================================================
# Kwargs LiteLLM communs — provider (ollama api_base, catgpt handler),
# schéma de sortie, réglages optionnels
# ============================================================

_SKIP_TEMPERATURE_PROVIDERS = {"openai"}  # certains modèles (famille GPT-5) rejettent
                                           # temperature=0 (litellm.UnsupportedParamsError) ;
                                           # voir sense_fr_frontier.py::_translate_batches


def _provider_of(model: str) -> str:
    return model.split("/", 1)[0]


def _completion_kwargs(
    model: str, *, response_model: type[BaseModel] | None,
    reasoning_effort: str | None, max_tokens: int | None,
) -> dict:
    kwargs: dict[str, Any] = {
        "response_format": response_model if response_model is not None else {"type": "json_object"},
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    provider = _provider_of(model)
    if provider not in _SKIP_TEMPERATURE_PROVIDERS:
        kwargs["temperature"] = config.LLM_TEMPERATURE
    if provider == "ollama":
        kwargs["api_base"] = config.OLLAMA_URL
    kwargs.update(llm_litellm_catgpt.call_kwargs(model))
    return kwargs


# ============================================================
# Appel unitaire
# ============================================================


def call(
    *, model: str, system: str, prompt: str,
    response_model: type[BaseModel] | None = None,
    cache_key_fields: dict, cache_prefix: str = "",
    reasoning_effort: str | None = None, max_tokens: int | None = None,
    timeout: float | None = None, return_cost: bool = False,
):
    """Un appel LLM, avec cache disque. Renvoie un ``dict`` (JSON libre) si
    ``response_model`` est ``None``, sinon une instance de ce modèle
    Pydantic. Lève ``LLMError`` si le provider est injoignable ou si la
    réponse ne respecte pas le contrat attendu — l'appelant doit alors
    dégrader (ex. décision "incertain"), jamais laisser planter le run.

    ``return_cost=True`` renvoie ``(parsed, cost_usd)`` à la place — coût
    ``0.0`` sur un hit de cache (aucun appel réellement effectué) ou si
    ``litellm.completion_cost`` échoue (modèle sans tarification connue).
    Par défaut ``False`` pour ne rien changer au contrat des appelants déjà
    migrés (Lots U2/U3) ; utile aux outils de diagnostic (ex.
    fix_pipeline/evaluate_s3_judges.py) qui veulent le coût réel."""
    cache_file = cache_path_for(cache_key_fields, prefix=cache_prefix)
    cached = _cache_read(cache_file, response_model=response_model)
    if cached is not None:
        return (cached, 0.0) if return_cost else cached

    kwargs = _completion_kwargs(
        model, response_model=response_model,
        reasoning_effort=reasoning_effort, max_tokens=max_tokens,
    )
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            timeout=timeout,
            **kwargs,
        )
    except Exception as exc:
        raise LLMError(f"{model} injoignable ou erreur d'appel : {exc}") from exc

    content = response.choices[0].message.content
    try:
        parsed = (
            response_model.model_validate_json(content) if response_model is not None
            else json.loads(content)
        )
    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
        raise LLMError(f"réponse {model} invalide : {content[:200]!r}") from exc

    _cache_write(cache_file, parsed, response_model=response_model)
    if not return_cost:
        return parsed
    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:
        cost = 0.0
    return parsed, cost


# ============================================================
# Appels multiples indépendants — litellm.batch_completion (parallélisme
# HTTP, JAMAIS à confondre avec mode_batch : chaque BatchItem est un
# prompt complet et distinct, voir la mise en garde du plan multi-modèles)
# ============================================================


@dataclass
class BatchItem:
    system: str
    user: str
    cache_key_fields: dict
    cache_prefix: str = ""
    label: Any = None  # identifiant libre reporté à on_error, pour un message utile


def call_batch_completion(
    items: list[BatchItem], *, model: str,
    response_model: type[BaseModel],
    reasoning_effort: str | None = None, max_tokens: int | None = None,
    max_workers: int = 1,
    on_error: Callable[[int, BatchItem, Exception], None] | None = None,
) -> tuple[list[Any | None], float]:
    """Renvoie ``(results, total_cost_usd)``, ``results`` dans l'ordre de
    ``items`` ; ``results[i]`` est ``None`` si ce cas a échoué (l'appelant
    décide du repli, comme avant : un dict vide côté S6b/S6c). Le coût ne
    compte que les appels réellement effectués (un item servi par le cache
    ne coûte rien — cohérent avec le reste du pipeline)."""
    to_call: list[tuple[int, BatchItem]] = []
    results: list[Any | None] = [None] * len(items)

    for i, item in enumerate(items):
        cache_file = cache_path_for(item.cache_key_fields, prefix=item.cache_prefix)
        cached = _cache_read(cache_file, response_model=response_model)
        if cached is not None:
            results[i] = cached
        else:
            to_call.append((i, item))

    total_cost = 0.0
    if not to_call:
        return results, total_cost

    kwargs = _completion_kwargs(
        model, response_model=response_model,
        reasoning_effort=reasoning_effort, max_tokens=max_tokens,
    )
    messages = [
        [{"role": "system", "content": item.system}, {"role": "user", "content": item.user}]
        for _, item in to_call
    ]
    try:
        responses = litellm.batch_completion(
            model=model, messages=messages, max_workers=max_workers, **kwargs,
        )
    except Exception as exc:
        raise LLMError(f"{model} injoignable ou erreur d'appel (lot) : {exc}") from exc

    for (i, item), response in zip(to_call, responses):
        if isinstance(response, Exception):
            if on_error is not None:
                on_error(i, item, response)
            continue
        try:
            total_cost += litellm.completion_cost(completion_response=response)
        except Exception:
            pass
        content = response.choices[0].message.content
        try:
            parsed = response_model.model_validate_json(content)
        except ValidationError as exc:
            if on_error is not None:
                on_error(i, item, exc)
            continue
        cache_file = cache_path_for(item.cache_key_fields, prefix=item.cache_prefix)
        _cache_write(cache_file, parsed, response_model=response_model)
        results[i] = parsed

    return results, total_cost


# ============================================================
# run_units — appeler en LOT, stocker en UNITAIRE (pipeline/llm_store.py)
# ============================================================


@dataclass
class Unit:
    """Unité de travail pour `run_units` — indépendante du lot où elle
    atterrit. `unit_id`/`payload` sont ce qui identifie et signe la ligne
    dans `pipeline/llm_store.py` (jamais le texte du prompt rendu, jamais la
    position dans un lot) ; `data` est l'objet métier que l'appelant
    retrouve dans ses fonctions de rendu/parsing."""
    unit_id: str
    payload: dict
    data: Any = None


def dedupe_batch_items(items: Any, *, id_key: str) -> dict[str, dict]:
    """Éclate une liste de décisions ``{id_key: ..., ...}`` (le contenu
    d'une enveloppe de lot — ``decisions``/``translations``/``guesses``/
    ``verdicts``, selon la tâche) en dict indexé par ``id_key``, en écartant
    les entrées dupliquées (jamais l'une des deux gardée au hasard) — motif
    répété à l'identique dans les 7 tâches en lot avant `run_units` (voir
    p.ex. l'ancien `mwe_judge.py::judge_occurrences_batch`)."""
    received: dict[str, dict] = {}
    duplicates: set[str] = set()
    for item in items:
        item_id = getattr(item, id_key, None) if not isinstance(item, dict) else item.get(id_key)
        if not item_id:
            continue
        if item_id in received:
            duplicates.add(item_id)
        else:
            received[item_id] = item
    for dup in duplicates:
        received.pop(dup, None)
    return received


def _complete_batch_raw(
    message_pairs: list[tuple[str, str]], *, model: str,
    response_model: type[BaseModel] | None,
    reasoning_effort: str | None, max_tokens: int | None,
    max_workers: int, timeout: float | None = None,
) -> tuple[list[Any], float]:
    """``litellm.batch_completion`` sur une liste de ``(system, user)`` —
    SANS aucun cache disque (contrairement à `call_batch_completion`, dont
    la mécanique de cache reste inchangée pour ses propres appelants). Un
    élément du résultat est soit l'objet parsé, soit l'``Exception``
    rencontrée pour CET item précis — jamais une exception globale qui
    invaliderait tous les autres (`litellm.batch_completion` renvoie déjà
    les erreurs par item dans la liste plutôt que de lever)."""
    kwargs = _completion_kwargs(
        model, response_model=response_model,
        reasoning_effort=reasoning_effort, max_tokens=max_tokens,
    )
    if timeout is not None:
        kwargs["timeout"] = timeout
    messages = [
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for system, user in message_pairs
    ]
    try:
        responses = litellm.batch_completion(model=model, messages=messages, max_workers=max_workers, **kwargs)
    except Exception as exc:
        raise LLMError(f"{model} injoignable ou erreur d'appel (lot) : {exc}") from exc

    total_cost = 0.0
    out: list[Any] = []
    for response in responses:
        if isinstance(response, Exception):
            out.append(response)
            continue
        try:
            total_cost += litellm.completion_cost(completion_response=response)
        except Exception:
            pass
        content = response.choices[0].message.content
        try:
            parsed = (
                response_model.model_validate_json(content) if response_model is not None
                else json.loads(content)
            )
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            out.append(exc)
            continue
        out.append(parsed)
    return out, total_cost


def run_units(
    units: list[Unit], *, task_id: str, model: str, protocol: str,
    render_unit: Callable[[Unit], tuple[str, str]],
    render_batch: Callable[[list[Unit]], tuple[str, str]],
    parse_unit: Callable[[Any, Unit], dict],
    parse_batch: Callable[[Any, list["Unit"]], dict[str, dict]],
    response_model_unit: type[BaseModel] | None,
    response_model_batch: type[BaseModel] | None,
    batch_size: int, mode_batch: bool,
    reasoning_effort: str | None = None, max_tokens: int | None = None,
    max_workers: int = 1, timeout: float | None = None,
    on_failure: Callable[[Unit, Exception | str], None] | None = None,
    return_cost: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], float]:
    """Appelle en LOT, stocke en UNITAIRE — la pièce centrale du plan de
    décorrélation lot/stockage. Renvoie ``{unit_id: décision}`` pour toute
    unité résolue (hit du magasin OU appel réussi ce run-ci) ; une unité en
    échec est simplement absente du résultat et signalée via ``on_failure``.

    ``return_cost=True`` renvoie ``(résultats, cost_usd)`` à la place — même
    convention que ``call(return_cost=True)`` : coût des SEULS appels
    réellement effectués (un hit du magasin ne coûte rien).
    NE LÈVE JAMAIS ``LLMError`` : une panne systémique du provider (tout le
    lot de tranches injoignable) est traitée exactement comme une tranche
    en échec — chaque unité encore en attente part vers ``on_failure``, mais
    les hits déjà trouvés dans `llm_store` avant l'appel réseau restent dans
    le résultat renvoyé (jamais perdus par une exception qui remonterait
    avant le `return`). Un ``parse_unit``/``parse_batch`` fourni par
    l'appelant qui lève est traité de la même façon, jamais laissé planter
    le run.

    1. Un hit dans `pipeline/llm_store.py` sort directement, quels que
       soient ``batch_size``/``mode_batch`` COURANTS : la taille de lot
       n'entre JAMAIS dans la clé de stockage. Changer `batch_size` d'un run
       à l'autre, ou repasser du lot à l'unitaire, ne repaie donc plus rien
       pour les unités déjà décidées par ce modèle sous ce protocole.
    2. Les miss sont regroupés en tranches d'au plus ``batch_size`` — une
       seule unité par tranche si ``not mode_batch`` ou ``batch_size < 2``,
       même seuil que `pipeline.llm_tasks.use_batch_prompt` — puis envoyées
       en parallèle HTTP (`litellm.batch_completion`, ``max_workers``
       tranches à la fois ; PAS `mode_batch` LiteLLM, voir la mise en garde
       en tête de ce module).
    3. Chaque réponse de tranche est réextraite PAR UNITÉ (``parse_unit``/
       ``parse_batch``) puis écrite dans `llm_store` en une seule
       transaction — si une tranche de 50 unités n'en valide que 47, les 47
       sont quand même stockées (`llm_store.put_many`, jamais de
       tout-ou-rien par lot ; gain net face à l'ancien cache disque, où un
       lot qui échouait au parsing ne stockait RIEN). Une unité absente ou
       dupliquée dans sa tranche, ou une tranche qui échoue entièrement,
       part vers ``on_failure`` et n'est jamais stockée — même règle que le
       magasin métier historique (voir `mwe_judge.py`, "pannes LLM — pas
       mises en cache non plus")."""
    def _done(results: dict[str, dict], cost: float):
        return (results, cost) if return_cost else results

    if not units:
        return _done({}, 0.0)

    unit_by_id: dict[str, Unit] = {u.unit_id: u for u in units}
    sig_by_id = {uid: llm_store.payload_sig(u.payload) for uid, u in unit_by_id.items()}

    results: dict[str, dict] = dict(llm_store.get_many(
        task_id=task_id, model=model, protocol=protocol, wanted=list(sig_by_id.items()),
    ))
    missing = [u for uid, u in unit_by_id.items() if uid not in results]
    if not missing:
        return _done(results, 0.0)

    use_batch = mode_batch and batch_size >= 2
    chunk_size = batch_size if use_batch else 1
    chunks = [missing[i:i + chunk_size] for i in range(0, len(missing), chunk_size)]
    response_model = response_model_batch if use_batch else response_model_unit
    message_pairs = [
        render_batch(chunk) if use_batch else render_unit(chunk[0])
        for chunk in chunks
    ]

    try:
        responses, cost = _complete_batch_raw(
            message_pairs, model=model, response_model=response_model,
            reasoning_effort=reasoning_effort, max_tokens=max_tokens,
            max_workers=max_workers, timeout=timeout,
        )
    except LLMError as exc:
        # Panne systémique (provider entièrement injoignable) : NE JAMAIS
        # perdre les hits déjà trouvés dans `results` pour ça — chaque unité
        # qui restait à calculer part vers `on_failure`, `results` est
        # renvoyé tel quel plutôt que de laisser l'exception tout emporter.
        for chunk in chunks:
            for unit in chunk:
                if on_failure is not None:
                    on_failure(unit, exc)
        return _done(results, 0.0)

    rows: list[llm_store.ResultRow] = []
    for chunk, response in zip(chunks, responses):
        if isinstance(response, Exception):
            for unit in chunk:
                if on_failure is not None:
                    on_failure(unit, response)
            continue
        try:
            if use_batch:
                decisions = parse_batch(response, chunk)
            else:
                decisions = {chunk[0].unit_id: parse_unit(response, chunk[0])}
        except Exception as exc:  # parse_unit/parse_batch de l'appelant : jamais planter le run
            for unit in chunk:
                if on_failure is not None:
                    on_failure(unit, exc)
            continue
        for unit in chunk:
            decision = decisions.get(unit.unit_id)
            if decision is None:
                if on_failure is not None:
                    on_failure(unit, "unité absente ou dupliquée dans la réponse de lot")
                continue
            results[unit.unit_id] = decision
            rows.append(llm_store.ResultRow(
                task_id=task_id, model=model, protocol=protocol,
                unit_id=unit.unit_id, payload=unit.payload, result=decision,
                batch_size=len(chunk), mode_batch=use_batch, source="live",
            ))
    llm_store.put_many(rows)
    return _done(results, cost)


# ============================================================
# Disponibilité — reste un simple GET, jamais routé par LiteLLM (aucune
# raison de faire passer un health-check par la mécanique de complétion)
# ============================================================


def is_available(*, backend: str | None = None) -> bool:
    """``backend`` explicite (``"ollama"``/``"catgpt"``/``"openai"``) pour
    pinger le provider réellement résolu par un appelant (p.ex.
    ``pipeline.llm_tasks.task_config(task_id).provider``, qui honore
    l'alias ``.env`` ``PROVIDER=chatgpt``) plutôt que ``config.LLM_BACKEND``
    seul — sans ``backend``, repli sur ``config.LLM_BACKEND``."""
    backend = backend or config.LLM_BACKEND
    try:
        if backend == "catgpt":
            req = urllib.request.Request(
                f"{config.CATGPT_BASE_URL}/models",
                headers={"Authorization": f"Bearer {config.CATGPT_API_TOKEN}"},
            )
        elif backend == "openai":
            import os
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            req = urllib.request.Request(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
            )
        else:
            req = urllib.request.Request(f"{config.OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
