"""Client LLM unique, basé sur LiteLLM, pour toutes les tâches du pipeline.

fix_pipeline/multi_models/report_multi_models.md §4bis constatait deux
clients LLM indépendants dans ce dépôt : `pipeline/llm.py` (stdlib
`urllib`, JSON fait main, pour S3/S5/S6-*-local) et LiteLLM appelé
directement dans `sense_fr_frontier.py`/`sense_fr_reassign.py`/
`sense_fr_adjudicate.py` (pour les 4 tâches S6 batchées). Ce module
remplace les deux : toute tâche du registre `pipeline/llm_tasks.py`
passe désormais par `call()` (un appel) ou `call_batch_completion()`
(plusieurs appels indépendants en parallèle HTTP, `litellm.batch_completion`
— pas `mode_batch`, voir la mise en garde de
`fix_pipeline/multi_models/prompts_multi_models.md`).

Ce module NE décide PAS de la structure des clés de cache : chaque
appelant construit son propre dict `cache_key_fields` (et, pour les 4
tâches historiquement LiteLLM, son propre préfixe de fichier) — préservé
tel quel pour ne pas invalider un cache de traduction déjà payé (voir
Lot U3 du plan d'unification). `call()` ne fait que hasher ce dict et
gérer lecture/écriture/appel réseau autour.
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

from pipeline import config
from pipeline import llm_litellm_catgpt


class LLMError(RuntimeError):
    pass


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
    return parsed


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
