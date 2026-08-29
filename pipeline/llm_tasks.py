"""Registre et résolution de configuration des tâches LLM.

Ce module ne route aucun appel LLM. Il décrit seulement le modèle et le mode
de prompt effectifs de chaque tâche (lot multi-items, pas parallélisme HTTP).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Mapping

from . import config


ALLOWED_PROVIDERS = frozenset({"ollama", "openai", "catgpt"})
PROVIDER_ALIASES = {"chatgpt": "catgpt"}


class TaskConfigError(ValueError):
    """Configuration invalide d'une tâche LLM."""


@dataclass(frozen=True)
class TaskDescriptor:
    task_id: str
    batch_allowed: bool
    default_model: str
    default_mode_batch: bool
    default_batch_size: int
    global_model_fallback: bool = False


@dataclass(frozen=True)
class TaskLlmConfig:
    task_id: str
    batch_allowed: bool
    model: str
    provider: str
    bare_model: str
    mode_batch: bool
    batch_size: int


def _descriptor(
    task_id: str,
    *,
    batch_allowed: bool,
    model: str,
    mode_batch: bool,
    batch_size: int,
    global_model_fallback: bool = False,
) -> TaskDescriptor:
    return TaskDescriptor(
        task_id=task_id,
        batch_allowed=batch_allowed,
        default_model=model,
        default_mode_batch=mode_batch,
        default_batch_size=batch_size,
        global_model_fallback=global_model_fallback,
    )


_TASKS = (
    _descriptor("S3-judge-occurrence", batch_allowed=True, model="ollama/mistral-small:24b", mode_batch=False, batch_size=1, global_model_fallback=True),
    _descriptor("S3-definition-cluster", batch_allowed=True, model="ollama/mistral-small:24b", mode_batch=False, batch_size=1, global_model_fallback=True),
    _descriptor("S5-arbitrate", batch_allowed=True, model="ollama/mistral-small:24b", mode_batch=False, batch_size=1, global_model_fallback=True),
    _descriptor("S6-translate-frontier", batch_allowed=True, model=config.SENSE_FR_FRONTIER_MODEL, mode_batch=True, batch_size=config.SENSE_FR_FRONTIER_BATCH_SIZE),
    _descriptor("S6-backtranslate", batch_allowed=True, model=config.SENSE_FR_FRONTIER_MODEL, mode_batch=True, batch_size=config.SENSE_FR_BACKTRANSLATE_BATCH_SIZE),
    _descriptor("S6-judge-dossier", batch_allowed=True, model=config.SENSE_FR_FRONTIER_MODEL, mode_batch=True, batch_size=config.SENSE_FR_JUDGE_BATCH_SIZE),
    _descriptor("S6-reassign", batch_allowed=True, model=config.SENSE_FR_FRONTIER_MODEL, mode_batch=True, batch_size=config.SENSE_FR_REASSIGN_BATCH_SIZE),
    _descriptor("S6-translate-local", batch_allowed=False, model="ollama/mistral-small:24b", mode_batch=False, batch_size=1, global_model_fallback=True),
    _descriptor("S6-backtranslate-local", batch_allowed=False, model="ollama/mistral-small:24b", mode_batch=False, batch_size=1, global_model_fallback=True),
)

TASK_REGISTRY: Mapping[str, TaskDescriptor] = MappingProxyType(
    {descriptor.task_id: descriptor for descriptor in _TASKS}
)


def _provider(value: str) -> str:
    normalized = value.strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


def _split_model(model: str, *, context: str) -> tuple[str, str]:
    provider, separator, bare_model = model.strip().partition("/")
    provider = _provider(provider)
    bare_model = bare_model.strip()
    if not separator or provider not in ALLOWED_PROVIDERS or not bare_model:
        allowed = ", ".join(sorted(ALLOWED_PROVIDERS))
        raise TaskConfigError(
            f"{context}: modèle attendu sous la forme provider/nom "
            f"(providers autorisés : {allowed})"
        )
    return provider, bare_model


def _global_model(descriptor: TaskDescriptor) -> str:
    backend_value = os.getenv("VOCAB_LLM_BACKEND")
    if backend_value is None:
        backend_value = os.getenv("PROVIDER")
    backend = _provider(backend_value if backend_value is not None else config.LLM_BACKEND)
    if backend not in {"ollama", "catgpt"}:
        raise TaskConfigError(
            f"{descriptor.task_id}: backend global inconnu {backend!r}; "
            "attendu ollama, catgpt ou l'alias chatgpt"
        )
    if backend == "catgpt":
        bare_model = os.getenv("CATGPT_MODEL", config.CATGPT_MODEL).strip()
    else:
        bare_model = os.getenv("OLLAMA_MODEL", config.OLLAMA_MODEL).strip()
    if not bare_model:
        raise TaskConfigError(f"{descriptor.task_id}: modèle global vide pour {backend}")
    return f"{backend}/{bare_model}"


def _env_key(task_id: str) -> str:
    return f"VOCAB_LLM_{task_id.replace('-', '_')}"


def _parse_bool(value: str, *, field: str, task_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise TaskConfigError(f"{task_id}: {field} doit valoir true ou false")


def _parse_override(raw: str, *, task_id: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in raw.split(";")]
    model = parts[0] if parts else ""
    if not model:
        raise TaskConfigError(f"{task_id}: modèle absent dans {_env_key(task_id)}")
    options: dict[str, str] = {}
    allowed = {"batch", "mode_batch", "batch_size"}
    for part in parts[1:]:
        key, separator, value = part.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if not separator or not key or not value:
            raise TaskConfigError(f"{task_id}: option invalide {part!r}")
        if key not in allowed:
            raise TaskConfigError(f"{task_id}: option inconnue {key!r}")
        canonical_key = "batch" if key == "mode_batch" else key
        if canonical_key in options:
            raise TaskConfigError(f"{task_id}: option dupliquée {canonical_key!r}")
        options[canonical_key] = value
    return model, options


def task_config(task_id: str) -> TaskLlmConfig:
    """Résout et valide la configuration effective de ``task_id``."""

    try:
        descriptor = TASK_REGISTRY[task_id]
    except KeyError as exc:
        raise TaskConfigError(f"tâche LLM inconnue : {task_id}") from exc

    model = (
        _global_model(descriptor)
        if descriptor.global_model_fallback
        else descriptor.default_model
    )
    mode_batch = descriptor.default_mode_batch
    batch_size = descriptor.default_batch_size

    raw_override = os.getenv(_env_key(task_id))
    if raw_override is not None:
        model, options = _parse_override(raw_override, task_id=task_id)
        if "batch" in options:
            mode_batch = _parse_bool(options["batch"], field="batch", task_id=task_id)
        if "batch_size" in options:
            try:
                batch_size = int(options["batch_size"])
            except ValueError as exc:
                raise TaskConfigError(f"{task_id}: batch_size doit être un entier") from exc
        if options.get("batch", "").strip().lower() == "true" and "batch_size" not in options:
            raise TaskConfigError(f"{task_id}: batch_size est requis quand batch=true")

    provider, bare_model = _split_model(model, context=task_id)
    if batch_size < 1:
        raise TaskConfigError(f"{task_id}: batch_size doit être >= 1")
    if mode_batch and not descriptor.batch_allowed:
        raise TaskConfigError(
            f"{task_id}: mode_batch=true interdit car batch_allowed=false"
        )
    return TaskLlmConfig(
        task_id=task_id,
        batch_allowed=descriptor.batch_allowed,
        model=f"{provider}/{bare_model}",
        provider=provider,
        bare_model=bare_model,
        mode_batch=mode_batch,
        batch_size=batch_size,
    )


def effective_batch_size(task: str | TaskLlmConfig) -> int:
    """Retourne 1 en mode unitaire, sinon la taille de lot validée."""

    resolved = task_config(task) if isinstance(task, str) else task
    return resolved.batch_size if resolved.mode_batch else 1


def use_batch_prompt(task: str | TaskLlmConfig, batch_size: int | None = None) -> bool:
    """Vrai si le chemin de prompt LOT doit être utilisé pour cette tâche :
    ``mode_batch`` actif ET taille effective (ou ``batch_size`` fourni
    explicitement par l'appelant, p.ex. un lot final réduit) >= 2.
    ``batch=true;batch_size=1`` — ou un appelant qui force ``batch_size=1``
    pour un chemin unitaire explicite (plan §0, §6) — bascule sur le
    chemin unitaire, jamais sur un prompt lot d'un seul item."""

    resolved = task_config(task) if isinstance(task, str) else task
    size = batch_size if batch_size is not None else effective_batch_size(resolved)
    return resolved.mode_batch and size >= 2
