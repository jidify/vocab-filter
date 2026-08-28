"""Client JSON minimal pour Ollama ou CatGPT-Gateway, utilisé pour :

- S3 : idiome / phrasal_verb / semi_figé / littéral / incertain ;
- S5 : arbitrage de sens quand GlossBERT + preuve française sont
  ambigus ou en désaccord.

Stdlib uniquement (urllib), température 0, cache disque indexé par hash
du prompt+schéma -> reproductibilité et re-runs gratuits (proposition_1
§5.3 : "La température doit être fixée à zéro").
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request

from pipeline import config


class LLMError(RuntimeError):
    pass


def _cache_path(cache_key: str):
    config.ensure_out_dir()
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return config.CACHE_DIR / f"{digest}.json"


def call_json(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    timeout: float = 60.0,
    cache_metadata: dict | None = None,
) -> dict:
    """Appelle le backend configuré en mode JSON, avec cache disque. Lève
    LLMError si l'endpoint est injoignable ou si la réponse n'est pas
    un JSON valide — l'appelant doit alors mettre le cas en révision
    plutôt que planter tout le batch."""

    backend = config.LLM_BACKEND
    model = model or config.llm_model()
    effective_timeout = config.CATGPT_TIMEOUT if backend == "catgpt" and timeout == 60.0 else timeout
    cache_key = json.dumps(
        {"backend": backend, "model": model, "system": system, "prompt": prompt,
         "temp": config.LLM_TEMPERATURE, "cache_metadata": cache_metadata or {}},
        sort_keys=True,
    )
    cache_file = _cache_path(cache_key)
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if backend == "catgpt":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages, "stream": False,
                   "temperature": config.LLM_TEMPERATURE}
        url = f"{config.CATGPT_BASE_URL}/chat/completions"
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {config.CATGPT_API_TOKEN}"}
    elif backend == "ollama":
        payload = {"model": model, "prompt": prompt, "system": system or "",
                   "format": "json", "stream": False,
                   "options": {"temperature": config.LLM_TEMPERATURE}}
        url = f"{config.OLLAMA_URL}/api/generate"
        headers = {"Content-Type": "application/json"}
    else:
        raise LLMError(f"backend LLM inconnu : {backend!r}")

    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=effective_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"{backend} injoignable ({url}): {exc}") from exc

    try:
        raw = (body["choices"][0]["message"]["content"] if backend == "catgpt"
               else body.get("response", ""))
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"réponse {backend} inattendue: {body!r}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"réponse LLM non-JSON: {raw[:200]!r}") from exc

    cache_file.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    return parsed


def is_available() -> bool:
    try:
        if config.LLM_BACKEND == "catgpt":
            req = urllib.request.Request(
                f"{config.CATGPT_BASE_URL}/models",
                headers={"Authorization": f"Bearer {config.CATGPT_API_TOKEN}"},
            )
        else:
            req = urllib.request.Request(f"{config.OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
