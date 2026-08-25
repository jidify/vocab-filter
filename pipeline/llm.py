"""Client minimal pour le LLM local (ollama), utilisé pour :

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
    model: str = config.OLLAMA_MODEL,
    system: str | None = None,
    timeout: float = 60.0,
) -> dict:
    """Appelle ollama en mode génération JSON, avec cache disque. Lève
    LLMError si l'endpoint est injoignable ou si la réponse n'est pas
    un JSON valide — l'appelant doit alors mettre le cas en révision
    plutôt que planter tout le batch."""

    cache_key = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "temp": config.LLM_TEMPERATURE},
        sort_keys=True,
    )
    cache_file = _cache_path(cache_key)
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system or "",
        "format": "json",
        "stream": False,
        "options": {"temperature": config.LLM_TEMPERATURE},
    }

    request = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"ollama injoignable ({config.OLLAMA_URL}): {exc}") from exc

    raw = body.get("response", "")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"réponse LLM non-JSON: {raw[:200]!r}") from exc

    cache_file.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    return parsed


def is_available() -> bool:
    try:
        req = urllib.request.Request(f"{config.OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False
