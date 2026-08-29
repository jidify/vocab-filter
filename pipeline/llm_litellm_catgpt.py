"""Adaptateur minimal LiteLLM -> CatGPT-Gateway.

``catgpt`` n'est pas un provider natif de LiteLLM (voir
fix_pipeline/multi_models/report_multi_models.md §2.1) : poser
``catgpt/<modele>`` sur une des 4 tâches S6 routées par LiteLLM
(sense_fr_frontier, sense_fr_reassign, sense_fr_adjudicate Stage B/C)
passe la validation de configuration mais échouait jusqu'ici à l'appel
réel. Ce module comble UNIQUEMENT le strict nécessaire pour ces 4
tâches : envoyer un prompt, récupérer le texte de la réponse — pas de
streaming, pas d'async, pas d'embeddings, pas de tools.

Réutilise la mécanique HTTP déjà éprouvée de pipeline/llm.py pour le
backend "catgpt" (POST OpenAI-compatible sur CATGPT_BASE_URL, Bearer
CATGPT_API_TOKEN, ``response_format: {"type": "json_object"}`` — le
gateway ne sait pas faire de ``json_schema`` structuré).

Injection du schéma dans le prompt : les 4 tâches passent
``response_format=<PydanticModel>`` ; pour un modèle qui sait faire du
``json_schema`` (ex. openai/*), LiteLLM le convertit nativement et rien
ici n'y touche (``call_kwargs`` renvoie ``{}`` pour tout modèle qui n'est
pas ``catgpt/...``). Pour catgpt, ce JSON Schema est extrait de
``optional_params`` et rajouté en suffixe du message ``system`` avant
l'envoi — exactement le motif déjà utilisé à la main par les appelants
CatGPT existants (mwe_judge.PROMPT_SCHEMA, senses.ARBITRATION_TEMPLATE,
sense_fr.BACKTRANSLATE_TEMPLATE) : on rend le prompt complet, on ne fait
ni retry ni réparation de JSON ni validation ici — l'appelant garde la
responsabilité de ``Model.model_validate_json(content)``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import litellm
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from pipeline import config

_SCHEMA_SUFFIX = "\n\nRéponds en JSON strict conforme à ce schéma :\n{schema}"


def _inject_schema(messages: list[dict], optional_params: dict) -> list[dict]:
    """Renvoie une copie de ``messages`` où le schéma JSON demandé via
    ``response_format`` (converti par LiteLLM en json_schema) est rajouté
    en suffixe du message système — le gateway CatGPT ne sait produire
    que du JSON libre (``json_object``), pas du JSON contraint par
    schéma."""
    response_format = optional_params.get("response_format") or {}
    schema = (response_format.get("json_schema") or {}).get("schema")
    if not schema:
        return messages

    suffix = _SCHEMA_SUFFIX.format(schema=json.dumps(schema, ensure_ascii=False))
    messages = [dict(m) for m in messages]
    for m in messages:
        if m.get("role") == "system":
            m["content"] = (m.get("content") or "") + suffix
            return messages
    messages.insert(0, {"role": "system", "content": suffix.lstrip()})
    return messages


class _CatGptLLM(CustomLLM):
    """Handler CustomLLM minimal : une seule méthode réellement
    implémentée (``completion``, appel synchrone non streamé — le seul
    chemin emprunté par ce dépôt). Tout le reste (acompletion,
    streaming, astreaming, embeddings, images) reste celui de la classe
    de base, qui lève ``CustomLLMError`` — comportement voulu, jamais
    emprunté ici."""

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        optional_params: dict,
        **kwargs,
    ) -> ModelResponse:
        payload = {
            "model": model,
            "messages": _inject_schema(messages, optional_params),
            "stream": False,
            "temperature": config.LLM_TEMPERATURE,
            "response_format": {"type": "json_object"},
        }
        url = f"{config.CATGPT_BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.CATGPT_API_TOKEN}",
        }
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=config.CATGPT_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CustomLLMError(status_code=502, message=f"catgpt injoignable ({url}): {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CustomLLMError(status_code=502, message=f"réponse catgpt inattendue: {body!r}") from exc

        model_response.choices = [
            Choices(index=0, finish_reason="stop", message=Message(role="assistant", content=content))
        ]
        model_response.model = model
        model_response.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return model_response


_HANDLER = _CatGptLLM()


def register() -> None:
    """Enregistre le provider "catgpt" auprès de LiteLLM. Idempotent —
    sans effet si déjà fait (par ce process ou un appel précédent)."""
    if any(item.get("provider") == "catgpt" for item in litellm.custom_provider_map):
        return
    litellm.custom_provider_map.append({"provider": "catgpt", "custom_handler": _HANDLER})


def call_kwargs(model: str) -> dict:
    """Kwargs supplémentaires à passer à ``litellm.completion``/
    ``batch_completion`` pour que ``model`` fonctionne réellement.

    Pour tout modèle non "catgpt/..." (openai/*, ollama/*...), renvoie
    ``{}`` sans effet de bord : ces providers gèrent nativement
    ``reasoning_effort`` et ``response_format=<PydanticModel>``
    (json_schema natif pour openai) — rien ici ne doit y toucher.

    Pour "catgpt/...", enregistre le provider (si besoin) et autorise
    explicitement ``reasoning_effort``, que LiteLLM rejetterait sinon
    AVANT même d'atteindre le handler (``UnsupportedParamsError`` côté
    validation générique des paramètres OpenAI par provider)."""
    if not model.startswith("catgpt/"):
        return {}
    register()
    return {"allowed_openai_params": ["reasoning_effort"]}
