"""S3 — juger chaque occurrence MWE avant toute synthèse de type.

Le contexte, et non les trois premières occurrences d'un groupe, porte la
décision.  Chaque hypothèse occurrence+canon reçoit donc un verdict complet ;
la ligne de type dans ``mwe_decisions.jsonl`` n'est qu'une synthèse d'audit et
ne peut jamais réserver un span à la place de la décision d'occurrence.
"""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from collections import defaultdict

from nltk.corpus import wordnet as nwn

from pipeline import atomic, config, llm_client, mwe_stores
from pipeline.llm_tasks import TaskConfigError, TaskLlmConfig, effective_batch_size, task_config, use_batch_prompt
from pipeline.prompt_variants import EVIDENCE_TAGS, PromptOverride, PromptVariantError
from pipeline.prompt_variants import render as render_prompt

VALID_LABELS = {"idiome", "phrasal_verb", "semi_fige", "littéral", "incertain"}

# Lot 3 : préfixes distinguant un "incertain" GENUINEMENT décidé par le
# modèle (confiance auto-déclarée, potentiellement 0.0 elle-même) d'un
# "incertain" de repli faute d'avoir pu interroger le modèle du tout
# (ollama injoignable — vérifié en pratique : LAN intermittent, voir
# is_llm_failure) ou d'une réponse illisible. Seule la première catégorie
# est une DÉCISION — la mettre dans un magasin permanent au même titre
# qu'une vraie décision figerait une panne réseau transitoire en verdict
# définitif ("wake up" jugé "incertain" pour toujours parce qu'ollama avait
# un timeout ce jour-là), à l'exact opposé du principe d'abstention du plan
# (point 9 : abstention = pas de réservation, jamais suppression sur
# hypothèse incertaine — une panne n'est pas une hypothèse).
_LLM_FAILURE_REASON_PREFIXES = ("LLM indisponible:", "réponse LLM invalide:")


def is_llm_failure(decision: dict) -> bool:
    return decision["reason"].startswith(_LLM_FAILURE_REASON_PREFIXES)

SYSTEM_PROMPT = (
    "Tu es linguiste, spécialiste de l'anglais lexicalisé pour l'enseignement "
    "à des apprenants francophones avancés. On te donne une expression "
    "candidate détectée automatiquement dans une pièce de théâtre, avec "
    "jusqu'à 3 occurrences réelles en contexte. Tu dois juger si l'expression "
    "forme, dans CES occurrences, une unité lexicale à part entière — dont le "
    "sens ne se déduit pas simplement de ses mots pris séparément — ou si "
    "c'est une combinaison libre/compositionnelle (les mots gardent leur sens "
    "habituel, juste juxtaposés)."
)

PROMPT_TEMPLATE = """Expression candidate : "{idiom}"

Occurrences dans le texte :
{examples}

Classe cette expression dans EXACTEMENT une de ces catégories :
- "idiome" : sens opaque, non déductible des mots (ex: "wing it", "get away with")
- "phrasal_verb" : verbe + particule à sens spécialisé (ex: "figure out", "call up")
- "semi_fige" : construction récurrente à sens partiellement prévisible mais notable
  pour un apprenant (ex: "care package")
- "littéral" : combinaison libre, chaque mot garde son sens ordinaire, PAS une unité
  à enseigner comme telle (ex: "go to [the store]", "I do [want that]", "know someone")
- "incertain" : tu ne peux pas trancher avec ce contexte
"""

# Lot 4 (point G) : ajouté au prompt ci-dessus, jamais un appel LLM séparé,
# uniquement quand plusieurs synsets WordNet candidats existent pour
# l'idiome (voir wordnet_synset_candidates) — un seul candidat est utilisé
# directement sans solliciter le modèle (même court-circuit que
# senses.py::analyze_occurrence pour GlossBERT). WordNet n'est ici qu'une
# source de glose : la clé d'identité de l'unité reste toujours
# `mwe:{sense_id}` ; le label reste un type d'unité, jamais son identité.
WORDNET_SENSE_BLOCK = """
Cette expression a plusieurs sens WordNet candidats. Si l'un d'eux correspond
au sens réellement employé dans ces occurrences, indique son identifiant dans
"wordnet_sense_id" ; sinon indique null.
{candidates}
"""

PROMPT_SCHEMA = """
Réponds en JSON strict avec ce schéma :
{"label": "<une des 5 catégories>", "confidence": <0.0-1.0>, "reason": "<1 phrase en français>"}
"""

PROMPT_SCHEMA_WITH_WORDNET = """
Réponds en JSON strict avec ce schéma :
{"label": "<une des 5 catégories>", "confidence": <0.0-1.0>, "reason": "<1 phrase en français>", "wordnet_sense_id": "<identifiant du sens WordNet choisi ci-dessus, ou null>"}
"""


def format_examples(occurrences: list[dict], segments_by_idx: dict) -> str:
    lines = []
    for occ in occurrences[:3]:
        seg = segments_by_idx.get(occ["segment_idx"])
        text = seg.en if seg else occ["surface"]
        lines.append(f'- "{text}" (span détecté : "{occ["surface"]}")')
    return "\n".join(lines)


def wordnet_synset_candidates(idiom: str) -> list:
    """Synsets WordNet dont un lemme correspond exactement à `idiom` (même
    logique de filtrage que senses.py::get_synsets, mais sans filtre de
    POS : un idiome confirmé n'a pas de wn_pos assigné en amont — voir
    plan Lot 4 point G). nltk indexe les MWE avec underscore, jamais
    espace (nwn.synsets('wake up') == [] mais nwn.synsets('wake_up') ==
    [awaken.v.01, wake_up.v.02])."""
    results = []
    for synset in nwn.synsets(idiom.replace(" ", "_")):
        for lemma in synset.lemmas():
            if lemma.name().replace("_", " ").casefold() == idiom.casefold():
                results.append(synset)
                break
    return results


def judge_type(
    idiom: str, occurrences: list[dict], segments_by_idx: dict,
    wordnet_candidates: list | None = None,
) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        idiom=idiom,
        examples=format_examples(occurrences, segments_by_idx),
    )
    if wordnet_candidates:
        gloss_lines = "\n".join(
            f"- {s.name()}: {s.definition()}" for s in wordnet_candidates
        )
        prompt += WORDNET_SENSE_BLOCK.format(candidates=gloss_lines)
        prompt += PROMPT_SCHEMA_WITH_WORDNET
    else:
        prompt += PROMPT_SCHEMA

    task = task_config("S3-judge-type")
    try:
        result = llm_client.call(
            model=task.model, system=SYSTEM_PROMPT, prompt=prompt, timeout=120,
            cache_key_fields=llm_client.build_cache_key(
                model=task.model, system=SYSTEM_PROMPT, prompt=prompt,
                extra={"task_id": task.task_id, "mode_batch": False, "batch_size": 1},
            ),
        )
    except llm_client.LLMError as exc:
        return {"label": "incertain", "confidence": 0.0, "reason": f"LLM indisponible: {exc}"}

    label = result.get("label")
    if label not in VALID_LABELS:
        return {"label": "incertain", "confidence": 0.0,
                "reason": f"réponse LLM invalide: {result!r}"}

    decision = {
        "label": label,
        "confidence": float(result.get("confidence", 0.0)),
        "reason": result.get("reason", ""),
    }
    if wordnet_candidates:
        valid_ids = {s.name() for s in wordnet_candidates}
        selected = result.get("wordnet_sense_id")
        decision["wordnet_sense_id"] = selected if selected in valid_ids else None
    return decision


# ============================================================
# Escalade occurrence par occurrence (point 14 / point C du plan) — SEULS
# les candidats que le garde-fou directionnel du détecteur VPC signale
# comme dépendants du contexte (mwe.py::load_vpc_candidates,
# `directional_context_dependent`) passent ici. La question n'est plus "cette
# expression est-elle lexicalisée en général" (judge_type) mais "l'est-elle
# DANS CETTE PHRASE précise" — ex. "walk up the stairs" (littéral, monter les
# marches) vs "walk up to someone" (phrasal_verb, s'approcher de).
# ============================================================

OCC_SYSTEM_PROMPT = (
    "Tu es linguiste, spécialiste de l'anglais lexicalisé pour l'enseignement "
    "à des apprenants francophones avancés. Un détecteur syntaxique a repéré, "
    "dans CETTE phrase précise une expression candidate, issue de l'une de "
    "plusieurs sources possibles. Elle peut être lexicalisée dans ce contexte "
    "ou n'être qu'une combinaison transparente. Tu dois juger CETTE occurrence "
    "précise, jamais l'expression en général. Une valeur de confiance déclarée "
    "n'est pas une justification : fournis une paraphrase contextuelle et des "
    "indices linguistiques vérifiables."
)

OCC_PROMPT_TEMPLATE = """Expression candidate : "{idiom}"

Phrase : "{sentence}"
Span détecté : "{surface}"

Source(s) de candidature : {source}.
Indice syntaxique éventuel : {vpc_decision_reason}.

Classe CETTE occurrence dans EXACTEMENT une de ces catégories :
- "idiome" : sens conventionnel non compositionnel ; notamment, la lecture
  littérale contredit le sens communiqué ou inverse sa polarité
- "phrasal_verb" : verbe + particule à sens spécialisé, DANS CETTE PHRASE
- "semi_fige" : collocation contrainte ou formulation conventionnelle dont
  le sens global reste compositionnel et compatible avec les mots
- "littéral" : ici, direction/complément littéral — chaque mot garde son sens ordinaire
- "incertain" : tu ne peux pas trancher avec ce contexte

En cas de conflit, la non-compositionnalité prime sur le caractère seulement figé.

Propose aussi le canon, le POS lexical (NOUN, VERB, ADJ, ADV ou OTHER), une
paraphrase anglaise de CE sens en contexte, et au moins un indice linguistique
observable (substitution impossible, sens spécialisé, structure libre, etc.).

Réponds en JSON strict avec ce schéma :
{{"label": "<une des 5 catégories>", "canonical_form": "<canon>", "pos": "<POS>", "contextual_paraphrase": "<paraphrase anglaise>", "confidence": <0.0-1.0>, "evidence": ["<indice observable>"], "wordnet_sense_id": "<sens WordNet exact si fourni et applicable, sinon null>", "reason": "<1 phrase en français>"}}
"""

VALID_POS = {"NOUN", "VERB", "ADJ", "ADV", "OTHER"}
S3_PROMPT_VERSION = "s3-judge-prompt-5"
S3_DECISION_SCHEMA_VERSION = "s3-decision-schema-3"


def _occurrence_payload(idiom: str, occ: dict, segments_by_idx: dict) -> dict:
    """Entrée SÉMANTIQUE d'une occurrence — tout ce qui peut faire changer
    le verdict, jamais le texte du prompt rendu. Double usage : signature du
    magasin métier (`occurrence_context_signature`, ci-dessous) ET payload
    du magasin LLM unitaire (`pipeline/llm_store.py`, voir `run()` plus
    bas) — les deux doivent invalider ensemble si la phrase, le span ou
    l'indice syntaxique changent.

    Corrigé (plan de décorrélation lot/stockage) : utilisait
    ``getattr(segment, "text", "")``, qui renvoyait toujours ``""`` —
    ``Segment`` (pipeline/corpus.py) n'a pas de champ ``text``, seulement
    ``en``. Corriger une phrase du livre n'a donc jamais changé cette
    signature avant ce lot."""
    segment = segments_by_idx.get(occ.get("segment_idx"))
    sentence = segment.en if segment is not None else ""
    return {
        "canonical_form": idiom.casefold().strip(), "surface": occ.get("surface") or "",
        "sentence": sentence, "source": occ.get("source"),
        "member_char_spans": occ.get("member_char_spans"),
        "vpc_decision_reason": occ.get("vpc_decision_reason"),
    }


def occurrence_context_signature(idiom: str, occ: dict, segments_by_idx: dict) -> str:
    """Signature stable de toute donnée susceptible de changer le verdict —
    utilisée par le magasin métier (`occurrence_store_key` ci-dessous)."""
    raw = json.dumps(_occurrence_payload(idiom, occ, segments_by_idx),
                     ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def occurrence_store_key(idiom: str, occurrence_id: str, *, model: str | None = None,
                         backend: str | None = None, context_signature: str = "") -> str:
    """Clé protocolaire : les anciennes décisions sont auditables mais
    illisibles. Ne porte plus ``mode_batch``/``batch_size`` (plan de
    décorrélation lot/stockage) — changer la taille de lot ou le mode
    unitaire/lot n'invalide plus ce magasin métier, exactement comme il
    n'invalide plus le magasin LLM unitaire en dessous (llm_store.py)."""
    payload = {
        "prompt_version": S3_PROMPT_VERSION,
        "backend": backend or config.LLM_BACKEND,
        "model": model or config.llm_model(),
        "schema_version": S3_DECISION_SCHEMA_VERSION,
        "canonical_form": idiom.casefold().strip(), "occurrence_id": occurrence_id,
        "context_signature": context_signature,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"s3:{S3_PROMPT_VERSION}:{S3_DECISION_SCHEMA_VERSION}:{digest}"


CUSTOM_SENSE_VERSION = "mwe-custom-v1"
_GLOSS_STOPWORDS = {"a", "an", "the", "to", "of", "one", "oneself", "someone", "something"}


def normalize_sense_gloss(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(token for token in tokens if token not in _GLOSS_STOPWORDS)


def paraphrases_compatible(left: str, right: str) -> bool:
    """Compatibilité conservatrice et reproductible, auditée par ses glosses."""
    a, b = set(normalize_sense_gloss(left).split()), set(normalize_sense_gloss(right).split())
    if not a or not b:
        return False
    return a == b or len(a & b) / len(a | b) >= 0.6


def custom_sense_id(canonical_form: str, pos: str, cluster_gloss: str) -> str:
    normalized = "\t".join((canonical_form.casefold().strip(), pos.upper(),
                             normalize_sense_gloss(cluster_gloss)))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{CUSTOM_SENSE_VERSION}:{digest}"


def exact_dbnary_sense_id(canonical_form: str, pos: str, gloss: str) -> str | None:
    """Retourne une clé DBnary seulement si sa glose coïncide exactement."""
    try:
        from pipeline import lex_bilingual
        wn_pos = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}.get(pos.upper())
        expected = normalize_sense_gloss(gloss)
        for key, entry in lex_bilingual.dbnary_senses_for_lemma(canonical_form, wn_pos):
            if expected and normalize_sense_gloss(entry.get("gloss_en") or "") == expected:
                return f"dbnary:{key}"
    except (OSError, ValueError, KeyError):
        pass
    return None


def assign_sense_ids(records: list[dict]) -> None:
    """Regroupe après jugement sur canon+POS+sens compatible, en place."""
    clusters: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in records:
        for occ in entry["occurrences"]:
            decision = occ["occurrence_decision"]
            if decision["label"] not in LEXICALIZED_LABELS:
                continue
            canonical = decision["canonical_form"].casefold().strip()
            pos = decision["pos"].upper()
            wn_id = decision.get("wordnet_sense_id")
            if wn_id:
                decision["sense_id"] = wn_id
                decision["sense_id_source"] = "wordnet"
                continue
            gloss = decision["contextual_paraphrase"]
            dbnary_id = exact_dbnary_sense_id(canonical, pos, gloss)
            if dbnary_id:
                decision["sense_id"] = dbnary_id
                decision["sense_id_source"] = "dbnary"
                continue
            bucket = clusters[(canonical, pos)]
            cluster = next((c for c in bucket if paraphrases_compatible(gloss, c["gloss"])), None)
            if cluster is None:
                cluster = {"gloss": gloss, "sense_id": custom_sense_id(canonical, pos, gloss)}
                bucket.append(cluster)
            decision["sense_id"] = cluster["sense_id"]
            decision["sense_id_source"] = CUSTOM_SENSE_VERSION


DEFINITION_SYSTEM_PROMPT = (
    "Tu désambiguïses la définition d'une expression anglaise dans un livre. "
    "Compare chaque définition candidate à TOUTES les occurrences du cluster. "
    "Ne choisis une entrée que si elle décrit exactement le même sens dans "
    "chaque occurrence ; sinon rédige une définition anglaise custom, précise "
    "et compatible avec toutes les occurrences."
)

DEFINITION_PROMPT_TEMPLATE = """Expression : {canonical_form}
POS : {pos}

Occurrences du même cluster :
{occurrences}

Définitions candidates (l'ordre n'indique aucune préférence) :
{candidates}

Réponds en JSON strict :
{{"candidate_id": "<identifiant exact, ou null>", "custom_definition": "<définition si aucun candidat exact, sinon chaîne vide>", "occurrence_checks": [{{"occurrence_id": "<id>", "contradicts": false}}], "reason": "<justification brève>"}}
"""

DEFINITION_BATCH_SYSTEM_PROMPT = DEFINITION_SYSTEM_PROMPT + (
    " Tu reçois plusieurs clusters indépendants : compare uniquement les "
    "occurrences et candidats du cluster portant le même cluster_id."
)

DEFINITION_BATCH_PROMPT_TEMPLATE = """Choisis séparément une définition pour les {count} clusters suivants.

{items}

Réponds en JSON strict avec ce schéma :
{{"decisions":[{{"cluster_id":"<id exact>","candidate_id":"<identifiant exact ou null>","custom_definition":"<définition si aucun candidat exact, sinon chaîne vide>","occurrence_checks":[{{"occurrence_id":"<id>","contradicts":false}}],"reason":"<justification brève>"}}]}}
Il doit y avoir exactement une décision par cluster_id, dans le même ordre.
"""


def definition_candidates(canonical_form: str, pos: str) -> list[dict]:
    """Inventaire exhaustif et dédupliqué des glosses disponibles."""
    from pipeline import lex_bilingual, mwe

    candidates: list[dict] = []
    for index, sense in enumerate(mwe.get_idiom_senses(canonical_form)):
        candidates.append({"candidate_id": f"idiomatch:{canonical_form}:{index}",
                           "definition": sense["content"], "source": "idiomatch"})
    for synset in wordnet_synset_candidates(canonical_form):
        candidates.append({"candidate_id": synset.name(), "definition": synset.definition(),
                           "source": "wordnet"})
    wn_pos = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}.get(pos.upper())
    for key, entry in lex_bilingual.dbnary_senses_for_lemma(canonical_form, wn_pos):
        if entry.get("gloss_en"):
            candidates.append({"candidate_id": f"dbnary:{key}",
                               "definition": entry["gloss_en"], "source": "dbnary"})
    unique, seen = [], set()
    for candidate in candidates:
        normalized = normalize_sense_gloss(candidate["definition"])
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def _definition_request(canonical_form: str, pos: str, occurrences: list[dict],
                        segments_by_idx: dict, *, cluster_id: str | None = None) -> dict:
    candidates = definition_candidates(canonical_form, pos)
    occurrence_lines, occurrence_ids, payload_occurrences = [], [], []
    for occ in occurrences:
        occurrence_id = occ["occurrence_id"]
        occurrence_ids.append(occurrence_id)
        segment = segments_by_idx.get(occ.get("segment_idx"))
        # Même bug corrigé qu'_occurrence_payload ci-dessus : `Segment` n'a
        # pas de champ `text` (seulement `en`) — ce prompt envoyait donc la
        # phrase vide au modèle pour CHAQUE occurrence depuis toujours.
        text = segment.en if segment is not None else ""
        paraphrase = occ["occurrence_decision"].get("contextual_paraphrase", "")
        occurrence_lines.append(f'- {occurrence_id}: "{text}" | paraphrase: {paraphrase}')
        payload_occurrences.append({"occurrence_id": occurrence_id, "sentence": text, "paraphrase": paraphrase})
    candidate_lines = [f'- {c["candidate_id"]}: {c["definition"]}' for c in candidates]
    prompt = DEFINITION_PROMPT_TEMPLATE.format(
        canonical_form=canonical_form, pos=pos, occurrences="\n".join(occurrence_lines),
        candidates="\n".join(candidate_lines) or "- (aucune entrée disponible)",
    )
    cluster_id = cluster_id or f"{canonical_form}|{pos}|unit"
    return {
        "cluster_id": cluster_id,
        "canonical_form": canonical_form, "pos": pos, "occurrences": occurrences,
        "candidates": candidates, "occurrence_ids": occurrence_ids, "prompt": prompt,
        # Entrée SÉMANTIQUE — payload du magasin LLM unitaire
        # (pipeline/llm_store.py), jamais le texte du prompt rendu. Porté ici
        # (plutôt que recalculé depuis segments_by_idx en aval) car
        # choose_cluster_definitions_batch n'a historiquement pas accès aux
        # segments — seulement à ce dict déjà construit.
        "payload": {
            "cluster_id": cluster_id, "canonical_form": canonical_form, "pos": pos,
            "occurrences": payload_occurrences,
            "candidates": [{"candidate_id": c["candidate_id"], "definition": c["definition"]}
                          for c in candidates],
        },
    }


def _definition_selection(request: dict, result: object) -> dict:
    result = result if isinstance(result, dict) else {}
    candidates = request["candidates"]
    occurrence_ids = request["occurrence_ids"]

    by_id = {c["candidate_id"]: c for c in candidates}
    candidate_id = result.get("candidate_id")
    custom = str(result.get("custom_definition") or "").strip()
    checks = {c.get("occurrence_id"): bool(c.get("contradicts"))
              for c in result.get("occurrence_checks", []) if isinstance(c, dict)}
    checks_complete = set(checks) == set(occurrence_ids) and not any(checks.values())
    if candidate_id in by_id and checks_complete:
        chosen = by_id[candidate_id]
        return {"definition_en": chosen["definition"], "definition_source": chosen["source"],
                "definition_candidate_id": candidate_id, "definition_needs_review": False}
    if custom and checks_complete:
        return {"definition_en": custom, "definition_source": "custom",
                "definition_candidate_id": None, "definition_needs_review": False}
    fallback = next((o["occurrence_decision"].get("contextual_paraphrase", "")
                     for o in request["occurrences"]
                     if o["occurrence_decision"].get("contextual_paraphrase")),
                    request["canonical_form"])
    return {"definition_en": fallback, "definition_source": "custom",
            "definition_candidate_id": None, "definition_needs_review": True}


def _choose_cluster_definitions(
    requests: list[dict], *, task: TaskLlmConfig, model: str,
    batch_size: int, mode_batch: bool,
) -> dict[str, dict]:
    """Cœur commun de `choose_cluster_definition`/`choose_cluster_definitions_batch`
    /`assign_cluster_definitions` — même motif que `_judge_occurrence_units`
    pour S3-judge-occurrence : appelle en LOT, stocke en UNITAIRE. Renvoie
    TOUJOURS une entrée par cluster_id demandé (succès ou repli via
    `_definition_selection(request, {})` — identique au comportement d'une
    panne LLM avant ce lot)."""
    if not requests:
        return {}
    units = [
        llm_client.Unit(unit_id=request["cluster_id"], payload=request["payload"], data=request)
        for request in requests
    ]

    def render_unit(unit: "llm_client.Unit") -> tuple[str, str]:
        return DEFINITION_SYSTEM_PROMPT, unit.data["prompt"]

    def render_batch(chunk: list["llm_client.Unit"]) -> tuple[str, str]:
        items = "\n\n".join(
            f"{index}. cluster_id={unit.unit_id!r}\n{unit.data['prompt']}"
            for index, unit in enumerate(chunk, start=1)
        )
        return DEFINITION_BATCH_SYSTEM_PROMPT, DEFINITION_BATCH_PROMPT_TEMPLATE.format(
            count=len(chunk), items=items,
        )

    def parse_unit(result: object, unit: "llm_client.Unit") -> dict:
        return _definition_selection(unit.data, result)

    def parse_batch(raw: object, chunk: list["llm_client.Unit"]) -> dict[str, dict]:
        by_id = llm_client.dedupe_batch_items(
            raw.get("decisions", []) if isinstance(raw, dict) else [], id_key="cluster_id",
        )
        return {
            unit.unit_id: _definition_selection(unit.data, by_id[unit.unit_id])
            for unit in chunk if unit.unit_id in by_id
        }

    fallbacks: dict[str, dict] = {}

    def on_failure(unit: "llm_client.Unit", _reason: object) -> None:
        fallbacks[unit.unit_id] = _definition_selection(unit.data, {})

    results = llm_client.run_units(
        units, task_id=task.task_id, model=model,
        protocol=f"{S3_PROMPT_VERSION}:{S3_DECISION_SCHEMA_VERSION}:definition",
        render_unit=render_unit, render_batch=render_batch,
        parse_unit=parse_unit, parse_batch=parse_batch,
        response_model_unit=None, response_model_batch=None,
        batch_size=batch_size, mode_batch=mode_batch, timeout=120,
        on_failure=on_failure,
    )
    results.update(fallbacks)
    return results


def choose_cluster_definition(canonical_form: str, pos: str, occurrences: list[dict],
                              segments_by_idx: dict, *, model: str | None = None) -> dict:
    """Chemin unitaire explicite de S3-definition-cluster."""
    task = task_config("S3-definition-cluster")
    resolved_model = model or task.model
    request = _definition_request(canonical_form, pos, occurrences, segments_by_idx)
    results = _choose_cluster_definitions([request], task=task, model=resolved_model,
                                          batch_size=1, mode_batch=False)
    return results[request["cluster_id"]]


def choose_cluster_definitions_batch(requests: list[dict], *, model: str | None = None) -> dict[str, dict]:
    """Chemin lot explicite de S3-definition-cluster, contrôlé par
    cluster_id — UN prompt de lot couvrant exactement ``requests``."""
    task = task_config("S3-definition-cluster")
    if not requests:
        return {}
    if len(requests) > effective_batch_size(task):
        raise ValueError(f"lot S3 définition de {len(requests)} clusters > batch_size={effective_batch_size(task)}")
    resolved_model = model or task.model
    return _choose_cluster_definitions(requests, task=task, model=resolved_model,
                                       batch_size=len(requests), mode_batch=True)


def assign_cluster_definitions(records: list[dict], segments_by_idx: dict) -> None:
    clusters: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for entry in records:
        for occ in entry["occurrences"]:
            decision = occ["occurrence_decision"]
            if decision.get("sense_id") and decision["label"] in LEXICALIZED_LABELS:
                key = (decision["canonical_form"].casefold().strip(), decision["pos"],
                       decision["sense_id"])
                clusters[key].append(occ)
    task = task_config("S3-definition-cluster")
    cluster_requests = [
        _definition_request(canonical_form, pos, occurrences, segments_by_idx,
                            cluster_id=f"{canonical_form}|{pos}|{sense_id}")
        for (canonical_form, pos, sense_id), occurrences in clusters.items()
    ]
    size = effective_batch_size(task)
    # Lot de décorrélation lot/stockage : plus de découpage manuel ici — un
    # seul appel à _choose_cluster_definitions couvre tous les clusters,
    # `llm_client.run_units` fait le découpage ET le cache unitaire.
    selections = _choose_cluster_definitions(
        cluster_requests, task=task, model=task.model,
        batch_size=size, mode_batch=use_batch_prompt(task, size),
    )
    for request in cluster_requests:
        selection = selections[request["cluster_id"]]
        for occ in request["occurrences"]:
            occ["occurrence_decision"].update(selection)


def _calibrate_occurrence(result: dict, label: str, *, schema_variant: str = "default") -> tuple[float, list[str]]:
    """Score reproductible, distinct de la confiance autodéclarée du modèle.

    Ce n'est pas une calibration statistique finale (Q0-2 la mesurera), mais
    un score calibrable : complétude des preuves et cohérence du verdict sont
    des features explicites. La confiance brute n'est jamais utilisée seule.

    ``reason`` ne compte plus dans ``complete`` (Lot U4 du plan d'unification,
    fix_pipeline/multi_models/report_multi_models.md §4bis) : c'était le seul
    des 4 champs jamais relu par aucune logique du pipeline — canonical_form,
    pos et contextual_paraphrase le sont tous (voir sense_id/clustering). Même
    formule pour toute variante de prompt, ``default`` et ``tags`` compris.

    Pour ``schema_variant="tags"`` (fix_pipeline/evaluate_s3_judges.py —
    variante s3-occurrence-tags), ``evidence`` doit être 1-2 étiquettes
    fermées (prompt_variants.EVIDENCE_TAGS) plutôt qu'un texte libre : une
    étiquette hors liste blanche ne compte pas comme preuve observable,
    sinon la variante compacte perdrait le garde-fou anti-confiance-
    autodéclarée que ce score existe pour imposer."""
    raw = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    evidence_raw = [str(x).strip() for x in result.get("evidence", []) if str(x).strip()]
    evidence = [tag for tag in evidence_raw if tag in EVIDENCE_TAGS] if schema_variant == "tags" else evidence_raw
    complete = all(str(result.get(k, "")).strip() for k in (
        "canonical_form", "pos", "contextual_paraphrase"
    ))
    features = [f"llm_raw={raw:.3f}"]
    if evidence:
        features.append("observable_evidence_present")
    if complete:
        features.append("required_fields_complete")
    # Sans preuve observable, aucune assurance élevée n'est admise. Un verdict
    # incertain reste par définition sous le seuil de réservation.
    score = raw * (0.65 + 0.20 * bool(evidence) + 0.15 * complete)
    if not evidence:
        score = min(score, 0.49)
    if label == "incertain":
        score = min(score, 0.49)
    return round(score, 4), features


OCC_BATCH_SYSTEM_PROMPT = OCC_SYSTEM_PROMPT + (
    " Tu reçois plusieurs occurrences indépendantes : ne les fusionne jamais et "
    "renvoie une décision séparée pour chaque occurrence_id."
)

OCC_BATCH_PROMPT_TEMPLATE = """Classe séparément les {count} occurrences suivantes.
Chaque occurrence doit être jugée seulement dans sa propre phrase, selon les
mêmes catégories et critères que le jugement unitaire.

{items}

Réponds en JSON strict avec ce schéma :
{{"decisions":[{{"occurrence_id":"<id exact>","label":"<une des 5 catégories>","canonical_form":"<canon>","pos":"<NOUN|VERB|ADJ|ADV|OTHER>","contextual_paraphrase":"<paraphrase anglaise>","confidence":<0.0-1.0>,"evidence":["<indice observable>"],"wordnet_sense_id":"<sens WordNet exact ou null>","reason":"<1 phrase en français>"}}]}}
Il doit y avoir exactement une décision par occurrence_id, dans le même ordre.
"""


def _occurrence_prompt(idiom: str, occ: dict, segments_by_idx: dict,
                       *, custom_prompt: PromptOverride | None = None) -> tuple[str, list]:
    """``custom_prompt.user_template``, s'il est fourni, remplace
    ``OCC_PROMPT_TEMPLATE`` — champs disponibles : ``idiom``/``canonical_form``
    (alias), ``sentence``/``context`` (alias), ``surface``, ``source``,
    ``vpc_decision_reason`` (voir prompt_variants.render, placeholder inconnu
    -> TaskConfigError, jamais un KeyError nu)."""
    seg = segments_by_idx.get(occ["segment_idx"])
    sentence = seg.en if seg else occ["surface"]
    fields = {
        "idiom": idiom, "canonical_form": idiom,
        "sentence": sentence, "context": sentence,
        "surface": occ["surface"],
        "source": occ.get("source", "inconnue"),
        "vpc_decision_reason": occ.get("vpc_decision_reason") or "voir le détecteur VPC",
    }
    if custom_prompt is not None and custom_prompt.user_template is not None:
        try:
            prompt = render_prompt(custom_prompt.user_template, fields)
        except PromptVariantError as exc:
            raise TaskConfigError(f"S3-judge-occurrence: prompt personnalisé invalide : {exc}") from exc
    else:
        prompt = OCC_PROMPT_TEMPLATE.format(**fields)
    wn_candidates = wordnet_synset_candidates(idiom)
    if wn_candidates:
        prompt += "\nSens WordNet autorisés (choisir seulement une correspondance exacte) :\n"
        prompt += "\n".join(f"- {s.name()}: {s.definition()}" for s in wn_candidates)
    return prompt, wn_candidates


def _occurrence_failure(idiom: str, reason: str, *, invalid: bool = False) -> dict:
    return {
        "label": "incertain", "verdict": "incertain",
        "canonical_form": idiom, "pos": "OTHER",
        "contextual_paraphrase": "", "model_confidence": 0.0,
        "confidence": 0.0,
        "confidence_features": ["invalid_response" if invalid else "llm_failure"],
        "evidence": [], "reason": reason,
    }


def _normalize_occurrence_result(idiom: str, result: object, wn_candidates: list,
                                 *, schema_variant: str = "default") -> dict:
    if not isinstance(result, dict) or result.get("label") not in VALID_LABELS:
        return _occurrence_failure(idiom, f"réponse LLM invalide: {result!r}", invalid=True)

    label = result["label"]
    canonical = str(result.get("canonical_form") or idiom).strip()
    pos = str(result.get("pos") or "OTHER").upper()
    if pos not in VALID_POS:
        pos = "OTHER"
    try:
        model_confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except (TypeError, ValueError):
        model_confidence = 0.0
    calibrated_input = {**result, "confidence": model_confidence}
    confidence, confidence_features = _calibrate_occurrence(calibrated_input, label, schema_variant=schema_variant)
    lexicalized = label in {"idiome", "phrasal_verb", "semi_fige"}
    valid_wn_ids = {s.name() for s in wn_candidates}
    selected_wn_id = result.get("wordnet_sense_id")
    evidence_raw = [str(x).strip() for x in result.get("evidence", []) if str(x).strip()]
    evidence = [tag for tag in evidence_raw if tag in EVIDENCE_TAGS] if schema_variant == "tags" else evidence_raw
    return {
        "label": label,
        "verdict": "lexicalisé" if lexicalized else label,
        "canonical_form": canonical,
        "pos": pos,
        "contextual_paraphrase": str(result.get("contextual_paraphrase") or "").strip(),
        "model_confidence": model_confidence,
        "confidence": confidence,
        "confidence_features": confidence_features,
        "evidence": evidence,
        "wordnet_sense_id": selected_wn_id if selected_wn_id in valid_wn_ids else None,
        "reason": result.get("reason", ""),
    }


def _occurrence_protocol(schema_variant: str) -> str:
    return f"{S3_PROMPT_VERSION}:{S3_DECISION_SCHEMA_VERSION}:occurrence:{schema_variant}"


def _occurrence_unit(idiom: str, occ: dict, segments_by_idx: dict,
                     *, custom_prompt: PromptOverride | None) -> "llm_client.Unit":
    """Une Unit S3-judge-occurrence : corps de prompt et candidats WordNet
    précalculés une seule fois (``unit.data``), réutilisés tels quels que
    cette occurrence finisse seule dans une tranche unitaire ou concaténée
    dans une tranche en lot — jamais recalculés par tranche."""
    prompt_body, wn_candidates = _occurrence_prompt(idiom, occ, segments_by_idx, custom_prompt=custom_prompt)
    return llm_client.Unit(
        unit_id=occ["occurrence_id"],
        payload=_occurrence_payload(idiom, occ, segments_by_idx),
        data=(idiom, wn_candidates, prompt_body),
    )


def _judge_occurrence_units(
    units: list["llm_client.Unit"], *, task: TaskLlmConfig, model: str,
    batch_size: int, mode_batch: bool,
) -> dict[str, dict]:
    """Cœur commun de `judge_occurrence`/`judge_occurrences_batch`/la boucle
    de `run()` — appelle en LOT (``batch_size``), stocke en UNITAIRE
    (pipeline/llm_store.py, via `llm_client.run_units`). Renvoie TOUJOURS une
    entrée par unité demandée (succès ou repli `_occurrence_failure` — jamais
    de clé manquante), pour ne pas changer le contrat des trois appelants."""
    if not units:
        return {}
    custom = task.custom_prompt
    schema_variant = custom.schema_variant if custom else "default"
    system = custom.system if custom and custom.system else OCC_SYSTEM_PROMPT
    batch_template = custom.batch_template if custom and custom.batch_template else OCC_BATCH_PROMPT_TEMPLATE
    batch_system = custom.batch_system if custom and custom.batch_system else OCC_BATCH_SYSTEM_PROMPT

    def render_unit(unit: "llm_client.Unit") -> tuple[str, str]:
        _idiom, _wn, prompt_body = unit.data
        return system, prompt_body

    def render_batch(chunk: list["llm_client.Unit"]) -> tuple[str, str]:
        numbered = "\n\n".join(
            f"{index}. occurrence_id={unit.unit_id!r}\n{unit.data[2]}"
            for index, unit in enumerate(chunk, start=1)
        )
        try:
            prompt = render_prompt(batch_template, {"count": len(chunk), "items": numbered})
        except PromptVariantError as exc:
            raise TaskConfigError(f"S3-judge-occurrence: prompt personnalisé invalide : {exc}") from exc
        return batch_system, prompt

    def parse_unit(result: object, unit: "llm_client.Unit") -> dict:
        idiom, wn_candidates, _prompt = unit.data
        return _normalize_occurrence_result(idiom, result, wn_candidates, schema_variant=schema_variant)

    def parse_batch(raw: object, chunk: list["llm_client.Unit"]) -> dict[str, dict]:
        by_id = llm_client.dedupe_batch_items(
            raw.get("decisions", []) if isinstance(raw, dict) else [], id_key="occurrence_id",
        )
        decisions: dict[str, dict] = {}
        for unit in chunk:
            item = by_id.get(unit.unit_id)
            if item is None:
                continue
            idiom, wn_candidates, _prompt = unit.data
            decisions[unit.unit_id] = _normalize_occurrence_result(
                idiom, item, wn_candidates, schema_variant=schema_variant,
            )
        return decisions

    failures: dict[str, dict] = {}

    def on_failure(unit: "llm_client.Unit", reason: object) -> None:
        idiom = unit.data[0]
        if isinstance(reason, Exception):
            failures[unit.unit_id] = _occurrence_failure(idiom, f"LLM indisponible: {reason}")
        else:
            failures[unit.unit_id] = _occurrence_failure(
                idiom, f"réponse LLM invalide: occurrence_id manquant ou dupliqué "
                       f"({reason}): {unit.unit_id}", invalid=True,
            )

    results = llm_client.run_units(
        units, task_id=task.task_id, model=model, protocol=_occurrence_protocol(schema_variant),
        render_unit=render_unit, render_batch=render_batch,
        parse_unit=parse_unit, parse_batch=parse_batch,
        response_model_unit=None, response_model_batch=None,
        batch_size=batch_size, mode_batch=mode_batch, timeout=120,
        on_failure=on_failure,
    )
    results.update(failures)
    return results


def judge_occurrence(idiom: str, occ: dict, segments_by_idx: dict,
                     *, model: str | None = None) -> dict:
    """Chemin unitaire explicite de S3-judge-occurrence. ``task.custom_prompt``
    (VOCAB_LLM_S3_JUDGE_OCCURRENCE=...;prompt=<nom>, pipeline/llm_tasks.py)
    remplace system/template quand posé — voir pipeline/prompt_variants.py."""
    task = task_config("S3-judge-occurrence")
    resolved_model = model or task.model
    unit = _occurrence_unit(idiom, occ, segments_by_idx, custom_prompt=task.custom_prompt)
    results = _judge_occurrence_units([unit], task=task, model=resolved_model,
                                      batch_size=1, mode_batch=False)
    return results[unit.unit_id]


def judge_occurrences_batch(items: list[tuple[str, dict]], segments_by_idx: dict,
                            *, model: str | None = None) -> dict[str, dict]:
    """Chemin lot S3 explicite : UN prompt de lot couvrant exactement
    ``items``, une décision contrôlée par occurrence_id. Même prise en compte
    de ``task.custom_prompt`` que judge_occurrence. Pour juger un nombre
    quelconque d'occurrences en attente avec découpage/parallélisme internes,
    voir `_judge_occurrence_units` (utilisée directement par `run()`)."""
    task = task_config("S3-judge-occurrence")
    if not items:
        return {}
    if len(items) > effective_batch_size(task):
        raise ValueError(f"lot S3 de {len(items)} occurrences > batch_size={effective_batch_size(task)}")
    resolved_model = model or task.model
    units = [_occurrence_unit(idiom, occ, segments_by_idx, custom_prompt=task.custom_prompt)
            for idiom, occ in items]
    results = _judge_occurrence_units(units, task=task, model=resolved_model,
                                      batch_size=len(units), mode_batch=True)
    return {occ["occurrence_id"]: results[occ["occurrence_id"]] for _, occ in items}


LEXICALIZED_LABELS = {"idiome", "phrasal_verb", "semi_fige"}
MIN_CONFIDENCE = 0.5


def _spans_overlap(a_spans: list, b_spans: list) -> bool:
    return any(a0 < b1 and b0 < a1 for a0, a1 in a_spans for b0, b1 in b_spans)


def select_mwe_spans(decisions: list[dict]) -> dict[int, list[dict]]:
    """Réserve les spans occurrence par occurrence pour les types
    confirmés lexicalisés (proposition_1 §3.3), sur leurs MEMBRES exacts
    — jamais leur enveloppe (défaut A, plan Partie 2 point D/17) : dans
    "turn the radio down", seuls "turn"/"down" sont membres, donc "radio"
    ne bloque plus aucun autre candidat. Priorité au candidat le plus
    riche en membres en cas de chevauchement (ex : "put up with" > "put
    up"). Retourne {segment_idx: [spans]}.

    Un candidat sans `member_char_spans` (absent — vieil artefact avant
    le Lot 1 — ou vide — alignement ambigu, pipeline/mwe_alignment.py) ne
    réserve jamais rien : abstention plutôt que suppression incertaine.

    Lot 3 (point C) : si `occ["occurrence_decision"]` existe (escalade
    occurrence par occurrence sur signal du garde-fou directionnel VPC —
    voir mwe_judge.py::run), c'est ELLE qui décide pour cette occurrence
    précise, pas la décision de type — le type peut rester globalement
    "phrasal_verb" alors qu'une occurrence isolée est jugée "littéral"
    ("walk up the stairs" au milieu d'un livre où "walk up" est par
    ailleurs authentiquement un phrasal_verb)."""

    candidates = []
    for entry in decisions:
        for occ in entry["occurrences"]:
            occ_decision = occ.get("occurrence_decision")
            if not occ_decision:
                continue
            label = occ_decision["label"]
            confidence = occ_decision["confidence"]
            if label not in LEXICALIZED_LABELS or confidence < MIN_CONFIDENCE:
                continue
            member_char_spans = occ.get("member_char_spans")
            if occ.get("ambiguous_alignment") or not member_char_spans:
                continue
            canonical_form = occ_decision.get("canonical_form", entry["idiom"])
            pos = occ_decision.get("pos", "OTHER")
            sense_id = occ_decision.get("sense_id") or custom_sense_id(
                canonical_form, pos, occ_decision.get("contextual_paraphrase", entry["idiom"])
            )
            candidates.append({
                "occurrence_id": occ["occurrence_id"],
                "idiom": entry["idiom"],
                "label": label,
                "confidence": confidence,
                "segment_idx": occ["segment_idx"],
                "start_char": occ["start_char"],
                "end_char": occ["end_char"],
                "surface": occ["surface"],
                "n_tokens": occ["n_tokens_span"],
                "member_char_spans": member_char_spans,
                # Lot 4 (point G) : décision de TYPE uniquement (jamais
                # d'occurrence_decision pour le sens WordNet) — propagé
                # jusqu'à mwe_confirmed_spans.jsonl pour que select.py
                # puisse s'en servir comme repli de glose quand idioms.yml
                # n'a rien pour cet idiome (cas de toutes les MWE apportées
                # par la fusion VPC, ex. "wake up").
                "canonical_form": canonical_form,
                "pos": pos,
                "contextual_paraphrase": occ_decision.get("contextual_paraphrase", ""),
                "definition_en": occ_decision.get("definition_en"),
                "definition_source": occ_decision.get("definition_source"),
                "definition_candidate_id": occ_decision.get("definition_candidate_id"),
                "definition_needs_review": occ_decision.get("definition_needs_review", True),
                "sense_id": sense_id,
                "sense_id_source": occ_decision.get("sense_id_source", CUSTOM_SENSE_VERSION),
                "wordnet_sense_id": occ_decision.get("wordnet_sense_id"),
            })

    by_segment: dict[int, list[dict]] = defaultdict(list)
    for c in candidates:
        by_segment[c["segment_idx"]].append(c)

    resolved: dict[int, list[dict]] = {}
    for seg_idx, spans in by_segment.items():
        # Le plus riche en MEMBRES d'abord (pas l'enveloppe), départagé
        # par la longueur totale de ces membres puis par occurrence_id
        # pour un ordre stable ; on ne garde un span que si ses membres
        # ne chevauchent les membres d'aucun span déjà retenu.
        spans_sorted = sorted(
            spans,
            key=lambda s: (
                -len(s["member_char_spans"]),
                -sum(b - a for a, b in s["member_char_spans"]),
                s["occurrence_id"],
            ),
        )
        kept: list[dict] = []
        for s in spans_sorted:
            overlaps = any(
                _spans_overlap(s["member_char_spans"], k["member_char_spans"])
                for k in kept
            )
            if not overlaps:
                kept.append(s)
        resolved[seg_idx] = kept

    return resolved


def run() -> int:
    config.ensure_out_dir()

    from pipeline.corpus import load_segments
    segments_by_idx = {s.idx: s for s in load_segments()}

    types = []
    with config.MWE_CANDIDATES_PATH.open(encoding="utf-8") as f:
        for line in f:
            types.append(json.loads(line))

    # Lot 3 (point C) : les deux magasins permanents sont consultés AVANT
    # tout appel LLM — si une clé existe déjà (n'importe quel statut), on ne
    # rejuge pas et on n'écrit rien à cette clé : c'est ce qui rend un run
    # complet gratuit (hors types/occurrences jamais vus) d'un livre à
    # l'autre, ET ce qui garantit qu'une entrée `status: validated`
    # (mwe_stores.PROTECTED_STATUSES, écrite à la main par une relecture
    # future) n'est jamais recalculée ni écrasée — cas particulier de la
    # règle générale "on ne touche jamais une clé déjà présente".
    occurrence_store = mwe_stores.load_occurrence_store()
    legacy_store_entries = sum(not key.startswith(f"s3:{S3_PROMPT_VERSION}:")
                               for key in occurrence_store)

    print(f"{len(types)} groupes à juger occurrence par occurrence...")
    if legacy_store_entries:
        print(f"  {legacy_store_entries} ancienne(s) décision(s) globale(s)/incompatible(s) "
              "conservée(s) pour audit mais invalidée(s) par le protocole S3 courant.")

    occurrence_task = task_config("S3-judge-occurrence")
    occurrence_batch_size = effective_batch_size(occurrence_task)
    occurrence_batch_mode = use_batch_prompt(occurrence_task, occurrence_batch_size)
    print(f"S3-judge-occurrence : {occurrence_task.model}, "
          f"prompt {'lot' if occurrence_batch_mode else 'unitaire'}.")
    lexicalized = 0
    n_occ_llm_failures = 0
    n_occ_escalated = 0
    records = []
    pending: list[tuple[str, dict, str]] = []
    pending_units: list[llm_client.Unit] = []
    for entry in types:
        idiom = entry["idiom"]
        occurrences = []
        for occ in entry["occurrences"]:
            n_occ_escalated += 1
            signature = occurrence_context_signature(idiom, occ, segments_by_idx)
            store_key = occurrence_store_key(
                idiom, occ["occurrence_id"], model=occurrence_task.model,
                backend=occurrence_task.provider, context_signature=signature,
            )
            occ_cached = occurrence_store.get(store_key)
            prepared_occ = {**occ}
            if occ_cached is not None:
                prepared_occ["occurrence_decision"] = {
                    k: v for k, v in occ_cached.items()
                    if k not in {"key", "status", "decided_at"}
                }
            else:
                pending.append((idiom, prepared_occ, store_key))
                pending_units.append(_occurrence_unit(
                    idiom, prepared_occ, segments_by_idx, custom_prompt=occurrence_task.custom_prompt,
                ))
            occurrences.append(prepared_occ)

        records.append({**entry, "occurrences": occurrences})

    # Lot de décorrélation lot/stockage : un seul appel couvrant TOUTES les
    # occurrences en attente, quel que soit leur nombre — _judge_occurrence_units
    # (via llm_client.run_units) fait le découpage en tranches de
    # occurrence_batch_size ET le cache unitaire ; plus de boucle manuelle
    # `for start in range(0, len(pending), occurrence_batch_size)` ici.
    decisions_by_id = _judge_occurrence_units(
        pending_units, task=occurrence_task, model=occurrence_task.model,
        batch_size=occurrence_batch_size, mode_batch=occurrence_batch_mode,
    )
    for idiom, occ, store_key in pending:
        occ_decision = decisions_by_id[occ["occurrence_id"]]
        occ["occurrence_decision"] = occ_decision
        if is_llm_failure(occ_decision):
            n_occ_llm_failures += 1
        else:
            occurrence_store[store_key] = mwe_stores.build_entry(store_key, occ_decision)

    for i, record in enumerate(records, start=1):
        occurrences = record["occurrences"]
        labels = [o["occurrence_decision"]["label"] for o in occurrences]
        heterogeneous = len(set(labels)) > 1
        decision = {
            "label": labels[0] if labels and not heterogeneous else "incertain",
            "confidence": min((o["occurrence_decision"]["confidence"] for o in occurrences), default=0.0),
            "reason": "synthèse homogène" if labels and not heterogeneous else "groupe hétérogène; voir les décisions d'occurrence",
            "heterogeneous": heterogeneous,
            "decision_scope": "summary_only",
        }
        if any(label in LEXICALIZED_LABELS for label in labels):
            lexicalized += 1
        record.update(decision)
        if i % 25 == 0 or i == len(types):
            print(f"  {i}/{len(types)}")
    assign_sense_ids(records)
    assign_cluster_definitions(records, segments_by_idx)
    atomic.atomic_write_jsonl(config.MWE_DECISIONS_PATH, records)

    print(f"{lexicalized}/{len(types)} groupes contiennent au moins une occurrence lexicalisée.")
    if n_occ_escalated:
        print(f"{n_occ_escalated} occurrence(s) jugée(s), toutes sources confondues "
              f"({len(pending)} transmise(s) au juge — hors magasin métier "
              f"{config.MWE_OCCURRENCE_STORE_PATH.name}, le reste déjà décidé).")
        if n_occ_llm_failures:
            print(f"  {n_occ_llm_failures} panne(s) LLM d'occurrence — pas mises en cache non plus.")
    print(f"-> {config.MWE_DECISIONS_PATH}")

    mwe_stores.write_occurrence_store(occurrence_store)
    print(f"-> {config.MWE_OCCURRENCE_STORE_PATH} ({len(occurrence_store)} entrées)")

    write_confirmed_spans()
    return 0


def load_decisions() -> list[dict]:
    with config.MWE_DECISIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_confirmed_spans() -> None:
    decisions = load_decisions()
    resolved = select_mwe_spans(decisions)

    n_spans = sum(len(v) for v in resolved.values())
    atomic.atomic_write_jsonl(
        config.MWE_SPANS_PATH,
        ({"segment_idx": seg_idx, "spans": spans} for seg_idx, spans in resolved.items()),
    )

    print(f"{n_spans} spans MWE confirmés et réservés -> {config.MWE_SPANS_PATH}")


if __name__ == "__main__":
    raise SystemExit(run())
