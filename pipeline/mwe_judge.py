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

from pipeline import atomic, config, llm, mwe_stores

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

    try:
        result = llm.call_json(prompt, system=SYSTEM_PROMPT, timeout=120)
    except llm.LLMError as exc:
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


def occurrence_context_signature(idiom: str, occ: dict, segments_by_idx: dict) -> str:
    """Signature stable de toute donnée susceptible de changer le verdict."""
    segment = segments_by_idx.get(occ.get("segment_idx"))
    sentence = getattr(segment, "text", "") if segment is not None else ""
    payload = {
        "canonical_form": idiom.casefold().strip(), "surface": occ.get("surface") or "",
        "sentence": sentence, "source": occ.get("source"),
        "member_char_spans": occ.get("member_char_spans"),
        "vpc_decision_reason": occ.get("vpc_decision_reason"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def occurrence_store_key(idiom: str, occurrence_id: str, *, model: str | None = None,
                         backend: str | None = None, context_signature: str = "") -> str:
    """Clé protocolaire : les anciennes décisions sont auditables mais illisibles."""
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


def choose_cluster_definition(canonical_form: str, pos: str, occurrences: list[dict],
                              segments_by_idx: dict) -> dict:
    """Choisit une glose après comparaison de tous les candidats/contexts."""
    candidates = definition_candidates(canonical_form, pos)
    occurrence_lines, occurrence_ids = [], []
    for occ in occurrences:
        occurrence_id = occ["occurrence_id"]
        occurrence_ids.append(occurrence_id)
        segment = segments_by_idx.get(occ.get("segment_idx"))
        text = getattr(segment, "text", "") if segment is not None else ""
        paraphrase = occ["occurrence_decision"].get("contextual_paraphrase", "")
        occurrence_lines.append(f'- {occurrence_id}: "{text}" | paraphrase: {paraphrase}')
    candidate_lines = [f'- {c["candidate_id"]}: {c["definition"]}' for c in candidates]
    prompt = DEFINITION_PROMPT_TEMPLATE.format(
        canonical_form=canonical_form, pos=pos, occurrences="\n".join(occurrence_lines),
        candidates="\n".join(candidate_lines) or "- (aucune entrée disponible)",
    )
    try:
        result = llm.call_json(prompt, system=DEFINITION_SYSTEM_PROMPT, timeout=120)
    except llm.LLMError as exc:
        result = {"candidate_id": None, "custom_definition": "", "reason": str(exc)}

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
                     for o in occurrences
                     if o["occurrence_decision"].get("contextual_paraphrase")), canonical_form)
    return {"definition_en": fallback, "definition_source": "custom",
            "definition_candidate_id": None, "definition_needs_review": True}


def assign_cluster_definitions(records: list[dict], segments_by_idx: dict) -> None:
    clusters: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for entry in records:
        for occ in entry["occurrences"]:
            decision = occ["occurrence_decision"]
            if decision.get("sense_id") and decision["label"] in LEXICALIZED_LABELS:
                key = (decision["canonical_form"].casefold().strip(), decision["pos"],
                       decision["sense_id"])
                clusters[key].append(occ)
    for (canonical_form, pos, _sense_id), occurrences in clusters.items():
        selection = choose_cluster_definition(canonical_form, pos, occurrences, segments_by_idx)
        for occ in occurrences:
            occ["occurrence_decision"].update(selection)


def _calibrate_occurrence(result: dict, label: str) -> tuple[float, list[str]]:
    """Score reproductible, distinct de la confiance autodéclarée du modèle.

    Ce n'est pas une calibration statistique finale (Q0-2 la mesurera), mais
    un score calibrable : complétude des preuves et cohérence du verdict sont
    des features explicites. La confiance brute n'est jamais utilisée seule.
    """
    raw = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    evidence = [str(x).strip() for x in result.get("evidence", []) if str(x).strip()]
    complete = all(str(result.get(k, "")).strip() for k in (
        "canonical_form", "pos", "contextual_paraphrase", "reason"
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


def judge_occurrence(idiom: str, occ: dict, segments_by_idx: dict,
                     *, model: str | None = None) -> dict:
    seg = segments_by_idx.get(occ["segment_idx"])
    sentence = seg.en if seg else occ["surface"]
    prompt = OCC_PROMPT_TEMPLATE.format(
        idiom=idiom,
        sentence=sentence,
        surface=occ["surface"],
        source=occ.get("source", "inconnue"),
        vpc_decision_reason=occ.get("vpc_decision_reason") or "voir le détecteur VPC",
    )
    wn_candidates = wordnet_synset_candidates(idiom)
    if wn_candidates:
        prompt += "\nSens WordNet autorisés (choisir seulement une correspondance exacte) :\n"
        prompt += "\n".join(f"- {s.name()}: {s.definition()}" for s in wn_candidates)
    try:
        result = llm.call_json(
            prompt, system=OCC_SYSTEM_PROMPT, model=model, timeout=120,
            cache_metadata={"protocol": S3_PROMPT_VERSION,
                            "schema": S3_DECISION_SCHEMA_VERSION},
        )
    except llm.LLMError as exc:
        return {
            "label": "incertain", "verdict": "incertain",
            "canonical_form": idiom, "pos": "OTHER",
            "contextual_paraphrase": "", "model_confidence": 0.0,
            "confidence": 0.0, "confidence_features": ["llm_failure"],
            "evidence": [], "reason": f"LLM indisponible: {exc}",
        }

    label = result.get("label")
    if label not in VALID_LABELS:
        return {
            "label": "incertain", "verdict": "incertain",
            "canonical_form": idiom, "pos": "OTHER",
            "contextual_paraphrase": "", "model_confidence": 0.0,
            "confidence": 0.0, "confidence_features": ["invalid_response"],
            "evidence": [], "reason": f"réponse LLM invalide: {result!r}",
        }

    canonical = str(result.get("canonical_form") or idiom).strip()
    pos = str(result.get("pos") or "OTHER").upper()
    if pos not in VALID_POS:
        pos = "OTHER"
    confidence, confidence_features = _calibrate_occurrence(result, label)
    lexicalized = label in {"idiome", "phrasal_verb", "semi_fige"}
    valid_wn_ids = {s.name() for s in wn_candidates}
    selected_wn_id = result.get("wordnet_sense_id")
    return {
        "label": label,
        "verdict": "lexicalisé" if lexicalized else label,
        "canonical_form": canonical,
        "pos": pos,
        "contextual_paraphrase": str(result.get("contextual_paraphrase") or "").strip(),
        "model_confidence": max(0.0, min(1.0, float(result.get("confidence", 0.0)))),
        "confidence": confidence,
        "confidence_features": confidence_features,
        "evidence": [str(x).strip() for x in result.get("evidence", []) if str(x).strip()],
        "wordnet_sense_id": selected_wn_id if selected_wn_id in valid_wn_ids else None,
        "reason": result.get("reason", ""),
    }


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

    print(f"{len(types)} groupes à juger occurrence par occurrence via {config.llm_model()}...")
    if legacy_store_entries:
        print(f"  {legacy_store_entries} ancienne(s) décision(s) globale(s)/incompatible(s) "
              "conservée(s) pour audit mais invalidée(s) par le protocole S3 courant.")

    lexicalized = 0
    n_occ_llm_calls = 0
    n_occ_llm_failures = 0
    n_occ_escalated = 0
    records = []
    for i, entry in enumerate(types, start=1):
        idiom = entry["idiom"]
        occurrences = []
        for occ in entry["occurrences"]:
            n_occ_escalated += 1
            signature = occurrence_context_signature(idiom, occ, segments_by_idx)
            store_key = occurrence_store_key(
                idiom, occ["occurrence_id"], model=config.llm_model(),
                backend=config.LLM_BACKEND,
                context_signature=signature,
            )
            occ_cached = occurrence_store.get(store_key)
            if occ_cached is not None:
                occ_decision = {k: v for k, v in occ_cached.items()
                                if k not in {"key", "status", "decided_at"}}
            else:
                occ_decision = judge_occurrence(idiom, occ, segments_by_idx)
                n_occ_llm_calls += 1
                if is_llm_failure(occ_decision):
                    n_occ_llm_failures += 1
                else:
                    occurrence_store[store_key] = mwe_stores.build_entry(store_key, occ_decision)
            occ = {**occ, "occurrence_decision": occ_decision}
            occurrences.append(occ)

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
        records.append({**entry, **decision, "occurrences": occurrences})
        if i % 25 == 0 or i == len(types):
            print(f"  {i}/{len(types)}")
    assign_sense_ids(records)
    assign_cluster_definitions(records, segments_by_idx)
    atomic.atomic_write_jsonl(config.MWE_DECISIONS_PATH, records)

    print(f"{lexicalized}/{len(types)} groupes contiennent au moins une occurrence lexicalisée.")
    if n_occ_escalated:
        print(f"{n_occ_escalated} occurrence(s) jugée(s), toutes sources confondues "
              f"({n_occ_llm_calls} appel(s) LLM d'occurrence, le reste déjà dans "
              f"{config.MWE_OCCURRENCE_STORE_PATH.name}).")
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
