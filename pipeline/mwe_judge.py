"""S3 — Valider chaque type MWE : idiome / phrasal_verb / semi_figé /
littéral / incertain (proposition_1 §3.3).

Un type confirmé comme lexicalisé réserve ses spans OCCURRENCE PAR
OCCURRENCE — un `figure out` confirmé ne bloque pas un `figure` isolé
ailleurs dans le livre (proposition_1 §3.3, exemple `figure out`/`open
door`). C'est select_mwe_spans() plus bas qui applique cette règle, y
compris la priorité au match le plus long (`get away with` > `get
away`).

Le jugement se fait sur un ÉCHANTILLON d'occurrences par type (jusqu'à
3), pas occurrence par occurrence : la question "know someone" est-il
compositionnel ne dépend quasiment jamais du contexte précis, et
demander un jugement occurrence par occurrence coûterait 979 appels
LLM pour une réponse qui ne change pas.
"""

from __future__ import annotations

import json
from collections import defaultdict

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

Réponds en JSON strict avec ce schéma :
{{"label": "<une des 5 catégories>", "confidence": <0.0-1.0>, "reason": "<1 phrase en français>"}}
"""


def format_examples(occurrences: list[dict], segments_by_idx: dict) -> str:
    lines = []
    for occ in occurrences[:3]:
        seg = segments_by_idx.get(occ["segment_idx"])
        text = seg.en if seg else occ["surface"]
        lines.append(f'- "{text}" (span détecté : "{occ["surface"]}")')
    return "\n".join(lines)


def judge_type(idiom: str, occurrences: list[dict], segments_by_idx: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        idiom=idiom,
        examples=format_examples(occurrences, segments_by_idx),
    )
    try:
        result = llm.call_json(prompt, system=SYSTEM_PROMPT, timeout=120)
    except llm.LLMError as exc:
        return {"label": "incertain", "confidence": 0.0, "reason": f"LLM indisponible: {exc}"}

    label = result.get("label")
    if label not in VALID_LABELS:
        return {"label": "incertain", "confidence": 0.0,
                "reason": f"réponse LLM invalide: {result!r}"}

    return {
        "label": label,
        "confidence": float(result.get("confidence", 0.0)),
        "reason": result.get("reason", ""),
    }


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
    "dans CETTE phrase précise, un verbe de mouvement suivi d'une particule "
    "directionnelle (up/down/in/out...) qui peut être soit un phrasal verb à "
    "sens spécialisé, soit un simple complément de direction littéral (ex. "
    "\"walk up the stairs\" = monter les marches, vs \"walk up to someone\" = "
    "s'approcher de quelqu'un). Tu dois juger CETTE occurrence précise, pas "
    "l'expression en général."
)

OCC_PROMPT_TEMPLATE = """Expression candidate : "{idiom}"

Phrase : "{sentence}"
Span détecté : "{surface}"

Le détecteur syntaxique signale cette occurrence comme potentiellement
littérale malgré une correspondance lexicale attestée pour "{idiom}"
({vpc_decision_reason}).

Classe CETTE occurrence dans EXACTEMENT une de ces catégories :
- "idiome" : sens opaque, non déductible des mots
- "phrasal_verb" : verbe + particule à sens spécialisé, DANS CETTE PHRASE
- "semi_fige" : construction récurrente à sens partiellement prévisible mais notable
- "littéral" : ici, direction/complément littéral — chaque mot garde son sens ordinaire
- "incertain" : tu ne peux pas trancher avec ce contexte

Réponds en JSON strict avec ce schéma :
{{"label": "<une des 5 catégories>", "confidence": <0.0-1.0>, "reason": "<1 phrase en français>"}}
"""


def judge_occurrence(idiom: str, occ: dict, segments_by_idx: dict) -> dict:
    seg = segments_by_idx.get(occ["segment_idx"])
    sentence = seg.en if seg else occ["surface"]
    prompt = OCC_PROMPT_TEMPLATE.format(
        idiom=idiom,
        sentence=sentence,
        surface=occ["surface"],
        vpc_decision_reason=occ.get("vpc_decision_reason") or "voir le détecteur VPC",
    )
    try:
        result = llm.call_json(prompt, system=OCC_SYSTEM_PROMPT, timeout=120)
    except llm.LLMError as exc:
        return {"label": "incertain", "confidence": 0.0, "reason": f"LLM indisponible: {exc}"}

    label = result.get("label")
    if label not in VALID_LABELS:
        return {"label": "incertain", "confidence": 0.0,
                "reason": f"réponse LLM invalide: {result!r}"}

    return {
        "label": label,
        "confidence": float(result.get("confidence", 0.0)),
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
            label = occ_decision["label"] if occ_decision else entry["label"]
            confidence = occ_decision["confidence"] if occ_decision else entry["confidence"]
            if label not in LEXICALIZED_LABELS or confidence < MIN_CONFIDENCE:
                continue
            member_char_spans = occ.get("member_char_spans")
            if occ.get("ambiguous_alignment") or not member_char_spans:
                continue
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
    type_store = mwe_stores.load_type_store()
    occurrence_store = mwe_stores.load_occurrence_store()

    print(f"{len(types)} types à juger via {config.OLLAMA_MODEL} "
          f"({len(type_store)} déjà dans {config.MWE_TYPE_STORE_PATH.name})...")

    lexicalized = 0
    n_type_llm_calls = 0
    n_type_llm_failures = 0
    n_occ_llm_calls = 0
    n_occ_llm_failures = 0
    n_occ_escalated = 0
    records = []
    for i, entry in enumerate(types, start=1):
        idiom = entry["idiom"]
        cached = type_store.get(idiom)
        if cached is not None:
            decision = {"label": cached["label"], "confidence": cached["confidence"],
                        "reason": cached.get("reason", "")}
        else:
            decision = judge_type(idiom, entry["occurrences"], segments_by_idx)
            n_type_llm_calls += 1
            # Une panne (ollama injoignable, réponse illisible) n'est PAS une
            # décision : ne jamais la figer dans le magasin permanent (voir
            # is_llm_failure) — sinon un simple hoquet réseau condamnerait ce
            # type à "incertain" pour toujours, sur CE livre et les suivants.
            if is_llm_failure(decision):
                n_type_llm_failures += 1
            else:
                type_store[idiom] = mwe_stores.build_entry(idiom, decision)
        if decision["label"] in LEXICALIZED_LABELS:
            lexicalized += 1

        occurrences = []
        for occ in entry["occurrences"]:
            if occ.get("directional_context_dependent"):
                n_occ_escalated += 1
                occ_cached = occurrence_store.get(occ["occurrence_id"])
                if occ_cached is not None:
                    occ_decision = {"label": occ_cached["label"], "confidence": occ_cached["confidence"],
                                     "reason": occ_cached.get("reason", "")}
                else:
                    occ_decision = judge_occurrence(idiom, occ, segments_by_idx)
                    n_occ_llm_calls += 1
                    if is_llm_failure(occ_decision):
                        n_occ_llm_failures += 1
                    else:
                        occurrence_store[occ["occurrence_id"]] = mwe_stores.build_entry(
                            occ["occurrence_id"], occ_decision
                        )
                occ = {**occ, "occurrence_decision": occ_decision}
            occurrences.append(occ)

        records.append({**entry, **decision, "occurrences": occurrences})
        if i % 25 == 0 or i == len(types):
            print(f"  {i}/{len(types)}")
    atomic.atomic_write_jsonl(config.MWE_DECISIONS_PATH, records)

    print(f"{lexicalized}/{len(types)} types jugés lexicalisés (idiome/phrasal_verb/semi_fige) "
          f"({n_type_llm_calls} appel(s) LLM de type, {len(types) - n_type_llm_calls} "
          f"déjà dans {config.MWE_TYPE_STORE_PATH.name}).")
    if n_type_llm_failures:
        print(f"  {n_type_llm_failures} panne(s) LLM (voir is_llm_failure) — pas mises en cache, "
              f"resteront à retenter au prochain run.")
    if n_occ_escalated:
        print(f"{n_occ_escalated} occurrence(s) escaladée(s) par le garde-fou directionnel VPC "
              f"({n_occ_llm_calls} appel(s) LLM d'occurrence, le reste déjà dans "
              f"{config.MWE_OCCURRENCE_STORE_PATH.name}).")
        if n_occ_llm_failures:
            print(f"  {n_occ_llm_failures} panne(s) LLM d'occurrence — pas mises en cache non plus.")
    print(f"-> {config.MWE_DECISIONS_PATH}")

    mwe_stores.write_type_store(type_store)
    mwe_stores.write_occurrence_store(occurrence_store)
    print(f"-> {config.MWE_TYPE_STORE_PATH} ({len(type_store)} entrées)")
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
