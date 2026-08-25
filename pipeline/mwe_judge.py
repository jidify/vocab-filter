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

from pipeline import config, llm

VALID_LABELS = {"idiome", "phrasal_verb", "semi_fige", "littéral", "incertain"}

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


LEXICALIZED_LABELS = {"idiome", "phrasal_verb", "semi_fige"}
MIN_CONFIDENCE = 0.5


def select_mwe_spans(decisions: list[dict]) -> dict[int, list[dict]]:
    """Réserve les spans occurrence par occurrence pour les types
    confirmés lexicalisés (proposition_1 §3.3) : priorité au match le
    plus long en cas de chevauchement dans un même segment (ex :
    "get away with" > "get away"). Retourne {segment_idx: [spans]}."""

    candidates = []
    for entry in decisions:
        if entry["label"] not in LEXICALIZED_LABELS:
            continue
        if entry["confidence"] < MIN_CONFIDENCE:
            continue
        for occ in entry["occurrences"]:
            candidates.append({
                "idiom": entry["idiom"],
                "label": entry["label"],
                "confidence": entry["confidence"],
                "segment_idx": occ["segment_idx"],
                "start_char": occ["start_char"],
                "end_char": occ["end_char"],
                "surface": occ["surface"],
                "n_tokens": occ["n_tokens_span"],
            })

    by_segment: dict[int, list[dict]] = defaultdict(list)
    for c in candidates:
        by_segment[c["segment_idx"]].append(c)

    resolved: dict[int, list[dict]] = {}
    for seg_idx, spans in by_segment.items():
        # Le plus long d'abord (en tokens, puis en caractères) ; on ne
        # garde un span que s'il ne chevauche aucun span déjà retenu.
        spans_sorted = sorted(
            spans, key=lambda s: (-s["n_tokens"], -(s["end_char"] - s["start_char"]))
        )
        kept: list[dict] = []
        for s in spans_sorted:
            overlaps = any(
                s["start_char"] < k["end_char"] and k["start_char"] < s["end_char"]
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

    print(f"{len(types)} types à juger via {config.OLLAMA_MODEL}...")

    lexicalized = 0
    with config.MWE_DECISIONS_PATH.open("w", encoding="utf-8") as out:
        for i, entry in enumerate(types, start=1):
            decision = judge_type(entry["idiom"], entry["occurrences"], segments_by_idx)
            if decision["label"] in {"idiome", "phrasal_verb", "semi_fige"}:
                lexicalized += 1
            record = {**entry, **decision}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(types):
                print(f"  {i}/{len(types)}")

    print(f"{lexicalized}/{len(types)} types jugés lexicalisés (idiome/phrasal_verb/semi_fige).")
    print(f"-> {config.MWE_DECISIONS_PATH}")

    write_confirmed_spans()
    return 0


def load_decisions() -> list[dict]:
    with config.MWE_DECISIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_confirmed_spans() -> None:
    decisions = load_decisions()
    resolved = select_mwe_spans(decisions)

    n_spans = sum(len(v) for v in resolved.values())
    with config.MWE_SPANS_PATH.open("w", encoding="utf-8") as f:
        for seg_idx, spans in resolved.items():
            f.write(json.dumps({"segment_idx": seg_idx, "spans": spans}, ensure_ascii=False) + "\n")

    print(f"{n_spans} spans MWE confirmés et réservés -> {config.MWE_SPANS_PATH}")


if __name__ == "__main__":
    raise SystemExit(run())
