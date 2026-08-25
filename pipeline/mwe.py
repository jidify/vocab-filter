"""S2 — Détecter les expressions avant les mots simples.

`idiomatch` reste le générateur de candidats à haut rappel (§10 du
résumé), réglé à `n=2` (repris de la conclusion pratique §10.9). Mais
son inventaire n'a AUCUN marqueur "idiomatique" vs "littéral/composable"
exploitable automatiquement (vérifié : "know someone", "go to",
"talk about" ont des définitions dans idioms.yml au même titre que
"figure out" ou "wing it" — la différence n'est pas dans les données,
elle doit être jugée). Le pré-filtre ici ne fait donc QUE ce qui est
structurellement certain ; la décision idiome/littéral/compositionnel
revient entièrement à mwe_judge.py (S3), comme le veut §3.3 de
proposition_1.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from idiomatch import Idiomatcher

from pipeline import config
from pipeline.corpus import load_segments

_MATCHER = None
_IDIOMS_YML: dict[str, dict] | None = None

# Idiomes absents de la base idiomatch/Wiktionary mais identifiés dans
# vocab-filter-resume.md §10.4-10.6 comme manquants et ajoutés
# dynamiquement via add_idioms() (schéma de test_crack_open.py:10-30).
# "crack open" est confirmé présent dans le texte : "cracks the
# bathroom door open" et "cracks open another beer".
CUSTOM_IDIOMS = [
    {
        "etymology": None,
        "lemma": "crack open",
        "senses": [{"content": "To cause something to open.", "examples": []}],
        "source": "custom",
    },
]


def get_idiom_definition(idiom: str) -> str | None:
    """Glose anglaise de l'idiome telle qu'écrite dans idioms.yml
    (idiomatch/resources/idioms.yml) — sert directement de "sens" pour
    les unités multi-mots en S5/S6 : ces expressions n'ont
    généralement pas d'entrée WordNet propre, donc GlossBERT/omw-fr
    n'ont rien à désambiguïser pour elles."""

    global _IDIOMS_YML
    if _IDIOMS_YML is None:
        import yaml
        from idiomatch.idiomatcher import RESOURCES_DIR

        data = yaml.safe_load((RESOURCES_DIR / "idioms.yml").read_text(encoding="utf-8"))
        _IDIOMS_YML = {entry["lemma"]: entry for entry in data}
        for entry in CUSTOM_IDIOMS:
            _IDIOMS_YML[entry["lemma"]] = entry

    entry = _IDIOMS_YML.get(idiom)
    if not entry or not entry.get("senses"):
        return None
    return entry["senses"][0].get("content")


def get_matcher():
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = Idiomatcher.from_pretrained(n=2)
        _MATCHER.add_idioms(CUSTOM_IDIOMS)
    return _MATCHER


def find_candidates(segments):
    """Fait tourner idiomatch SEGMENT PAR SEGMENT (pas le livre entier
    en un seul nlp() comme test_idiomatch_book.py:57 — un match ne doit
    jamais pouvoir chevaucher deux segments)."""

    matcher = get_matcher()
    play_segments = [s for s in segments if s.kind != "hors_oeuvre"]

    for seg in play_segments:
        doc = matcher.nlp(seg.en)
        matches = matcher(doc)

        for m in matches:
            match_id, start, end = m["meta"]
            span = doc[start:end]
            yield {
                "segment_idx": seg.idx,
                "kind": seg.kind,
                "idiom": m["idiom"],
                "surface": span.text,
                "start_token": start,
                "end_token": end,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "n_tokens_span": end - start,
                "n_tokens_lemma": len(m["idiom"].split()),
            }


def structural_prefilter(candidates: list[dict]) -> list[dict]:
    """Ne rejette QUE ce qui est certain sans jugement sémantique :
    - déjà garanti par construction (un match idiomatch ne franchit
      jamais un segment, on le garde comme filet de sécurité) ;
    - candidats à un seul token (bruit de tokenisation, pas une MWE) ;
    - même (idiome, segment, span) en double.
    """

    seen = set()
    kept = []
    for c in candidates:
        if c["n_tokens_span"] < 2:
            continue
        key = (c["idiom"], c["segment_idx"], c["start_char"], c["end_char"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept


def group_by_type(candidates: list[dict]) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_type[c["idiom"]].append(c)
    return dict(by_type)


def run() -> int:
    config.ensure_out_dir()
    segments = load_segments()

    print("Chargement d'idiomatch (n=2)...")
    raw = list(find_candidates(segments))
    print(f"{len(raw)} occurrences brutes.")

    filtered = structural_prefilter(raw)
    print(f"{len(filtered)} après pré-filtre structurel.")

    by_type = group_by_type(filtered)
    print(f"{len(by_type)} types distincts.")

    with config.MWE_CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        for idiom, occs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            f.write(json.dumps(
                {"idiom": idiom, "count": len(occs), "occurrences": occs},
                ensure_ascii=False,
            ) + "\n")

    print(f"-> {config.MWE_CANDIDATES_PATH}")

    top = Counter({idiom: len(occs) for idiom, occs in by_type.items()}).most_common(15)
    print("\nTop 15 par fréquence (à juger en S3, pas à faire confiance ici) :")
    for idiom, count in top:
        print(f"  {count:4d}  {idiom}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
