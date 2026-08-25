"""Vérification de non-régression S5 (item 1 du plan) : les 14 cas de
`sense_in_context.py` doivent toujours donner 14/14 une fois passés
par pipeline/senses.py (contexte élargi par index de segment au lieu
de recherche par flux de tokens, phrase FR injectée directement au
lieu de venir de TESTS)."""

from __future__ import annotations

from pipeline.corpus import load_segments
from pipeline import senses

TESTS = [
    {"word": "diss", "pos": "v",
     "english": "You guys better not be dissing my home—do you even get how special a place like this is?",
     "french": "Vous n’avez pas le droit de vous moquer de mon chez-moi. Vous comprenez au moins à quel point un endroit comme celui-ci est exceptionnel ?",
     "expected": "diss.v.01"},
    {"word": "duplex", "pos": "n", "english": "No New Yorkers have duplex apartments.",
     "french": "Aucun New-Yorkais n’a d’appartement en duplex.", "expected": "duplex_apartment.n.01"},
    {"word": "standard", "pos": "a", "english": "No, that's standard for a ground-floor apartment.",
     "french": "Non, c’est tout à fait normal pour un appartement au rez-de-chaussée.", "expected": "standard.a.01"},
    {"word": "super", "pos": "n", "english": "Uh, must be the super, he's the only one who has access.",
     "french": "Ah, ça doit être le concierge, il est le seul à avoir la clé.", "expected": "superintendent.n.02"},
    {"word": "access", "pos": "n", "english": "He's the only one who has access.",
     "french": "Il est le seul à avoir la clé.", "expected": "access.n.02"},
    {"word": "view", "pos": "n", "english": "I wish you had more of a view.",
     "french": "J’aimerais que vous ayez une meilleure vue.", "expected": "view.n.02"},
    {"word": "alley", "pos": "n", "english": "It's an alley full of cigarette butts.",
     "french": "C’est une ruelle pleine de mégots de cigarettes.", "expected": "alley.n.01"},
    {"word": "butt", "pos": "n", "english": "It's an alley full of cigarette butts.",
     "french": "C’est une ruelle pleine de mégots de cigarettes.", "expected": "butt.n.09"},
    {"word": "courtyard", "pos": "n", "english": "It's an interior courtyard.",
     "french": "C’est une cour intérieure.", "expected": "court.n.10"},
    {"word": "score", "pos": "n", "english": "I wanna check the score of the game.",
     "french": "Je veux vérifier le score du match.", "expected": "score.n.03"},
    {"word": "plow", "pos": "v", "english": "The roads are all plowed—",
     "french": "Les routes ont toutes été déneigées—", "expected": "plow.v.01"},
    {"word": "key", "pos": "n", "english": "That is a terrible key for me.",
     "french": "C’est une tonalité épouvantable pour moi.", "expected": "key.n.04"},
    {"word": "gig", "pos": "n", "english": "You have any gigs lined up?",
     "french": "Tu as des concerts de prévus ?", "expected": "gig.n.06"},
    {"word": "shake", "pos": "n", "english": "Are her shakes in the fridge?",
     "french": "Ses milk-shakes sont-ils dans le frigo ?", "expected": "milkshake.n.01"},
]


import re as _re


def _normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("—", " ").casefold()
    return _re.sub(r"[^a-z0-9' ]", " ", text)


def _tokens(text: str) -> set[str]:
    return {t for t in _normalize(text).split() if len(t) > 2}


def find_segment_idx(segments, word: str, english_snippet: str) -> int | None:
    # 1) égalité normalisée, ou sous-chaîne — mais seulement quand le
    #    plus court des deux textes fait au moins 4 mots, pour éviter
    #    qu'une didascalie d'un seul mot ("[One.]") ne matche
    #    trivialement une sous-chaîne de la phrase cible.
    norm_target = _normalize(english_snippet).strip()
    target_word_count = len(norm_target.split())
    for s in segments:
        norm_seg = _normalize(s.en).strip()
        if norm_seg == norm_target:
            return s.idx
        shorter_len = min(len(norm_seg.split()), target_word_count)
        if shorter_len >= 4 and (norm_target in norm_seg or norm_seg in norm_target):
            return s.idx

    # 2) repli : la mise en page en deux colonnes du fichier source
    #    coupe parfois une réplique sur deux lignes physiques séparées
    #    par une didascalie (ex: "standard", voir The Humans ligne
    #    1007/1013). Le vrai pipeline s'en sort car le contexte élargi
    #    (±2 segments) englobe naturellement la suite — ce repli ne
    #    sert qu'à retrouver le bon segment de DÉPART pour ce test :
    #    le segment qui contient le mot cible ET partage le plus de
    #    mots avec la phrase de test.
    target_tokens = _tokens(english_snippet)
    best, best_overlap = None, 0
    for s in segments:
        if s.kind == "hors_oeuvre":
            continue
        if not _re.search(r"\b" + _re.escape(word) + r"\b", s.en, _re.IGNORECASE):
            continue
        overlap = len(target_tokens & _tokens(s.en))
        if overlap > best_overlap:
            best, best_overlap = s.idx, overlap

    return best if best_overlap >= 2 else None


def main() -> int:
    segments = load_segments()
    by_idx = {s.idx: s for s in segments}

    ok = 0
    for test in TESTS:
        seg_idx = find_segment_idx(segments, test["word"], test["english"])
        status = "SEGMENT INTROUVABLE"
        got = None

        if seg_idx is not None:
            original_fr = by_idx[seg_idx].fr
            by_idx[seg_idx].fr = test["french"]
            try:
                record = senses.analyze_occurrence(
                    test["word"], test["pos"], segments, seg_idx, allow_arbitration=False
                )
            finally:
                by_idx[seg_idx].fr = original_fr

            if record is not None:
                got = record["best_sense"]
                status = "OK" if got == test["expected"] else "ÉCHEC"
                if got == test["expected"]:
                    ok += 1

        print(f"{status:20s} {test['word']:10s} attendu={test['expected']:25s} obtenu={got}")

    print(f"\n{ok}/{len(TESTS)} OK")
    return 0 if ok == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
