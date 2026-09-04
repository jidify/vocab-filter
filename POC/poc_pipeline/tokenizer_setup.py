"""Point de configuration UNIQUE des tokenizers spaCy du POC — « stage 0 » au
sens d'un invariant de construction, pas d'une étape de
`build_vocabulary_to_learn_pipeline.py` : ce module ne produit aucun fichier,
il garantit que tout `nlp` (ou `Idiomatcher.nlp`) créé dans le POC passe par
`configure_tokenizer` avant le premier appel de parsing.

Trois consommateurs, tous mutés en mémoire, jamais réimportés séparément :
    - `poc_pipeline.analyze.get_nlp()`
    - `poc_pipeline.mwe.get_matcher()` (via `matcher.nlp`)
    - `POC/pipeline/stages/extract_word_contexts.py`

Voir `TODO/tokenizer_dash_after_punctuation.md` (le patch de tiret) et le
plan "Stage 0 du pipeline POC — configuration unique du tokenizer" pour le
contexte complet des deux besoins réunis ici.

Composition de `configure_tokenizer`, dans cet ordre, à chaque appel :
    1. `tokenizer_boundary_fix.patch_dash_after_punctuation` — toujours.
       Corrige le tiret collé après ponctuation fermante (`around?—we`).
       Idempotent par construction (voir sa docstring).
    2. `EMAIL_SPECIAL_CASES` + `custom_lexicon.load_tokenizer_surfaces()` —
       toujours. Socle historique de `analyze.get_nlp()`, déplacé ici pour
       que les TROIS tokenizers en bénéficient (avant ce module, seul B les
       recevait).
    3. Liste blanche de mots à tiret (`hyphen_words_without_npe.txt`) —
       seulement si `hyphen_whitelist=True`. Volontairement absente du
       tokenizer d'idiomatch (voir `configure_tokenizer` ci-dessous) : ses
       motifs sont compilés token par token, figer un composé en un seul
       token casserait un motif MWE construit dessus.

`add_special_case` est idempotent côté spaCy (ré-appliquer la même règle ne
duplique rien) — appeler `configure_tokenizer` plusieurs fois sur le même
`nlp` est donc sans effet cumulatif.
"""

from __future__ import annotations

from functools import lru_cache

from spacy.symbols import ORTH

from poc_pipeline import config, custom_lexicon
from poc_pipeline.tokenizer_boundary_fix import patch_dash_after_punctuation

# Voir poc_pipeline/analyze.py (définition d'origine, avant ce module) pour
# la justification empirique complète : sans ce cas spécial, spaCy coupe
# "e-mail" en 3 tokens et "mail" seul se désambiguïse à tort vers
# mail.v.01 "envoyer par la poste".
EMAIL_SPECIAL_CASES = ["e-mail", "e-mails", "e-mailing", "e-mailed"]

HYPHEN_WHITELIST_PATH = config.ROOT / "poc_datasets" / "hyphen_words_without_npe.txt"


@lru_cache(maxsize=1)
def load_hyphen_whitelist() -> frozenset[str]:
    """Formes à tiret admises telles quelles par le tokenizer, dérivées de
    `hyphen_words_without_npe.txt` (9536 lignes, noms propres déjà retirés en
    amont via `npe-list.txt` — voir ce fichier).

    Deux catégories de lignes sont écartées, PAS ajoutées comme cas
    spéciaux :
      - les 421 entrées portant un espace (`able-bodied seaman`) :
        `Tokenizer.add_special_case` exige une chaîne sans espace — ce sont
        des MWE, hors périmètre de ce module (idiomatch/VPC les couvrent,
        ou pas, séparément) ;
      - les 179 possessifs en `'s`/`'s` (`also-ran's`) : les garder entiers
        empêcherait spaCy de détacher le suffixe. Aucune perte — la forme de
        base est déjà présente dans le fichier, et spaCy retombe dessus
        après avoir scindé le possessif normalement.

    Résultat : ~8936 formes de base. Mémoïsé — lu une seule fois par
    processus (le fichier est immuable pendant un run)."""

    whitelist: set[str] = set()
    with HYPHEN_WHITELIST_PATH.open(encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if not word or " " in word:
                continue
            if word.endswith("'s") or word.endswith("’s"):
                continue
            whitelist.add(word)
    return frozenset(whitelist)


def _hyphen_case_variants(whitelist: frozenset[str]) -> set[str]:
    """Ajoute la forme capitalisée et tout-majuscules de chaque entrée —
    `add_special_case` est sensible à la casse, et ce corpus est une pièce
    de théâtre dont les didascalies/répliques sont capitalisées (même
    raison que `pipeline/config.py`::commentaire sur PROPN). Pas de
    lower/casefold : les entrées du fichier sont déjà en minuscules."""

    variants: set[str] = set()
    for word in whitelist:
        variants.add(word)
        variants.add(word.capitalize())
        variants.add(word.upper())
    return variants


def configure_tokenizer(
    nlp, *, hyphen_whitelist: bool = True, special_cases: bool = True,
) -> None:
    """Configure `nlp.tokenizer` en place, avant tout parsing. Voir la
    docstring du module pour l'ordre et la justification de chaque étape.

    `hyphen_whitelist=False` ET `special_cases=False` pour le `nlp` interne
    d'idiomatch (`poc_pipeline.mwe.get_matcher`) : SEUL le patch de tiret lui
    est destiné (voir le point 3 ci-dessus). Mesuré sur *The Humans* complet :
    fusionner "e-mail" en un token pour idiomatch change le compte de slop
    d'un match `to ... the ... letter` (slop=2) dans "to e-mail the rec
    letter" — un candidat de plus, mais pas un vrai idiome ("to the letter" =
    précisément). `special_cases=False` retire ce candidat sans rien changer
    au patch de tiret lui-même (effet nul sur le rappel MWE, comme mesuré
    dans le TODO)."""

    patch_dash_after_punctuation(nlp)
    if not special_cases:
        return

    surfaces = list(EMAIL_SPECIAL_CASES) + custom_lexicon.load_tokenizer_surfaces()
    if hyphen_whitelist:
        surfaces.extend(_hyphen_case_variants(load_hyphen_whitelist()))

    for surface in surfaces:
        nlp.tokenizer.add_special_case(surface, [{ORTH: surface}])
