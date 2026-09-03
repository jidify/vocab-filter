"""Fournisseur WordNet pour `PhrasalVerbDetector`, basé sur `nltk` — PAS un
portage de `booklex/resources/wordnet.py`.

Le projet source lit un fichier `index.verb` WNDB local (Open English WordNet
2025, format texte ligne par ligne). `nltk` est déjà une dépendance de
vocab-filter (`pipeline/senses.py`, `pipeline/select.py`, etc. l'utilisent
tous via `nltk.corpus.wordnet`) et expose le même genre d'information
(lemmes verbaux multi-mots + nombre de synsets) sans qu'il faille télécharger
ou maintenir un second index WordNet séparé — voir le plan, Partie 4
"Lot 2" / point 11.

`ENGLISH_VPC_PARTICLES` (liste fermée des particules autorisées en seconde
position d'un lemme verbal multi-mots) est reprise telle quelle de
`booklex/resources/wordnet.py` (même auteur, même projet sœur) : c'est une
liste de mots fermés, pas du code substantiel, mais l'origine est documentée
par honnêteté."""

from __future__ import annotations

from nltk.corpus import wordnet as nwn

from poc_pipeline.vpc.domain.phrasal_verbs import VpcLexicalCandidate
from poc_pipeline.vpc.domain.resources import ResourceMetadata

ENGLISH_VPC_PARTICLES = frozenset(
    {
        "about",
        "across",
        "after",
        "along",
        "around",
        "at",
        "away",
        "back",
        "by",
        "down",
        "forth",
        "forward",
        "in",
        "into",
        "off",
        "on",
        "out",
        "over",
        "round",
        "through",
        "together",
        "up",
    }
)

_RESOURCE = ResourceMetadata(
    resource_id="nltk-wordnet-vpc",
    name="WordNet (via NLTK)",
    language="en",
    source="nltk.corpus.wordnet",
    license="WordNet 3.0 license (Princeton University) — see NLTK's wordnet corpus README",
    attribution="Princeton University WordNet; distributed via NLTK",
)


class NltkWordNetVpcProvider:
    """Read VPC-shaped candidate lemmas from NLTK's local WordNet corpus.

    WordNet records multiword verb lemmas and senses, not a VPC label.
    Returned records are therefore candidate evidence only — the contextual
    detector (`pipeline.vpc.detectors.phrasal_verbs.PhrasalVerbDetector`)
    remains responsible for every acceptance decision."""

    def __init__(self) -> None:
        self._candidates = _build_candidates()
        self._index = {
            (item.verb_lemma, item.particle_lemmas): (item,) for item in self._candidates
        }

    @property
    def resource(self) -> ResourceMetadata:
        return _RESOURCE

    def candidates(self) -> tuple[VpcLexicalCandidate, ...]:
        return self._candidates

    def lookup(
        self,
        verb_lemma: str,
        particle_lemmas: tuple[str, ...],
    ) -> tuple[VpcLexicalCandidate, ...]:
        key = (
            verb_lemma.casefold().strip(),
            tuple(item.casefold().strip() for item in particle_lemmas),
        )
        return self._index.get(key, ())


def _build_candidates() -> tuple[VpcLexicalCandidate, ...]:
    candidates: list[VpcLexicalCandidate] = []
    for lemma in nwn.all_lemma_names(pos="v"):
        lemma = lemma.casefold()
        parts = tuple(lemma.split("_"))
        if len(parts) < 2 or not all(parts):
            continue
        verb, particles = parts[0], parts[1:]
        if any(particle not in ENGLISH_VPC_PARTICLES for particle in particles):
            continue
        sense_count = len(nwn.synsets(lemma, pos="v"))
        if sense_count < 1:
            continue
        candidates.append(
            VpcLexicalCandidate(
                verb_lemma=verb,
                particle_lemmas=particles,
                resource_entry_id=lemma,
                sense_count=sense_count,
                provenance=_RESOURCE,
            )
        )
    return tuple(
        sorted(candidates, key=lambda item: (item.verb_lemma, item.particle_lemmas))
    )
