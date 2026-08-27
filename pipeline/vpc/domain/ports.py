"""Protocols minimaux consommés par `pipeline/vpc/detectors/phrasal_verbs.py`.

Ce module N'EST PAS un portage de `booklex/domain/ports.py` — celui-ci tire
`domain/models.py` (le type `Document`) et `domain/vmwe.py`, tous deux
volontairement absents d'ici (voir `pipeline/vpc/__init__.py`). Seuls les deux
Protocol réellement utilisés par le détecteur sont redéfinis, à l'identique
côté signature."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pipeline.vpc.domain.phrasal_verbs import VpcLexicalCandidate, VpcReferenceLexeme
from pipeline.vpc.domain.resources import ResourceMetadata


@runtime_checkable
class VpcReferenceProvider(Protocol):
    """Minimal production port: frozen lexical reference, not an evaluation corpus."""

    @property
    def resource(self) -> ResourceMetadata: ...

    def vpc_references(self) -> tuple[VpcReferenceLexeme, ...]: ...


@runtime_checkable
class VpcLexiconProvider(Protocol):
    """Offline lexical candidate evidence, distinct from annotated VMWE gold."""

    @property
    def resource(self) -> ResourceMetadata: ...

    def candidates(self) -> tuple[VpcLexicalCandidate, ...]: ...

    def lookup(
        self,
        verb_lemma: str,
        particle_lemmas: tuple[str, ...],
    ) -> tuple[VpcLexicalCandidate, ...]: ...
