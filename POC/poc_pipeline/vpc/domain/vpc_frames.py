"""Vendoré depuis `book-lexical-analyzer` (booklex/domain/vpc_frames.py, commit
5bd3f5fce03d4b467f0c8431d0aa91c154cc2841, module booklex 0.1.0) — voir
`pipeline/vpc/__init__.py` pour l'attribution complète et ce qui n'est pas porté.
Seul changement : les chemins d'import vers `pipeline.vpc.domain.*`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from poc_pipeline.vpc.domain.phrasal_verbs import SyntaxSentence, SyntaxToken
from poc_pipeline.vpc.domain.resources import ResourceMetadata


FrameLevel = Literal["generic", "lexical"]
_OBJECT_RELATIONS = frozenset({"dative", "dobj", "iobj", "obj"})
_PARTICLE_COMPLEMENTS = frozenset({"dobj", "obj", "pcomp", "pobj", "prep", "xcomp"})


class VpcSyntacticFrame(BaseModel):
    """Small, interpretable spaCy frame shared by extraction and detection.

    The frame deliberately omits lexical complement lemmas and exact dependency
    distance.  TRAIN measurements showed those properties to be sparse or unstable;
    object topology and discontinuity retain the useful structural distinction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    particle_dependencies: tuple[str, ...] = Field(min_length=1)
    particle_pos: tuple[str, ...] = Field(min_length=1)
    directly_attached: tuple[bool, ...] = Field(min_length=1)
    relative_positions: tuple[Literal["before", "after"], ...] = Field(min_length=1)
    particle_count: int = Field(ge=1)
    is_discontinuous: bool
    object_positions: tuple[Literal["none", "before", "after"], ...] = Field(
        min_length=1
    )
    particle_complements: tuple[str, ...] = Field(min_length=1)


def syntactic_frame(
    sentence: SyntaxSentence,
    verb: SyntaxToken,
    particles: tuple[SyntaxToken, ...],
) -> VpcSyntacticFrame:
    """Describe an occurrence without lexicalizing arbitrary context words."""

    particle_indices = tuple(particle.index for particle in particles)
    member_indices = tuple(sorted((verb.index, *particle_indices)))
    first_particle = min(particle_indices)
    objects = tuple(
        sorted(
            "before" if token.index < first_particle else "after"
            for token in sentence.tokens
            if token.index != verb.index
            and token.dependency_head_index == verb.index
            and token.dependency_relation.casefold() in _OBJECT_RELATIONS
        )
    ) or ("none",)
    particle_complements = tuple(
        sorted(
            token.dependency_relation.casefold()
            for particle in particles
            for token in sentence.tokens
            if token.index != particle.index
            and token.dependency_head_index == particle.index
            and token.dependency_relation.casefold() in _PARTICLE_COMPLEMENTS
        )
    ) or ("none",)
    return VpcSyntacticFrame(
        particle_dependencies=tuple(
            particle.dependency_relation.casefold() for particle in particles
        ),
        particle_pos=tuple(particle.pos for particle in particles),
        directly_attached=tuple(
            particle.dependency_head_index == verb.index for particle in particles
        ),
        relative_positions=tuple(
            "after" if particle.index > verb.index else "before"
            for particle in particles
        ),
        particle_count=len(particles),
        is_discontinuous=any(
            right != left + 1 for left, right in zip(member_indices, member_indices[1:])
        ),
        object_positions=objects,
        particle_complements=particle_complements,
    )


class VpcFrameRecord(BaseModel):
    """Aggregated evidence for one generic or lexical TRAIN frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: FrameLevel
    verb_lemma: str | None = None
    particle_lemmas: tuple[str, ...] | None = None
    frame: VpcSyntacticFrame
    occurrence_count: int = Field(ge=1)
    sentence_count: int = Field(ge=1)
    document_count: int | None = Field(default=None, ge=1)
    lexical_type_count: int = Field(ge=1)
    categories: tuple[str, ...] = Field(min_length=1)
    example_ids: tuple[str, ...] = Field(min_length=1, max_length=5)


class VpcFrameInventory(BaseModel):
    """Versioned, serializable inventory reproduced solely from PARSEME TRAIN."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "booklex-vpc-frames-1.0"
    split: Literal["train"] = "train"
    provenance: ResourceMetadata
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nlp_engine: str
    nlp_model: str
    parameters: dict[str, object]
    extracted_occurrences: int = Field(ge=0)
    skipped_occurrences: int = Field(ge=0)
    generic_frames: tuple[VpcFrameRecord, ...]
    lexical_frames: tuple[VpcFrameRecord, ...]


class VpcFrameThresholdResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_occurrences: int = Field(ge=1)
    minimum_documents: int = Field(default=1, ge=1)
    selected_lexical_frames: int = Field(ge=0)
    selected_generic_frames: int = Field(ge=0)
    rejected_lexical_frames: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class SelectedVpcFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: FrameLevel
    verb_lemma: str | None = None
    particle_lemmas: tuple[str, ...] | None = None
    frame: VpcSyntacticFrame
    train_occurrences: int = Field(ge=1)
    train_documents: int = Field(default=1, ge=1)
    dev_true_positives: int = Field(ge=0)
    dev_false_positives: int = Field(ge=0)


class ProductionVpcFrame(BaseModel):
    """Frame fields required for detection, without evaluation measurements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: FrameLevel
    verb_lemma: str | None = None
    particle_lemmas: tuple[str, ...] | None = None
    frame: VpcSyntacticFrame
    train_occurrences: int = Field(ge=1)
    train_documents: int = Field(ge=1)


class ProductionVpcFrameSelection(BaseModel):
    """Frozen production policy projected from the Phase 2.8 selection artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str
    minimum_occurrences: int = Field(ge=1)
    minimum_documents: int = Field(ge=1)
    lexical_frames: tuple[ProductionVpcFrame, ...]
    generic_frames: tuple[ProductionVpcFrame, ...]
    rejected_lexical_frames: tuple[ProductionVpcFrame, ...]


class VpcFrameSelection(BaseModel):
    """Frozen DEV calibration consumed by the Phase 2.7 detector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "booklex-vpc-frame-selection-1.0"
    construction_split: Literal["train"] = "train"
    calibration_split: Literal["dev"] = "dev"
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_occurrences: int = Field(ge=1)
    minimum_documents: int = Field(default=1, ge=1)
    threshold_results: tuple[VpcFrameThresholdResult, ...] = Field(min_length=1)
    lexical_frames: tuple[SelectedVpcFrame, ...]
    generic_frames: tuple[SelectedVpcFrame, ...]
    rejected_lexical_frames: tuple[SelectedVpcFrame, ...]
    blocked_lexical_frames: int = Field(ge=0)
    blocked_generic_frames: int = Field(ge=0)
