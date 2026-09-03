"""Vendoré depuis `book-lexical-analyzer` (booklex/domain/phrasal_verbs.py, commit
5bd3f5fce03d4b467f0c8431d0aa91c154cc2841, module booklex 0.1.0) — voir
`pipeline/vpc/__init__.py` pour l'attribution complète et ce qui n'est pas porté.
Seul changement : le chemin d'import de `ResourceMetadata`
(`pipeline.vpc.domain.resources` au lieu de `booklex.domain.resources`)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from poc_pipeline.vpc.domain.resources import ResourceMetadata


class PhrasalVerbDecision(StrEnum):
    MATCHED_REFERENCE = "matched_reference"
    NOT_MATCHED_REFERENCE = "not_matched_reference"
    REJECTED_SYNTAX = "rejected_syntax"


class PhrasalVerbEvidenceType(StrEnum):
    """Non-probabilistic evidence used to explain a detector decision."""

    PARSEME_TRAIN = "parseme_train"
    PARSEME_TRAIN_FRAME = "parseme_train_frame"
    EXTERNAL_LEXICON = "external_lexicon"
    SYNTAX = "syntax"
    CONTEXT = "context"
    ABSENT_REFERENCE = "absent_reference"


class PhrasalVerbCategory(StrEnum):
    VPC_FULL = "VPC.full"
    VPC_SEMI = "VPC.semi"
    VPC_AMBIGUOUS = "VPC.ambiguous"
    VPC_UNSPECIFIED = "VPC.unspecified"


class PhrasalVerbEvidence(BaseModel):
    """One independently inspectable reason supporting or rejecting a candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    evidence_type: PhrasalVerbEvidenceType
    detail: str = Field(min_length=1)
    provenance: ResourceMetadata | None = None


class VpcLexicalCandidate(BaseModel):
    """External lexical evidence for a possible VPC, never contextual gold truth."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    verb_lemma: str = Field(min_length=1)
    particle_lemmas: tuple[str, ...] = Field(min_length=1)
    resource_entry_id: str = Field(min_length=1)
    sense_count: int = Field(ge=1)
    provenance: ResourceMetadata


class VpcReferenceLexeme(BaseModel):
    """Corpus-free lexical view derived from frozen PARSEME TRAIN annotations."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    verb_lemma: str = Field(min_length=1)
    particle_lemmas: tuple[str, ...] = Field(min_length=1)
    category: str = Field(pattern=r"^VPC\.(full|semi)$")


class FrozenVpcReferenceLexicon(BaseModel):
    """Versioned production resource derived from TRAIN, with no corpus sentences."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "booklex-vpc-reference-lexicon-1.0"
    construction_split: str = Field(pattern=r"^train$")
    provenance: ResourceMetadata
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    references: tuple[VpcReferenceLexeme, ...]


class PhrasalVerbProvenance(BaseModel):
    """Complete, serializable configuration provenance for one detector result."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    booklex_version: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    policy_id: str | None = None
    frozen_selection_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    nlp_engine: str | None = None
    nlp_engine_version: str | None = None
    nlp_model: str | None = None
    parseme_reference: ResourceMetadata
    external_lexicon: ResourceMetadata | None = None


class SyntaxToken(BaseModel):
    """Provider-neutral syntactic token consumed by contextual detectors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_id: str = Field(min_length=1)
    sentence_id: str = Field(min_length=1)
    index: int = Field(ge=0)
    surface_form: str = Field(min_length=1)
    lemma: str = Field(min_length=1)
    pos: str = Field(min_length=1)
    fine_pos: str | None = None
    dependency_relation: str = Field(min_length=1)
    dependency_head_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    normalized_start_char: int | None = Field(default=None, ge=0)
    normalized_end_char: int | None = Field(default=None, ge=0)
    original_start_char: int | None = Field(default=None, ge=0)
    original_end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> "SyntaxToken":
        if self.end_char <= self.start_char:
            raise ValueError("syntax token must have a non-empty character span")
        for label, start, end in (
            ("normalized", self.normalized_start_char, self.normalized_end_char),
            ("original", self.original_start_char, self.original_end_char),
        ):
            if (start is None) != (end is None):
                raise ValueError(f"{label} token span must be wholly present or absent")
            if start is not None and end is not None and end <= start:
                raise ValueError(f"{label} token span must be non-empty")
        return self


class SyntaxSentence(BaseModel):
    """Sentence-level syntax independent from any concrete NLP library."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sentence_id: str = Field(min_length=1)
    text: str
    tokens: tuple[SyntaxToken, ...] = ()
    nlp_engine: str | None = None
    nlp_engine_version: str | None = None
    nlp_model: str | None = None

    @model_validator(mode="after")
    def validate_tokens(self) -> "SyntaxSentence":
        indices = tuple(token.index for token in self.tokens)
        if indices != tuple(range(len(self.tokens))):
            raise ValueError("syntax token indices must be contiguous and zero-based")
        if any(token.sentence_id != self.sentence_id for token in self.tokens):
            raise ValueError("syntax token sentence_id must match its sentence")
        if any(token.dependency_head_index >= len(self.tokens) for token in self.tokens):
            raise ValueError("dependency head index must refer to a sentence token")
        for token in self.tokens:
            if token.end_char > len(self.text):
                raise ValueError("syntax token span must fit inside sentence text")
            if self.text[token.start_char : token.end_char] != token.surface_form:
                raise ValueError("syntax token span must align with its surface form")
        return self


class PhrasalVerbDetection(BaseModel):
    """Explainable decision for one verb-particle or verb-preposition candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    detection_id: str = Field(min_length=1)
    document_id: str | None = None
    sentence_id: str = Field(min_length=1)
    verb_token_id: str = Field(min_length=1)
    verb_token_index: int = Field(ge=0)
    particle_token_ids: tuple[str, ...] = Field(min_length=1)
    particle_token_indices: tuple[int, ...] = Field(min_length=1)
    token_indices: tuple[int, ...] = Field(min_length=2)
    verb_char_span: tuple[int, int]
    particle_char_spans: tuple[tuple[int, int], ...] = Field(min_length=1)
    token_char_spans: tuple[tuple[int, int], ...] = Field(min_length=2)
    normalized_char_spans: tuple[tuple[int, int], ...] | None = None
    original_char_spans: tuple[tuple[int, int], ...] | None = None
    verb_form: str = Field(min_length=1)
    verb_lemma: str = Field(min_length=1)
    particle_forms: tuple[str, ...] = Field(min_length=1)
    particle_lemmas: tuple[str, ...] = Field(min_length=1)
    normalized_expression: str = Field(min_length=3)
    observed_expression: str = Field(min_length=3)
    is_discontinuous: bool
    category: PhrasalVerbCategory | None
    reference_categories: tuple[str, ...] = ()
    decision: PhrasalVerbDecision
    rule_id: str = Field(min_length=1)
    decision_reason: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    provenance: PhrasalVerbProvenance
    reference_provenance: tuple[ResourceMetadata, ...] = ()
    evidence: tuple[PhrasalVerbEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_detection(self) -> "PhrasalVerbDetection":
        if tuple(sorted(set(self.particle_token_indices))) != self.particle_token_indices:
            raise ValueError("particle indices must be unique and strictly increasing")
        expected_indices = tuple(sorted((self.verb_token_index, *self.particle_token_indices)))
        if self.token_indices != expected_indices:
            raise ValueError("token_indices must contain the verb and particles")
        if len(self.particle_char_spans) != len(self.particle_token_indices):
            raise ValueError("particle spans must match particle indices")
        if len(self.token_char_spans) != len(self.token_indices):
            raise ValueError("token spans must match token indices")
        for spans in (
            (self.verb_char_span,),
            self.particle_char_spans,
            self.token_char_spans,
            self.normalized_char_spans or (),
            self.original_char_spans or (),
        ):
            if any(end <= start for start, end in spans):
                raise ValueError("detection character spans must be non-empty")
        if self.normalized_char_spans is not None and len(self.normalized_char_spans) != len(
            self.token_indices
        ):
            raise ValueError("normalized spans must match token indices")
        if self.original_char_spans is not None and len(self.original_char_spans) != len(
            self.token_indices
        ):
            raise ValueError("original spans must match token indices")
        expected_discontinuous = any(
            right != left + 1
            for left, right in zip(self.token_indices, self.token_indices[1:])
        )
        if self.is_discontinuous != expected_discontinuous:
            raise ValueError("is_discontinuous must match token indices")
        if self.decision == PhrasalVerbDecision.REJECTED_SYNTAX and self.category is not None:
            raise ValueError("a syntax rejection cannot have a VPC category")
        if self.decision != PhrasalVerbDecision.REJECTED_SYNTAX and self.category is None:
            raise ValueError("an accepted syntactic candidate requires a VPC category")
        if self.detector_version != self.provenance.detector_version:
            raise ValueError("detector version must match provenance")
        return self

    @property
    def is_detection(self) -> bool:
        return self.decision != PhrasalVerbDecision.REJECTED_SYNTAX
