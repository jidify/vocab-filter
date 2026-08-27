"""Vendoré depuis `book-lexical-analyzer` (booklex/domain/resources.py, commit
5bd3f5fce03d4b467f0c8431d0aa91c154cc2841, module booklex 0.1.0) — voir
`pipeline/vpc/__init__.py` pour l'attribution complète et ce qui n'est pas porté.
Copie non modifiée (mêmes classes, mêmes contraintes pydantic)."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LookupField(StrEnum):
    """Lexical fields that a provider can match without provider-specific logic."""

    TERM = "term"
    LEMMA = "lemma"
    TERM_OR_LEMMA = "term_or_lemma"


class FrequencyNormalization(StrEnum):
    """Supported interpretations of a normalized frequency value."""

    PER_TEN_THOUSAND = "per_10k"
    PER_MILLION = "per_million"
    PROPORTION = "proportion"
    ZIPF = "zipf"
    PROVIDER_DEFINED = "provider_defined"


class ResourceMetadata(BaseModel):
    """Identity, provenance, and licensing metadata for one resource version."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    resource_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str | None = None
    language: str = Field(min_length=2)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    attribution: str | None = None
    resource_date: date | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class LexicalMetadata(BaseModel):
    """Provider-neutral lexical qualifiers for a returned record."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    part_of_speech: str | None = None
    usage_register: str | None = None
    variety: str | None = None
    period: str | None = None
    tags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class LexicalRecord(BaseModel):
    """A lexical observation returned by a CorpusProvider."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    term: str = Field(min_length=1)
    lemma: str | None = Field(default=None, min_length=1)
    language: str = Field(min_length=2)
    frequency: int | None = Field(default=None, ge=0)
    normalized_frequency: float | None = Field(default=None, ge=0)
    normalization: FrequencyNormalization | None = None
    resource_entry_id: str | None = None
    metadata: LexicalMetadata = Field(default_factory=LexicalMetadata)
    provenance: ResourceMetadata

    @model_validator(mode="after")
    def validate_frequency_and_language(self) -> "LexicalRecord":
        if (self.normalized_frequency is None) != (self.normalization is None):
            raise ValueError(
                "normalized_frequency and normalization must either both be set or both be absent"
            )
        if self.provenance.language != "mul" and self.language != self.provenance.language:
            raise ValueError("record language must match the resource language")
        return self
