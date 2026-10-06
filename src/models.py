"""Pydantic data models shared across parsing, extraction, validation, and output."""

from typing import Literal

from pydantic import BaseModel, Field


# Internal normalized turn used by the extraction pipeline.
class TranscriptTurn(BaseModel):
    turn_id: str
    timestamp: str
    speaker: Literal["ADVISOR", "OPTIMIZE_REP", "UNKNOWN"]
    text: str


class CleanTranscriptTurn(BaseModel):
    # Clean export shape that keeps original speaker text and metadata.
    turn_id: str
    meeting_id: str | None = None
    meeting_topic: str | None = None
    start_timestamp: str
    end_timestamp: str
    speaker_raw: str
    speaker_role: Literal["ADVISOR", "OPTIMIZE_REP", "UNKNOWN"]
    text: str


class RepairedTranscriptTurn(BaseModel):
    # Turn shape expected from the LLM repair prompt.
    turn_id: str
    meeting_id: str | None = None
    meeting_topic: str | None = None
    start_timestamp: str
    end_timestamp: str
    speaker_raw: str
    speaker_role: Literal["ADVISOR", "OPTIMIZE_REP", "OTHER", "UNKNOWN"]
    text: str
    speaker_role_reason: str


class RepairedTranscriptResult(BaseModel):
    # Container model returned by transcript-repair prompt.
    turns: list[RepairedTranscriptTurn] = Field(default_factory=list)


# Candidate extracted from one chunk before deterministic checks.
class CandidateSignal(BaseModel):
    signal_type: Literal["driver", "blocker"]
    category: str
    advisor_quote: str
    timestamp: str
    source_turn_id: str
    evidence_strength: Literal["explicit", "implied"]
    rationale: str


class ExtractionResult(BaseModel):
    # Container model returned by chunk-level extraction prompt.
    candidates: list[CandidateSignal] = Field(default_factory=list)


class EvidenceValidationItem(BaseModel):
    # One validation decision for one candidate signal.
    candidate_id: str
    verdict: Literal["SUPPORTED", "WEAK", "UNSUPPORTED"]
    recommended_action: Literal["KEEP", "ESCALATE", "DROP"]
    quote_fidelity: Literal["pass", "fail"]
    speaker_validity: Literal["pass", "fail"]
    semantic_support: Literal["pass", "partial", "fail"]
    decision_relevance: Literal["pass", "partial", "fail"]
    representative_leakage: bool
    contradiction_present: bool
    corrected_evidence_strength: Literal["explicit", "implied"]
    validation_reason: str


class EvidenceValidationResult(BaseModel):
    # Container model returned by evidence-validation prompt.
    validations: list[EvidenceValidationItem] = Field(default_factory=list)


# Final output format returned by the pipeline.
class FinalSignal(BaseModel):
    # Final ranked signal returned to downstream consumers.
    signal_type: Literal["driver", "blocker"]
    rank: int
    category: str
    advisor_quote: str
    timestamp: str
    evidence_strength: Literal["explicit", "implied"]
    rationale: str


class FinalResult(BaseModel):
    # Top-level output payload for one transcript.
    transcript_id: str
    signals: list[FinalSignal] = Field(default_factory=list)


class RankedSignal(BaseModel):
    # Ranked signal shape produced by merge-and-rank stage.
    transcript_id: str
    item_type: Literal["driver", "blocker"]
    rank: int
    category: str
    advisor_quote: str
    timestamp: str
    evidence_strength: Literal["explicit", "implied"]
    rationale: str
    source_candidate_ids: list[str] = Field(default_factory=list)
    importance_score: int
    ranking_reason: str


class RankingResult(BaseModel):
    # Container model returned by merge-and-rank prompt.
    items: list[RankedSignal] = Field(default_factory=list)
