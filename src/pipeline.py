"""Core pipeline orchestration: parse/chunk/extract/verify/validate/rank."""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from src.llm import structured_call
from src.config import (
    CHUNKING_STRATEGY,
    CHUNKING_VERSION,
    CHUNK_OVERLAP,
    EXTRACTION_MODEL,
    MAX_CHUNK_TOKENS,
    MAX_CHUNK_TURNS,
    SOURCE_PARSER_VERSION,
    TARGET_CHUNK_TURNS,
    TOKENIZER_NAME,
    UNKNOWN_RATIO_MIN_TURNS,
    UNKNOWN_RATIO_THRESHOLD,
    VALIDATION_MODEL,
)
from src.models import (
    CandidateSignal,
    EvidenceValidationResult,
    ExtractionResult,
    FinalResult,
    FinalSignal,
    RankingResult,
    RepairedTranscriptResult,
    TranscriptTurn,
)
from src.parser import parse_transcript
from src.parser import parse_transcript_clean_json


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_prompt(prompt_filename: str) -> str:
    # All prompts live under advisor-signal-extractor/prompts.
    prompt_path = PROJECT_ROOT / "prompts" / prompt_filename
    return prompt_path.read_text(encoding="utf-8")


def _estimate_tokens_for_text(text: str) -> int:
    # Lightweight approximation: ~1 token per 4 chars.
    return max(1, (len(text) + 3) // 4)


def _estimate_tokens_for_turn(turn: TranscriptTurn) -> int:
    payload_text = f"{turn.turn_id} {turn.timestamp} {turn.speaker} {turn.text}"
    return _estimate_tokens_for_text(payload_text)


def _estimate_tokens_for_chunk(chunk_turns: list[TranscriptTurn]) -> int:
    return sum(_estimate_tokens_for_turn(turn) for turn in chunk_turns)


def _extract_meeting_id(clean_turns: list[dict[str, object]]) -> str | None:
    # Meeting metadata is duplicated across clean turns, so first turn is enough.
    if not clean_turns:
        return None

    meeting_id = clean_turns[0].get("meeting_id")
    if meeting_id is None:
        return None

    return str(meeting_id)


def _normalize_speaker_for_pipeline(role: str) -> str:
    return role if role in {"ADVISOR", "OPTIMIZE_REP"} else "UNKNOWN"


def repair_transcript_with_llm(
    raw_text: str,
    model: str,
) -> list[TranscriptTurn]:
    print("[pipeline] Using repair prompt because parser output was not reliable.")
    prompt = _read_prompt("01_transcript_repair.txt")
    repaired = structured_call(
        model=model,
        prompt=prompt,
        content=raw_text,
        output_model=RepairedTranscriptResult,
    )

    turns: list[TranscriptTurn] = []
    for index, repaired_turn in enumerate(repaired.turns, start=1):
        turns.append(
            TranscriptTurn(
                turn_id=f"turn-{index:03d}",
                timestamp=repaired_turn.start_timestamp,
                speaker=_normalize_speaker_for_pipeline(repaired_turn.speaker_role),
                text=" ".join(repaired_turn.text.split()),
            )
        )

    if not turns:
        raise ValueError("Transcript repair returned no turns.")

    print(f"[pipeline] Repair produced {len(turns)} turns.")

    return turns


def _evaluate_repair_need_for_turns(turns: list[TranscriptTurn]) -> tuple[bool, str, int, float]:
    """Return (needs_repair, reason, unknown_count, unknown_ratio)."""
    unknown_count = sum(1 for turn in turns if turn.speaker == "UNKNOWN")
    unknown_ratio = (unknown_count / len(turns)) if turns else 1.0

    # Hard fail: all speakers unknown means the parser output is unusable.
    if turns and unknown_count == len(turns):
        return True, "all_speakers_unknown", unknown_count, unknown_ratio

    # Soft fail: only apply ratio threshold when transcript is long enough.
    if len(turns) >= UNKNOWN_RATIO_MIN_TURNS and unknown_ratio >= UNKNOWN_RATIO_THRESHOLD:
        return True, "unknown_ratio_threshold", unknown_count, unknown_ratio

    return False, "parser_usable", unknown_count, unknown_ratio


def parse_turns_with_conditional_repair(
    raw_text: str,
    model: str,
) -> list[TranscriptTurn]:
    print("[pipeline] Step 1: trying normal parser.")
    try:
        turns = parse_transcript(raw_text)
    except ValueError:
        print("[pipeline] Normal parser failed (ValueError). Triggering repair prompt.")
        return repair_transcript_with_llm(raw_text, model)

    needs_repair, reason, unknown_count, unknown_ratio = _evaluate_repair_need_for_turns(turns)
    print(
        f"[pipeline] Normal parser succeeded with {len(turns)} turns, "
        f"UNKNOWN speakers: {unknown_count} ({unknown_ratio:.2%})."
    )
    if needs_repair:
        print(f"[pipeline] Triggering repair prompt ({reason}).")
        return repair_transcript_with_llm(raw_text, model)

    print("[pipeline] Parser output is usable. Keeping normal parsed turns.")
    return turns


def check_if_conditional_repair_needed(raw_text: str) -> tuple[bool, str]:
    """Return (needs_repair, reason) without calling the LLM repair prompt."""
    print("[pipeline-check] Evaluating whether conditional repair is needed.")
    try:
        turns = parse_transcript(raw_text)
    except ValueError:
        print("[pipeline-check] Parser failed -> repair needed (parser_failed).")
        return True, "parser_failed"

    needs_repair, reason, unknown_count, unknown_ratio = _evaluate_repair_need_for_turns(turns)
    print(
        f"[pipeline-check] Parsed {len(turns)} turns; UNKNOWN speakers: "
        f"{unknown_count} ({unknown_ratio:.2%})."
    )
    print(f"[pipeline-check] Repair {'needed' if needs_repair else 'not needed'} ({reason}).")
    return needs_repair, reason


def print_conditional_repair_decision(raw_text: str, source_label: str) -> tuple[bool, str]:
    """Print per-source decision and return (needs_repair, reason)."""
    needs_repair, reason = check_if_conditional_repair_needed(raw_text)
    status = "needed" if needs_repair else "not needed"
    print(f"{source_label}: conditional repair {status} ({reason})")
    return needs_repair, reason


def chunk_turns(
    turns: list[TranscriptTurn],
    chunk_size: int = TARGET_CHUNK_TURNS,
    overlap: int = CHUNK_OVERLAP,
    max_chunk_tokens: int = MAX_CHUNK_TOKENS,
) -> tuple[list[list[TranscriptTurn]], list[str]]:
    # Config guards keep bad settings from creating empty or infinite loops.
    if chunk_size <= 0:
        raise ValueError("TARGET_CHUNK_TURNS must be greater than 0.")

    if MAX_CHUNK_TURNS < chunk_size:
        raise ValueError("MAX_CHUNK_TURNS must be greater than or equal to TARGET_CHUNK_TURNS.")

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than TARGET_CHUNK_TURNS.")

    if max_chunk_tokens <= 0:
        raise ValueError("MAX_CHUNK_TOKENS must be greater than 0.")

    chunks: list[list[TranscriptTurn]] = []
    boundary_reasons: list[str] = []
    start = 0

    while start < len(turns):
        # First try the target turn window, then shrink if token budget is exceeded.
        max_end = min(start + chunk_size, len(turns))
        end = max_end
        token_limited = False

        while end > start:
            candidate_chunk = turns[start:end]
            if _estimate_tokens_for_chunk(candidate_chunk) <= max_chunk_tokens:
                break
            token_limited = True
            end -= 1

        if end == start:
            # If one turn alone exceeds the token budget, keep forward progress with a single-turn chunk.
            end = start + 1

        current_chunk = turns[start:end]
        chunks.append(current_chunk)

        if end >= len(turns):
            boundary_reasons.append("end_of_transcript")
        elif token_limited:
            boundary_reasons.append("max_tokens_reached")
        elif len(current_chunk) >= MAX_CHUNK_TURNS:
            boundary_reasons.append("max_turns_reached")
        elif len(current_chunk) >= chunk_size:
            boundary_reasons.append("target_size_reached")
        else:
            boundary_reasons.append("natural_boundary")

        if end >= len(turns):
            break

        # Keep a small context bridge between neighboring chunks.
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks, boundary_reasons


def save_chunk_metadata_json(
    transcript_id: str,
    chunks: list[list[TranscriptTurn]],
    boundary_reasons: list[str],
    output_path: Path,
    chunk_size: int,
    overlap: int,
    max_chunk_tokens: int,
    meeting_id: str | None,
    clean_turns_by_id: dict[str, dict[str, object]] | None,
) -> None:
    # Save chunk boundaries and turn lineage for debugging and reproducibility.
    chunk_items: list[dict[str, object]] = []

    previous_chunk_turn_ids: set[str] = set()

    for chunk_index, chunk in enumerate(chunks, start=1):
        first_turn = chunk[0]
        last_turn = chunk[-1]
        current_turn_ids = [turn.turn_id for turn in chunk]
        overlap_turn_ids = [turn_id for turn_id in current_turn_ids if turn_id in previous_chunk_turn_ids]
        is_overlap_with_previous = bool(overlap_turn_ids)
        speaker_roles = sorted({turn.speaker for turn in chunk})
        chunk_id_prefix = f"meeting_{meeting_id}" if meeting_id else f"transcript_{transcript_id}"

        # Store full normalized turn payload to preserve downstream evidence traceability.
        turns_payload: list[dict[str, object]] = []
        for turn in chunk:
            clean_turn = clean_turns_by_id.get(turn.turn_id) if clean_turns_by_id else None
            if clean_turn is not None:
                turns_payload.append(
                    {
                        "turn_id": clean_turn.get("turn_id", turn.turn_id),
                        "start_timestamp": clean_turn.get("start_timestamp", turn.timestamp),
                        "end_timestamp": clean_turn.get("end_timestamp"),
                        "speaker_raw": clean_turn.get("speaker_raw", turn.speaker),
                        "speaker_role": clean_turn.get("speaker_role", turn.speaker),
                        "text": clean_turn.get("text", turn.text),
                    }
                )
            else:
                turns_payload.append(
                    {
                        "turn_id": turn.turn_id,
                        "start_timestamp": turn.timestamp,
                        "end_timestamp": None,
                        "speaker_raw": turn.speaker,
                        "speaker_role": turn.speaker,
                        "text": turn.text,
                    }
                )

        chunk_items.append(
            {
                "chunk_id": f"{chunk_id_prefix}-chunk-{chunk_index:03d}",
                "chunk_index": chunk_index,
                "turn_count": len(chunk),
                "first_turn_id": first_turn.turn_id,
                "last_turn_id": last_turn.turn_id,
                "source_turn_ids": current_turn_ids,
                "start_timestamp": first_turn.timestamp,
                "end_timestamp": last_turn.timestamp,
                "estimated_tokens": _estimate_tokens_for_chunk(chunk),
                "is_overlap_with_previous": is_overlap_with_previous,
                "overlap_turn_ids": overlap_turn_ids,
                "boundary_reason": boundary_reasons[chunk_index - 1],
                "speaker_roles": speaker_roles,
                "turns": turns_payload,
            }
        )
        previous_chunk_turn_ids = set(current_turn_ids)

    payload = {
        "transcript_id": transcript_id,
        "meeting_id": meeting_id,
        "chunking_version": CHUNKING_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "chunking_strategy": CHUNKING_STRATEGY,
        "target_chunk_turns": TARGET_CHUNK_TURNS,
        "max_chunk_turns": MAX_CHUNK_TURNS,
        "chunk_size": chunk_size,
        "overlap_turns": overlap,
        "max_chunk_tokens": max_chunk_tokens,
        "tokenizer": TOKENIZER_NAME,
        "chunk_count": len(chunks),
        "source_parser_version": SOURCE_PARSER_VERSION,
        "chunks": chunk_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pipeline] Stored chunk metadata: {output_path}")


def save_extract_json(
    transcript_id: str,
    candidates: list[CandidateSignal],
    output_path: Path,
    extraction_model: str,
    chunk_count: int,
) -> None:
    # Persist raw extraction output before deterministic filtering and ranking.
    payload = {
        "transcript_id": transcript_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "extraction_model": extraction_model,
        "chunk_count": chunk_count,
        "candidate_count": len(candidates),
        "candidates": [candidate.model_dump() for candidate in candidates],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pipeline] Stored extract output: {output_path}")


def save_evidence_validation_json(
    transcript_id: str,
    validation_result: EvidenceValidationResult,
    output_path: Path,
    validation_model: str,
    candidate_count: int,
    deterministic_candidates: list[CandidateSignal],
    ambiguous_candidates: list[CandidateSignal],
) -> None:
    # This file is the Step-4 artifact used by Step-5 resume mode.
    # It contains only KEEP validations enriched with candidate fields.

    # Build lookup so LLM candidate_id (candidate-001, ...) can be enriched
    # with the original extracted candidate fields.
    ambiguous_by_id = {
        f"candidate-{idx:03d}": candidate
        for idx, candidate in enumerate(ambiguous_candidates, start=1)
    }

    # Deterministic candidates were already grounded by rule-based checks,
    # so we persist them as synthetic KEEP validations with full candidate fields.
    deterministic_validations = [
        {
            "candidate_id": f"deterministic-{idx:03d}",
            "verdict": "SUPPORTED",
            "recommended_action": "KEEP",
            "quote_fidelity": "pass",
            "speaker_validity": "pass",
            "semantic_support": "pass",
            "decision_relevance": "pass",
            "representative_leakage": False,
            "contradiction_present": False,
            "corrected_evidence_strength": candidate.evidence_strength,
            "validation_reason": "Deterministic grounding checks passed.",
            "signal_type": candidate.signal_type,
            "category": candidate.category,
            "advisor_quote": candidate.advisor_quote,
            "timestamp": candidate.timestamp,
            "source_turn_id": candidate.source_turn_id,
            "evidence_strength": candidate.evidence_strength,
            "rationale": candidate.rationale,
        }
        for idx, candidate in enumerate(deterministic_candidates, start=1)
    ]

    # Keep only LLM validations whose final action is KEEP.
    kept_validations = [
        item
        for item in validation_result.validations
        if item.recommended_action == "KEEP"
    ]

    # Enrich LLM KEEP validations with candidate fields from extract output.
    llm_keep_validations: list[dict[str, object]] = []
    for item in kept_validations:
        validation_payload = item.model_dump()
        source_candidate = ambiguous_by_id.get(item.candidate_id)

        if source_candidate is not None:
            validation_payload.update(
                {
                    "signal_type": source_candidate.signal_type,
                    "category": source_candidate.category,
                    "advisor_quote": source_candidate.advisor_quote,
                    "timestamp": source_candidate.timestamp,
                    "source_turn_id": source_candidate.source_turn_id,
                    "evidence_strength": source_candidate.evidence_strength,
                    "rationale": source_candidate.rationale,
                }
            )

        llm_keep_validations.append(validation_payload)

    combined_validations = deterministic_validations + llm_keep_validations

    payload = {
        "transcript_id": transcript_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "validation_model": validation_model,
        "candidate_count": candidate_count,
        "validation_count": len(combined_validations),
        "validations": combined_validations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pipeline] Stored evidence validation: {output_path}")


def load_turns_from_chunk_metadata(metadata_path: Path) -> list[TranscriptTurn]:
    # Rebuild parser-like turns from saved chunk metadata.
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])

    turns_by_id: dict[str, TranscriptTurn] = {}
    ordered_turn_ids: list[str] = []

    for chunk in chunks:
        for item in chunk.get("turns", []):
            turn_id = str(item.get("turn_id", ""))
            if not turn_id:
                continue

            if turn_id not in turns_by_id:
                # Keep first occurrence only to avoid duplicates from chunk overlap.
                ordered_turn_ids.append(turn_id)
                speaker_role = str(item.get("speaker_role", "UNKNOWN"))
                normalized_speaker = (
                    speaker_role
                    if speaker_role in {"ADVISOR", "OPTIMIZE_REP", "UNKNOWN"}
                    else "UNKNOWN"
                )

                turns_by_id[turn_id] = TranscriptTurn(
                    turn_id=turn_id,
                    timestamp=str(item.get("start_timestamp", "")),
                    speaker=normalized_speaker,
                    text=str(item.get("text", "")),
                )

    return [turns_by_id[turn_id] for turn_id in ordered_turn_ids]


def load_ranking_candidates_from_evidence_json(
    evidence_path: Path,
) -> list[dict[str, object]]:
    # Rebuild Step-5 input from the persisted evidence validation artifact.
    # This path is fully deterministic Python logic (no model call here).
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    validations = payload.get("validations", [])
    required_fields = {
        "candidate_id",
        "signal_type",
        "category",
        "advisor_quote",
        "timestamp",
        "evidence_strength",
        "rationale",
    }

    ranking_candidates: list[dict[str, object]] = []
    for item in validations:
        if item.get("recommended_action") != "KEEP":
            continue

        # Skip malformed rows that cannot be ranked safely.
        if not required_fields.issubset(item):
            continue

        # Keep confidence separate from ranking importance.
        evidence_confidence = (
            "high"
            if (
                item.get("verdict") == "SUPPORTED"
                and item.get("quote_fidelity") == "pass"
                and item.get("speaker_validity") == "pass"
            )
            else "medium"
        )

        ranking_candidates.append(
            {
                "source_candidate_id": item["candidate_id"],
                "signal_type": item["signal_type"],
                "category": item["category"],
                "advisor_quote": item["advisor_quote"],
                "timestamp": item["timestamp"],
                "evidence_strength": item["evidence_strength"],
                "rationale": item["rationale"],
                "evidence_confidence": evidence_confidence,
            }
        )

    return ranking_candidates


def extract_candidates(
    turns: list[TranscriptTurn],
    model: str,
) -> ExtractionResult:
    prompt = _read_prompt("02_signal_extraction.txt")
    content = json.dumps([turn.model_dump() for turn in turns], indent=2)

    return structured_call(
        model=model,
        prompt=prompt,
        content=content,
        output_model=ExtractionResult,
    )


def verify_candidate(
    candidate: CandidateSignal,
    turns: list[TranscriptTurn],
) -> bool:
    # Deterministic grounding checks: turn exists, advisor-only, exact quote, exact timestamp.
    turn_by_id = {turn.turn_id: turn for turn in turns}
    source_turn = turn_by_id.get(candidate.source_turn_id)

    if source_turn is None:
        return False

    if source_turn.speaker != "ADVISOR":
        return False

    if candidate.advisor_quote not in source_turn.text:
        return False

    if candidate.timestamp != source_turn.timestamp:
        return False

    return True


def is_ambiguous_candidate(candidate: CandidateSignal) -> bool:
    # Simple rule: implied evidence goes to LLM evidence judge.
    return candidate.evidence_strength == "implied"


def split_deterministic_and_ambiguous(
    candidates: list[CandidateSignal],
) -> tuple[list[CandidateSignal], list[CandidateSignal]]:
    deterministic_keep: list[CandidateSignal] = []
    ambiguous: list[CandidateSignal] = []

    for candidate in candidates:
        if is_ambiguous_candidate(candidate):
            ambiguous.append(candidate)
        else:
            deterministic_keep.append(candidate)

    return deterministic_keep, ambiguous


def apply_evidence_actions(
    ambiguous_candidates: list[CandidateSignal],
    evidence_validation: EvidenceValidationResult,
) -> list[CandidateSignal]:
    # Candidate IDs are assigned in-order as candidate-001, candidate-002, etc.
    action_by_candidate_id = {
        item.candidate_id: item.recommended_action for item in evidence_validation.validations
    }

    # Keep only candidates that validator marked as KEEP.
    kept: list[CandidateSignal] = []
    for idx, candidate in enumerate(ambiguous_candidates, start=1):
        candidate_id = f"candidate-{idx:03d}"
        action = action_by_candidate_id.get(candidate_id)

        if action == "KEEP":
            kept.append(candidate)

    return kept


def _build_ranking_candidates_payload(
    deterministic_candidates: list[CandidateSignal],
    ambiguous_candidates: list[CandidateSignal],
    evidence_validation: EvidenceValidationResult,
    should_run_evidence_judge: bool,
) -> list[dict[str, object]]:
    # Keep evidence confidence separate from ranking importance.
    # Confidence describes how reliable the evidence is, not how important the signal is.
    ranking_candidates: list[dict[str, object]] = []

    for idx, candidate in enumerate(deterministic_candidates, start=1):
        ranking_candidates.append(
            {
                "source_candidate_id": f"deterministic-{idx:03d}",
                "signal_type": candidate.signal_type,
                "category": candidate.category,
                "advisor_quote": candidate.advisor_quote,
                "timestamp": candidate.timestamp,
                "evidence_strength": candidate.evidence_strength,
                "rationale": candidate.rationale,
                "evidence_confidence": "high",
            }
        )

    if should_run_evidence_judge:
        validation_by_id = {
            item.candidate_id: item for item in evidence_validation.validations
        }

        for idx, candidate in enumerate(ambiguous_candidates, start=1):
            candidate_id = f"candidate-{idx:03d}"
            validation = validation_by_id.get(candidate_id)

            if validation is None or validation.recommended_action != "KEEP":
                continue

            evidence_confidence = (
                "high"
                if (
                    validation.verdict == "SUPPORTED"
                    and validation.quote_fidelity == "pass"
                    and validation.speaker_validity == "pass"
                )
                else "medium"
            )

            ranking_candidates.append(
                {
                    "source_candidate_id": candidate_id,
                    "signal_type": candidate.signal_type,
                    "category": candidate.category,
                    "advisor_quote": candidate.advisor_quote,
                    "timestamp": candidate.timestamp,
                    "evidence_strength": candidate.evidence_strength,
                    "rationale": candidate.rationale,
                    "evidence_confidence": evidence_confidence,
                }
            )
    else:
        for idx, candidate in enumerate(ambiguous_candidates, start=1):
            ranking_candidates.append(
                {
                    "source_candidate_id": f"candidate-{idx:03d}",
                    "signal_type": candidate.signal_type,
                    "category": candidate.category,
                    "advisor_quote": candidate.advisor_quote,
                    "timestamp": candidate.timestamp,
                    "evidence_strength": candidate.evidence_strength,
                    "rationale": candidate.rationale,
                    "evidence_confidence": "medium",
                }
            )

    return ranking_candidates


def merge_and_rank(
    transcript_id: str,
    turns: list[TranscriptTurn],
    ranking_candidates: list[dict[str, object]],
    model: str,
) -> RankingResult:
    # Step-5 model call: deduplicate and rank supported candidates.
    prompt = _read_prompt("04_merge_and_rank.txt")
    content = json.dumps(
        {
            "transcript_id": transcript_id,
            "transcript": [turn.model_dump() for turn in turns],
            "candidates": ranking_candidates,
            "rubric": {
                "decision_impact": "0-3",
                "gating_or_commitment": "0-3",
                "specificity": "0-2",
                "immediacy": "0-1",
                "emphasis_or_recurrence": "0-1",
                "total": "0-10",
            },
            "importance_threshold_guidance": {
                "top_ranked": "9-10",
                "meaningful": "7-8",
                "secondary": "5-6",
                "normally_exclude": "below 5",
            },
        },
        indent=2,
    )

    return structured_call(
        model=model,
        prompt=prompt,
        content=content,
        output_model=RankingResult,
    )


def _ranking_result_to_final_result(
    transcript_id: str,
    ranking_result: RankingResult,
) -> FinalResult:
    # Convert richer ranking output into the existing FinalResult shape.
    signals = [
        FinalSignal(
            signal_type=item.item_type,
            rank=item.rank,
            category=item.category,
            advisor_quote=item.advisor_quote,
            timestamp=item.timestamp,
            evidence_strength=item.evidence_strength,
            rationale=item.rationale,
        )
        for item in ranking_result.items
    ]
    return FinalResult(transcript_id=transcript_id, signals=signals)


def save_ranking_json(
    transcript_id: str,
    ranking_result: RankingResult,
    output_path: Path,
    ranking_model: str,
    candidate_count: int,
) -> None:
    # Save detailed ranking artifact with scoring metadata.
    payload = {
        "transcript_id": transcript_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ranking_model": ranking_model,
        "rubric_version": "importance_v1_0_10",
        "candidate_count": candidate_count,
        "selected_count": len(ranking_result.items),
        "drivers_selected": sum(1 for item in ranking_result.items if item.item_type == "driver"),
        "blockers_selected": sum(1 for item in ranking_result.items if item.item_type == "blocker"),
        "threshold_guidance": {
            "top_ranked": "9-10",
            "meaningful": "7-8",
            "secondary": "5-6",
            "normally_exclude": "below 5",
        },
        "rankings": [item.model_dump() for item in ranking_result.items],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pipeline] Stored ranking output: {output_path}")


def save_final_result_json(
    result: FinalResult,
    output_path: Path,
) -> None:
    # Deterministic Step-6 writer.
    # Keep only the final user-requested fields, with item_type naming.
    # Final JSON is intentionally minimal and stable for downstream consumers.
    payload = [
        {
            "transcript_id": result.transcript_id,
            "item_type": signal.signal_type,
            "rank": signal.rank,
            "category": signal.category,
            "advisor_quote": signal.advisor_quote,
            "timestamp": signal.timestamp,
            "evidence_strength": signal.evidence_strength,
            "rationale": signal.rationale,
        }
        for signal in result.signals
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pipeline] Stored final result: {output_path}")


def validate_and_rank(
    transcript_id: str,
    turns: list[TranscriptTurn],
    candidates: list[CandidateSignal],
    model: str,
) -> FinalResult:
    # Backward-compatible wrapper for tests and callers that expect FinalResult.
    ranking_candidates = [
        {
            "source_candidate_id": f"candidate-{idx:03d}",
            "signal_type": candidate.signal_type,
            "category": candidate.category,
            "advisor_quote": candidate.advisor_quote,
            "timestamp": candidate.timestamp,
            "evidence_strength": candidate.evidence_strength,
            "rationale": candidate.rationale,
            "evidence_confidence": "medium",
        }
        for idx, candidate in enumerate(candidates, start=1)
    ]

    ranking_result = merge_and_rank(
        transcript_id=transcript_id,
        turns=turns,
        ranking_candidates=ranking_candidates,
        model=model,
    )
    return _ranking_result_to_final_result(transcript_id, ranking_result)


def validate_evidence(
    transcript_id: str,
    turns: list[TranscriptTurn],
    candidates: list[CandidateSignal],
    model: str,
) -> EvidenceValidationResult:
    prompt = _read_prompt("03_evidence_validation.txt")

    payload_candidates: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_payload = candidate.model_dump()
        candidate_payload["candidate_id"] = f"candidate-{idx:03d}"
        payload_candidates.append(candidate_payload)

    content = json.dumps(
        {
            "transcript_id": transcript_id,
            "transcript": [turn.model_dump() for turn in turns],
            "candidates": payload_candidates,
        },
        indent=2,
    )

    return structured_call(
        model=model,
        prompt=prompt,
        content=content,
        output_model=EvidenceValidationResult,
    )


def enforce_output_rules(result: FinalResult) -> FinalResult:
    # Enforce hard caps and independent ranking for each signal type.
    drivers = [signal for signal in result.signals if signal.signal_type == "driver"][:3]
    blockers = [signal for signal in result.signals if signal.signal_type == "blocker"][:3]

    for rank, signal in enumerate(drivers, start=1):
        signal.rank = rank

    for rank, signal in enumerate(blockers, start=1):
        signal.rank = rank

    result.signals = drivers + blockers
    return result


def process_transcript(
    transcript_id: str,
    raw_text: str,
    model: str | None = None,
    cleaned_output_path: Path | None = None,
    extract_output_path: Path | None = None,
    evidence_validation_output_path: Path | None = None,
    ranking_output_path: Path | None = None,
    final_output_path: Path | None = None,
    check_only: bool = False,
    chunk_metadata_output_path: Path | None = None,
    start_from_step5: bool = False,
) -> FinalResult:
    # This function supports two run modes:
    # 1) full mode (Steps 1-6)
    # 2) resume mode (start_from_step5=True) that skips Steps 1-4

    # Check-only mode: parse + cleaned JSON + repair decision, then stop.
    if check_only:
        print(f"[pipeline] Processing transcript (check-only): {transcript_id}")
        print("[pipeline] Parsing transcript and preparing cleaned JSON.")
        cleaned = parse_transcript_clean_json(raw_text)

        if cleaned_output_path is not None:
            cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_output_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
            print(f"[pipeline] Stored cleaned transcript: {cleaned_output_path}")

        needs_repair, reason = check_if_conditional_repair_needed(raw_text)
        print(
            f"[pipeline] Decision: conditional repair {'needed' if needs_repair else 'not needed'} ({reason})"
        )

        print("[pipeline] Stopped after conditional check as requested.")
        return FinalResult(transcript_id=transcript_id, signals=[])

    print(f"[pipeline] Processing transcript: {transcript_id}")
    logger.info("Processing transcript: %s", transcript_id)

    # Keep optional override for tests/manual runs while defaulting to config models.
    extraction_model = model or EXTRACTION_MODEL
    validation_model = model or VALIDATION_MODEL

    # Persist cleaned parsed transcript when we are using raw transcript input.
    if cleaned_output_path is not None and not start_from_step5:
        cleaned = parse_transcript_clean_json(raw_text)
        cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_output_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
        print(f"[pipeline] Stored cleaned transcript: {cleaned_output_path}")

    if start_from_step5:
        # Step-5 resume mode uses existing artifacts instead of raw transcript parsing.
        if ranking_output_path is None:
            raise ValueError("start_from_step5=True requires ranking_output_path.")

        if evidence_validation_output_path is None:
            raise ValueError("start_from_step5=True requires evidence_validation_output_path.")

        if chunk_metadata_output_path is None:
            raise ValueError("start_from_step5=True requires chunk_metadata_output_path.")

        if not evidence_validation_output_path.exists():
            raise FileNotFoundError(
                f"Evidence validation JSON not found: {evidence_validation_output_path}"
            )

        if not chunk_metadata_output_path.exists():
            raise FileNotFoundError(f"Chunk metadata JSON not found: {chunk_metadata_output_path}")

        print("[pipeline] start_from_step5 enabled. Skipping Steps 1-4.")
        turns = load_turns_from_chunk_metadata(chunk_metadata_output_path)
        ranking_candidates = load_ranking_candidates_from_evidence_json(
            evidence_validation_output_path
        )
        print(f"[pipeline] Loaded turns from metadata: {len(turns)}")
        print(f"[pipeline] Loaded ranking candidates from evidence JSON: {len(ranking_candidates)}")

        if not ranking_candidates:
            empty_result = RankingResult(items=[])
            save_ranking_json(
                transcript_id=transcript_id,
                ranking_result=empty_result,
                output_path=ranking_output_path,
                ranking_model=validation_model,
                candidate_count=0,
            )
            return FinalResult(transcript_id=transcript_id, signals=[])

        ranking_result = merge_and_rank(
            transcript_id=transcript_id,
            turns=turns,
            ranking_candidates=ranking_candidates,
            model=validation_model,
        )
        save_ranking_json(
            transcript_id=transcript_id,
            ranking_result=ranking_result,
            output_path=ranking_output_path,
            ranking_model=validation_model,
            candidate_count=len(ranking_candidates),
        )
        final_result = enforce_output_rules(
            _ranking_result_to_final_result(transcript_id, ranking_result)
        )
        if final_output_path is not None:
            save_final_result_json(final_result, final_output_path)
        return final_result

    # Step 1: run parser, and only use repair when needed.
    turns = parse_turns_with_conditional_repair(raw_text, extraction_model)
    print(f"[pipeline] Conditional parse result turn count: {len(turns)}")
    logger.info("Conditional parse turn count: %s", len(turns))

    # Step 2: split into context-preserving chunks for extraction.
    effective_chunk_size = min(TARGET_CHUNK_TURNS, MAX_CHUNK_TURNS)
    chunks, boundary_reasons = chunk_turns(
        turns,
        chunk_size=effective_chunk_size,
        overlap=CHUNK_OVERLAP,
        max_chunk_tokens=MAX_CHUNK_TOKENS,
    )
    print(f"[pipeline] Created {len(chunks)} chunk(s) for extraction.")
    logger.info("Chunk count: %s", len(chunks))

    if chunk_metadata_output_path is not None:
        # Reuse the clean parser output so chunk JSON carries richer per-turn traceability fields.
        clean_turns = parse_transcript_clean_json(raw_text)
        clean_turns_by_id = {
            str(turn["turn_id"]): turn for turn in clean_turns if "turn_id" in turn
        }
        meeting_id = _extract_meeting_id(clean_turns)
        save_chunk_metadata_json(
            transcript_id=transcript_id,
            chunks=chunks,
            boundary_reasons=boundary_reasons,
            output_path=chunk_metadata_output_path,
            chunk_size=effective_chunk_size,
            overlap=CHUNK_OVERLAP,
            max_chunk_tokens=MAX_CHUNK_TOKENS,
            meeting_id=meeting_id,
            clean_turns_by_id=clean_turns_by_id,
        )

    # Step 3: extract candidate signals from each chunk.
    all_candidates: list[CandidateSignal] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"[pipeline] Extracting candidates from chunk {idx}/{len(chunks)}.")
        extraction = extract_candidates(turns=chunk, model=extraction_model)
        all_candidates.extend(extraction.candidates)
    print(f"[pipeline] Total extracted candidates: {len(all_candidates)}")
    logger.info("Extracted candidate count: %s", len(all_candidates))

    if extract_output_path is not None:
        save_extract_json(
            transcript_id=transcript_id,
            candidates=all_candidates,
            output_path=extract_output_path,
            extraction_model=extraction_model,
            chunk_count=len(chunks),
        )

    # Step 4: keep only deterministic, quote-grounded candidates.
    verified_candidates = [
        candidate for candidate in all_candidates if verify_candidate(candidate, turns)
    ]
    print(f"[pipeline] Verified candidates after grounding checks: {len(verified_candidates)}")
    logger.info("Verified candidate count: %s", len(verified_candidates))

    # Allow empty output instead of forcing weak or unverifiable signals.
    if not verified_candidates:
        print("[pipeline] No verified candidates. Returning empty result.")
        return FinalResult(transcript_id=transcript_id, signals=[])

    deterministic_keep, ambiguous_candidates = split_deterministic_and_ambiguous(verified_candidates)
    print(
        f"[pipeline] Deterministic keep: {len(deterministic_keep)}; "
        f"ambiguous for evidence judge: {len(ambiguous_candidates)}"
    )

    post_validation_keep = list(deterministic_keep)
    evidence_validation = EvidenceValidationResult(validations=[])

    # Run the LLM evidence judge only when we have ambiguous items and an output target.
    should_run_evidence_judge = bool(ambiguous_candidates and evidence_validation_output_path is not None)

    if should_run_evidence_judge:
        print("[pipeline] Running evidence validation on ambiguous candidates.")
        evidence_validation = validate_evidence(
            transcript_id=transcript_id,
            turns=turns,
            candidates=ambiguous_candidates,
            model=validation_model,
        )
        post_validation_keep.extend(
            apply_evidence_actions(ambiguous_candidates, evidence_validation)
        )
    else:
        # If judge is disabled, pass ambiguous items through unchanged.
        post_validation_keep.extend(ambiguous_candidates)

    if evidence_validation_output_path is not None:
        save_evidence_validation_json(
            transcript_id=transcript_id,
            validation_result=evidence_validation,
            output_path=evidence_validation_output_path,
            validation_model=validation_model,
            candidate_count=len(deterministic_keep) + len(ambiguous_candidates),
            deterministic_candidates=deterministic_keep,
            ambiguous_candidates=ambiguous_candidates,
        )

    if not post_validation_keep:
        print("[pipeline] No candidates left after evidence validation. Returning empty result.")
        return FinalResult(transcript_id=transcript_id, signals=[])

    # Step 5: deduplicate and rank supported candidates.
    print("[pipeline] Merging duplicates and ranking candidates.")
    if ranking_output_path is not None:
        ranking_candidates = _build_ranking_candidates_payload(
            deterministic_candidates=deterministic_keep,
            ambiguous_candidates=ambiguous_candidates,
            evidence_validation=evidence_validation,
            should_run_evidence_judge=should_run_evidence_judge,
        )
        ranking_result = merge_and_rank(
            transcript_id=transcript_id,
            turns=turns,
            ranking_candidates=ranking_candidates,
            model=validation_model,
        )
        save_ranking_json(
            transcript_id=transcript_id,
            ranking_result=ranking_result,
            output_path=ranking_output_path,
            ranking_model=validation_model,
            candidate_count=len(ranking_candidates),
        )
        result = _ranking_result_to_final_result(transcript_id, ranking_result)
    else:
        result = validate_and_rank(
            transcript_id=transcript_id,
            turns=turns,
            candidates=post_validation_keep,
            model=validation_model,
        )

    # Step 6: enforce hard caps and clean final ranks.
    print("[pipeline] Enforcing output limits and final ranks.")
    final_result = enforce_output_rules(result)

    if final_output_path is not None:
        save_final_result_json(final_result, final_output_path)

    return final_result
