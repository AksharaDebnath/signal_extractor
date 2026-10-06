"""Transcript parser utilities for normalized turns and cleaned JSON exports."""

import re
import unicodedata
from src.models import CleanTranscriptTurn, TranscriptTurn

# Match one transcript turn.
# It works for both:
# 1) newline-separated turns
# 2) inline turns on a single long line
TURN_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})"
    r"\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
    r"\s*\[(?P<speaker>[^\]]+)\]:\s*"
    r"(?P<text>.*?)(?=(?:\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*-->)|$)",
    re.DOTALL,
)

# Optional transcript-level metadata in the header.
MEETING_ID_PATTERN = re.compile(r"(?im)^\s*Meeting\s+ID:\s*(?P<value>.+?)\s*$")
MEETING_TOPIC_PATTERN = re.compile(r"(?im)^\s*Meeting\s+Topic:\s*(?P<value>.+?)\s*$")


def _extract_group_value(match: re.Match[str] | None, group_name: str = "value") -> str | None:
    # Return stripped regex group text when available.
    if match is None:
        return None
    return match.group(group_name).strip()


def map_speaker(raw_speaker: str) -> str:
    # Normalize speaker labels into the small fixed set used by the project.
    label = raw_speaker.strip().upper()
    # Normalize common separators so ADVISOR-1 and ADVISOR 1 are treated similarly.
    normalized = re.sub(r"[\s\-]+", "_", label)

    if normalized.startswith("ADVISOR"):
        return "ADVISOR"

    if (
        normalized in {"OPTIMIZE_REP", "REP", "HOST"}
        or normalized.startswith("OPTIMIZE_REP")
        or normalized.startswith("REP_")
        or normalized.startswith("HOST_")
    ):
        return "OPTIMIZE_REP"

    return "UNKNOWN"


def extract_meeting_metadata(raw_text: str) -> tuple[str | None, str | None]:
    # Metadata is optional, so missing fields are returned as None.
    meeting_id_match = MEETING_ID_PATTERN.search(raw_text)
    meeting_topic_match = MEETING_TOPIC_PATTERN.search(raw_text)

    meeting_id = _extract_group_value(meeting_id_match)
    meeting_topic = _extract_group_value(meeting_topic_match)
    return meeting_id, meeting_topic


def normalize_turn_text(text: str) -> str:
    # Normalize common unicode punctuation and transliterate to plain ASCII.
    punctuation_map = str.maketrans(
        {
            "…": "...",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
        }
    )
    normalized_text = unicodedata.normalize("NFKD", text).translate(punctuation_map)
    ascii_text = normalized_text.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split())


def parse_transcript(raw_text: str) -> list[TranscriptTurn]:
    # Parse raw transcript text into normalized turns used by pipeline logic.
    turns: list[TranscriptTurn] = []

    for index, match in enumerate(TURN_PATTERN.finditer(raw_text), start=1):
        start_timestamp = match.group("start")
        speaker_raw = match.group("speaker")
        turn_text = normalize_turn_text(match.group("text"))

        turns.append(
            TranscriptTurn(
                turn_id=f"turn-{index:03d}",
                timestamp=start_timestamp,
                speaker=map_speaker(speaker_raw),
                text=turn_text,
            )
        )

    if not turns:
        raise ValueError("No transcript turns were parsed.")

    return turns


def parse_transcript_clean(raw_text: str) -> list[CleanTranscriptTurn]:
    # Parse transcript into rich cleaned-turn objects used by result JSON files.
    cleaned_turns: list[CleanTranscriptTurn] = []
    meeting_id, meeting_topic = extract_meeting_metadata(raw_text)

    for index, match in enumerate(TURN_PATTERN.finditer(raw_text), start=1):
        start_timestamp = match.group("start")
        end_timestamp = match.group("end")
        speaker_raw = match.group("speaker").strip()
        turn_text = normalize_turn_text(match.group("text"))

        cleaned_turns.append(
            CleanTranscriptTurn(
                turn_id=f"turn-{index:03d}",
                meeting_id=meeting_id,
                meeting_topic=meeting_topic,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                speaker_raw=speaker_raw,
                speaker_role=map_speaker(speaker_raw),
                text=turn_text,
            )
        )

    if not cleaned_turns:
        raise ValueError("No transcript turns were parsed.")

    return cleaned_turns


def parse_transcript_clean_json(raw_text: str) -> list[dict[str, str | None]]:
    # Convert cleaned turns into plain dicts so callers can write JSON directly.
    return [turn.model_dump() for turn in parse_transcript_clean(raw_text)]