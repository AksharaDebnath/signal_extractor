"""CLI entrypoint: loops transcript files and runs the processing pipeline."""

import sys
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # Allow running this file directly via "python src/main.py ...".
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import process_transcript
from src.config import (
    CLEANED_OUTPUT_PATH,
    CHUNK_METADATA_OUTPUT_PATH,
    EVIDENCE_VALIDATION_OUTPUT_PATH,
    FINAL_OUTPUT_PATH,
    OUTPUT_PATH,
    RANKING_OUTPUT_PATH,
    TRANSCRIPT_PATH,
)


def _load_environment() -> None:
    # Load .env from common locations, then allow dotenv default lookup too.
    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"):
        if env_path.exists():
            load_dotenv(env_path)
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to advisor-signal-extractor/.env "
            "or the parent project's .env file."
        )


def _resolve_from_project(path_value: str) -> Path:
    # Allow relative paths in config while still supporting absolute overrides.
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _run_for_file(
    transcript_file: Path,
    cleaned_output_path: Path,
    extract_output_path: Path,
    chunk_metadata_output_path: Path,
    evidence_validation_output_path: Path,
    ranking_output_path: Path,
    final_output_path: Path,
) -> None:
    # Full mode reads transcript text and runs Steps 1-6.
    raw_text = transcript_file.read_text(encoding="utf-8")
    cleaned_file = cleaned_output_path / f"{transcript_file.stem}.json"
    extract_file = extract_output_path / f"{transcript_file.stem}.json"
    chunk_file = chunk_metadata_output_path / f"{transcript_file.stem}.json"
    evidence_file = evidence_validation_output_path / f"{transcript_file.stem}.json"
    ranking_file = ranking_output_path / f"{transcript_file.stem}.json"
    final_file = final_output_path / f"{transcript_file.stem}.json"

    # Run all stages from parser to final deterministic output.
    process_transcript(
        transcript_id=transcript_file.stem,
        raw_text=raw_text,
        cleaned_output_path=cleaned_file,
        extract_output_path=extract_file,
        evidence_validation_output_path=evidence_file,
        ranking_output_path=ranking_file,
        final_output_path=final_file,
        chunk_metadata_output_path=chunk_file,
    )


def main() -> None:
    _load_environment()

    transcript_path = _resolve_from_project(TRANSCRIPT_PATH)
    cleaned_output_path = _resolve_from_project(CLEANED_OUTPUT_PATH)
    extract_output_path = _resolve_from_project(OUTPUT_PATH)
    chunk_metadata_output_path = _resolve_from_project(CHUNK_METADATA_OUTPUT_PATH)
    evidence_validation_output_path = _resolve_from_project(EVIDENCE_VALIDATION_OUTPUT_PATH)
    ranking_output_path = _resolve_from_project(RANKING_OUTPUT_PATH)
    final_output_path = _resolve_from_project(FINAL_OUTPUT_PATH)

    # Create all Step 1-6 output folders.
    cleaned_output_path.mkdir(parents=True, exist_ok=True)
    extract_output_path.mkdir(parents=True, exist_ok=True)
    chunk_metadata_output_path.mkdir(parents=True, exist_ok=True)
    evidence_validation_output_path.mkdir(parents=True, exist_ok=True)
    ranking_output_path.mkdir(parents=True, exist_ok=True)
    final_output_path.mkdir(parents=True, exist_ok=True)

    if not transcript_path.exists():
        if transcript_path.suffix:
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        transcript_path.mkdir(parents=True, exist_ok=True)
        print(f"[main] Transcript directory did not exist. Created: {transcript_path}")
        print("[main] Add one or more .txt transcript files, then run again.")
        return

    # Directory mode: process every .txt file.
    if transcript_path.is_dir():
        txt_files = sorted(transcript_path.glob("*.txt"))
        if not txt_files:
            print(f"[main] No .txt files found in directory: {transcript_path}")
            return

        print(f"[main] Found {len(txt_files)} transcript file(s).")
        for txt_file in txt_files:
            _run_for_file(
                txt_file,
                cleaned_output_path,
                extract_output_path,
                chunk_metadata_output_path,
                evidence_validation_output_path,
                ranking_output_path,
                final_output_path,
            )
        return

    # Single-file mode.
    _run_for_file(
        transcript_path,
        cleaned_output_path,
        extract_output_path,
        chunk_metadata_output_path,
        evidence_validation_output_path,
        ranking_output_path,
        final_output_path,
    )


if __name__ == "__main__":
    main()
