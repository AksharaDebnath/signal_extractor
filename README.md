# Signal Extractor

An evidence-first pipeline that turns timestamped advisor conversations into a small, ranked set of decision drivers and blockers. It uses deterministic parsing and verification wherever possible, reserving LLM calls for semantic judgment.

## What it does

1. Parses timestamped speaker turns.
2. Repairs structure with an LLM only when the parser cannot reliably identify speakers.
3. Chunks long transcripts with overlap and a token budget.
4. Extracts candidate advisor-side drivers and blockers.
5. Rejects candidates whose quote, timestamp, or speaker cannot be verified deterministically.
6. Validates ambiguous evidence, deduplicates supported candidates, and ranks the final result.

The output contains at most three drivers and three blockers. An empty result is valid when no material, well-supported signal exists.

## Privacy and security

This repository intentionally contains **no transcripts, generated results, or API credentials**. Put private `.txt` transcripts in `data/transcripts/` when running locally; Git ignores them and all generated JSON artifacts. Never commit `.env`.

If a credential is ever committed or exposed, revoke it at the provider and replace it rather than relying only on deletion.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
cp .env.example .env       # Windows PowerShell: Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then place one or more timestamped `.txt` transcripts in `data/transcripts/`.

## Run

```bash
python src/main.py
```

Paths and model choices live in `src/config.py`. Results are written locally to `data/processing/` and `data/results/`.

## Input format

```text
Meeting ID: demo-001
Meeting Topic: Initial conversation

00:00:01.000 --> 00:00:04.000 [ADVISOR]: My current workflow is too manual.
00:00:05.000 --> 00:00:08.000 [OPTIMIZE_REP]: We can discuss our operations support.
```

Only statements made by `ADVISOR` can become evidence. Representative remarks provide context but never establish a signal by themselves.

## Synthetic example

[`examples/synthetic_transcript.txt`](examples/synthetic_transcript.txt) is a fully fictional input, and [`examples/expected_result.json`](examples/expected_result.json) illustrates the expected evidence-first output. These files are safe to publish; real inputs and generated local results remain ignored.

## Design principles

- **Evidence before inference:** quotes are checked against the source turn.
- **Conservative outputs:** weak, polite, or representative-led signals are excluded.
- **Traceability:** outputs retain timestamps and source-turn lineage.
- **Cost awareness:** LLM repair is conditional and transcripts are chunked.

## Project layout

```text
src/       Pipeline, schemas, parser, and LLM wrapper
prompts/   Structured prompts for repair, extraction, validation, and ranking
data/      Local-only transcript inputs and pipeline outputs
```
