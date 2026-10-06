"""Centralized runtime and pipeline configuration."""

TRANSCRIPT_PATH = "data/transcripts/"
CLEANED_OUTPUT_PATH = "data/processing/preprocessed"
OUTPUT_PATH = "data/processing/extracts"
CHUNK_METADATA_OUTPUT_PATH = "data/processing/chunks"
EVIDENCE_VALIDATION_OUTPUT_PATH = "data/processing/validation"
RANKING_OUTPUT_PATH = "data/processing/ranking"
FINAL_OUTPUT_PATH = "data/results"

# LLM runtime settings used by src/pipeline.py.
EXTRACTION_MODEL = "gpt-5-mini"
VALIDATION_MODEL = "gpt-5-mini"

# Conditional repair settings used by src/pipeline.py.
UNKNOWN_RATIO_THRESHOLD = 0.30
UNKNOWN_RATIO_MIN_TURNS = 5

# Chunking settings used by src/pipeline.py.
TARGET_CHUNK_TURNS = 40
MAX_CHUNK_TURNS = 60
CHUNK_OVERLAP = 5

# Token guardrail used by chunk construction.
MAX_CHUNK_TOKENS = 3000

# Chunk metadata/versioning fields used by src/pipeline.py.
CHUNKING_VERSION = "chunk-v1.0.0"
SOURCE_PARSER_VERSION = "parse-v1.0.0"
CHUNKING_STRATEGY = "turn_based_with_token_guardrail"
TOKENIZER_NAME = "approx-char-4"
