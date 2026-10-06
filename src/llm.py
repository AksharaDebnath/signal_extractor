"""Small wrapper around OpenAI structured responses used by the pipeline."""

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
_client: OpenAI | None = None


def get_client() -> OpenAI:
    # Create the API client on first use so imports stay cheap in tests.
    global _client
    if _client is None:
        # Lazy init keeps imports/test collection working without API credentials.
        _client = OpenAI()
    return _client


def structured_call(
    *,
    model: str,
    prompt: str,
    content: str,
    output_model: type[T],
) -> T:
    # Ask the model for structured output that matches the requested Pydantic schema.
    response = get_client().responses.parse(
        model=model,
        instructions=prompt,
        input=content,
        text_format=output_model,
    )

    if response.output_parsed is None:
        raise ValueError("The model returned no structured output.")

    return response.output_parsed
