"""LLM provider implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _openai_format_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tool schemas to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


@dataclass
class OpenAIBase:
    """Shared OpenAI-compatible provider behavior."""

    model: str
    api_key: str | None = None
    base_url: str | None = None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate completion using OpenAI-compatible API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 2048),
            **kwargs,
        }

        if tools:
            request_params["tools"] = _openai_format_tools(tools)

        response = await client.chat.completions.create(**request_params)
        message = response.choices[0].message

        result = {"role": "assistant", "content": message.content or ""}

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return result


@dataclass
class VLLM(OpenAIBase):
    """vLLM local model provider."""

    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    base_url: str | None = "http://localhost:8000/v1"
    api_key: str | None = "EMPTY"


@dataclass
class OpenAI(OpenAIBase):
    """OpenAI model provider."""

    model: str = "gpt-4"

