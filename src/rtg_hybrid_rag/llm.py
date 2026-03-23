from __future__ import annotations

import json
from collections.abc import Iterable

from openai import OpenAI

from rtg_hybrid_rag.settings import Settings


def _extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Could not find JSON object in model output: {text}")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError(f"Could not parse a complete JSON object from model output: {text}")


class OpenAICompatibleClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _client_for_provider(self, provider: str) -> OpenAI:
        if provider == "openrouter":
            if not self.settings.openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY is missing")
            return OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
            )
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is missing")
            return OpenAI(api_key=self.settings.openai_api_key)
        raise ValueError(f"Unsupported provider: {provider}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._client_for_provider(self.settings.embedding_provider)
        response = client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def json_completion(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict:
        client = self._client_for_provider(provider)
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content or ""
        return _extract_json(content)

    def text_completion(
        self,
        *,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        client = self._client_for_provider(provider)
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content or ""


def batched(items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
