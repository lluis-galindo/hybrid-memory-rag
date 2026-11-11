import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .utils import SqliteCache, count_tokens, make_key

Message = dict[str, str]
Schema = dict[str, Any]
T = TypeVar("T", bound=BaseModel)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
REPAIR_INSTRUCTION = "That reply was not valid JSON for the required schema. Reply again with only the JSON object."


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class LLMCall:
    role: str
    model: str
    completion: Completion
    cached: bool
    messages: list[Message]


@dataclass
class Usage:
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return sum(call.completion.prompt_tokens for call in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(call.completion.completion_tokens for call in self.calls)

    @property
    def latency_s(self) -> float:
        return sum(call.completion.latency_s for call in self.calls)

    def transcript(self) -> list[dict[str, Any]]:
        return [asdict(call) for call in self.calls]


_current_usage: ContextVar[Usage | None] = ContextVar("usage", default=None)


@contextmanager
def track_usage() -> Iterator[Usage]:
    usage = Usage()
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        _current_usage.reset(token)


class StructuredOutputError(RuntimeError):
    pass


def extract_json(text: str) -> str:
    cleaned = THINK_BLOCK.sub("", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return cleaned[start : end + 1]


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        options: dict[str, Any] | None = None,
        think: bool | None = False,
        timeout_s: float = 900.0,
        cache: SqliteCache | None = None,
    ):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.options = options or {}
        self.think = think
        self.cache = cache
        self._client = httpx.Client(timeout=timeout_s)

    def _generate(self, messages: list[Message], schema: Schema | None, options: dict[str, Any]) -> Completion:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if schema is not None:
            payload["format"] = schema
        if self.think is not None:
            payload["think"] = self.think
        started = time.perf_counter()
        response = self._client.post(self.url, json=payload)
        response.raise_for_status()
        body = response.json()
        latency = time.perf_counter() - started
        text = body["message"]["content"]
        prompt_tokens = body.get("prompt_eval_count") or sum(count_tokens(m["content"]) for m in messages)
        return Completion(text, prompt_tokens, body.get("eval_count") or count_tokens(text), latency)

    def complete(
        self,
        messages: list[Message],
        schema: Schema | None = None,
        role: str = "generic",
        num_ctx: int | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        options = dict(self.options)
        if num_ctx:
            options["num_ctx"] = num_ctx
        if max_tokens:
            options["num_predict"] = max_tokens
        key = make_key([self.model, self.think], messages, schema, options)
        stored = self.cache.get(key) if self.cache else None
        completion = Completion(**stored) if stored else self._generate(messages, schema, options)
        if self.cache and not stored:
            self.cache.put(key, asdict(completion))
        usage = _current_usage.get()
        if usage is not None:
            usage.calls.append(LLMCall(role, self.model, completion, bool(stored), messages))
        return completion

    def structured(
        self,
        messages: list[Message],
        output_type: type[T],
        role: str = "generic",
        num_ctx: int | None = None,
        attempts: int = 2,
        max_tokens: int | None = None,
    ) -> T:
        schema = output_type.model_json_schema()
        conversation = list(messages)
        last_error: Exception | None = None
        for _ in range(attempts):
            completion = self.complete(conversation, schema, role, num_ctx, max_tokens)
            try:
                return output_type.model_validate_json(extract_json(completion.text))
            except (ValidationError, ValueError) as error:
                last_error = error
                conversation = conversation + [
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": REPAIR_INSTRUCTION},
                ]
        raise StructuredOutputError(f"{output_type.__name__}: {last_error}")
