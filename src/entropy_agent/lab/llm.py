"""DeepSeek chat client used by the research lab.

Reads ``DEEPSEEK_API_KEY`` from the repository ``.env`` (falling back to the
process environment) and always asks for a JSON object, which the planner then
validates with Pydantic.  The client is deliberately tiny: one function, one
retry, explicit errors.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENV_PATH = ROOT / ".env"
API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


def load_api_key(env_path: pathlib.Path | None = None) -> str:
    path = env_path or ENV_PATH
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip('"').strip("'")
                if key:
                    return key
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing (put it in .env)")
    return key


def chat(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    json_mode: bool = True,
    timeout: float = 90.0,
    retries: int = 1,
) -> str:
    key = load_api_key()
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5)
    raise RuntimeError(f"DeepSeek call failed: {last_error}")


class FakeLLM:
    """Deterministic stand-in used by the tests (no network, no key)."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict], **_kwargs) -> str:
        self.calls.append(messages)
        if not self.replies:
            return json.dumps({"action": "finish", "reason": "no more replies"})
        return self.replies.pop(0)
