"""Bounded, stateless MiniMax multimodal requests; never persist credentials."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def config() -> dict:
    """Read explicit process configuration; never inspect personal shell files."""
    allowed = {"MINIMAX_API_KEY", "MINIMAX_BASE_URL", "MINIMAX_MODEL"}
    values = {k: os.environ[k] for k in allowed if os.environ.get(k)}
    if not all(values.get(k) for k in allowed):
        raise ValueError("Missing literal MiniMax configuration")
    if not values["MINIMAX_BASE_URL"].startswith("https://"):
        raise ValueError("MiniMax endpoint must use HTTPS")
    if values["MINIMAX_MODEL"] != "MiniMax-M3":
        raise ValueError("This pilot freezes the provider to MiniMax-M3; no fallback")
    return values


def ask(system: str, context: dict, images: list[Path], timeout: float = 90, *, max_tokens: int = 4096) -> dict:
    """Return a JSON decision and measured usage, or an explicit failed call."""
    cfg = config()
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False, allow_nan=False)}]
    for path in images:
        data = path.read_bytes()
        if len(data) > 2_000_000:
            raise ValueError("Pilot image exceeds 2 MB")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{mime};base64," + base64.b64encode(data).decode(), "detail": "low"}})
    payload = {"model": cfg["MINIMAX_MODEL"], "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.2, "reasoning_split": True}
    body = json.dumps(payload, ensure_ascii=False).encode()
    started = time.monotonic()
    record = {"model": cfg["MINIMAX_MODEL"], "request_sha256": hashlib.sha256(body).hexdigest(),
              "image_count": len(images), "usage": None, "status": "error", "decision": None}
    url = cfg["MINIMAX_BASE_URL"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    request = urllib.request.Request(url, data=body, headers={
        "Authorization": "Bearer " + cfg["MINIMAX_API_KEY"], "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        record["usage"] = result.get("usage")
        record["response_model"] = result.get("model")
        if result.get("model") and result["model"] != cfg["MINIMAX_MODEL"]:
            raise ValueError("Provider returned an unexpected model")
        choice = result["choices"][0]
        record["finish_reason"] = choice.get("finish_reason")
        if choice.get("finish_reason") != "stop":
            raise ValueError("Incomplete provider response")
        answer = choice["message"]["content"]
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer)
        record["decision"] = json.loads(answer)
        record["status"] = "ok"
    except urllib.error.HTTPError as exc:
        record["error"] = f"HTTP {exc.code}"  # Do not persist server bodies or headers.
    except Exception as exc:
        record["error"] = type(exc).__name__
    record["wallclock_s"] = time.monotonic() - started
    return record
