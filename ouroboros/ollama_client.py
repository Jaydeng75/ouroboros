"""Minimal Ollama HTTP client (stdlib only)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        data = self._get_json("/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]

    def generate(
        self,
        prompt: str,
        model: str,
        *,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = self._post_json("/api/generate", payload, timeout)
        if "response" not in data:
            raise OllamaError(f"Unexpected Ollama response: {sorted(data)}")
        return str(data["response"])

    def _get_json(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is it running? Start with: ollama serve"
            ) from exc

    def _post_json(self, path: str, payload: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaError(
                f"Ollama generate failed at {self.base_url}{path}: {exc}"
            ) from exc
