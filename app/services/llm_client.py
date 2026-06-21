import json
import re
from typing import Any

from ollama import Client, ResponseError


class LLMUnavailableError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.host = host
        self.client = Client(host=host)

    def health_check(self) -> dict[str, Any]:
        try:
            return self.client.list()
        except Exception as exc:  # pragma: no cover - process/network boundary
            raise LLMUnavailableError(
                f"Ollama is not reachable at {self.host}. Start Ollama and make sure "
                f"the model '{self.model}' is available."
            ) from exc

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
            )
        except ResponseError as exc:  # pragma: no cover - process/network boundary
            raise LLMUnavailableError(
                f"Ollama request failed for model '{self.model}': {exc.error}"
            ) from exc
        except Exception as exc:  # pragma: no cover - process/network boundary
            raise LLMUnavailableError(
                f"Ollama is not running or not reachable at {self.host}."
            ) from exc

        content = response.get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

    def extract_json(self, messages: list[dict[str, str]]) -> Any:
        text = self.chat(messages, temperature=0.1)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if fenced:
                return json.loads(fenced.group(1))

            inline = re.search(r"(\{.*\})", text, re.DOTALL)
            if inline:
                return json.loads(inline.group(1))

        raise ValueError(f"Model did not return valid JSON. Raw output: {text}")
