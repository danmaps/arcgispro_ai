import json
import ssl
import time
from typing import Any, Dict, List, Optional
from urllib import error, request

from ..core.model_registry import DEFAULT_GITHUB_MODELS


class GitHubModelsClient:
    """Client for GitHub Models chat completions."""

    def __init__(
        self,
        token: str,
        model: str = "openai/gpt-4.1",
        endpoint: str = "https://models.github.ai/inference/chat/completions",
    ):
        self.token = token
        self.model = model
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_available_models(self) -> List[str]:
        """Get available models from GitHub Models catalog, ordered by curation."""
        fallback_models = DEFAULT_GITHUB_MODELS
        req = request.Request(
            "https://models.github.ai/catalog/models",
            headers={
                "Authorization": self.headers["Authorization"],
                "Accept": "application/json",
                "X-GitHub-Api-Version": self.headers["X-GitHub-Api-Version"],
            },
            method="GET",
        )
        ssl_context = ssl.create_default_context()

        try:
            with request.urlopen(req, timeout=15, context=ssl_context) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body)
        except Exception:
            return fallback_models

        if not isinstance(payload, list):
            return fallback_models

        available_ids = {
            model.get("id")
            for model in payload
            if isinstance(model, dict) and model.get("id")
        }
        if not available_ids:
            return fallback_models

        curated_available = [
            model_id for model_id in fallback_models if model_id in available_ids
        ]
        if curated_available:
            return curated_available

        # If catalog shape changes or curated IDs drift, still offer a small, stable list.
        discovered = sorted(str(model_id) for model_id in available_ids if isinstance(model_id, str))
        return discovered[:20] if discovered else fallback_models

    @staticmethod
    def _extract_text_content(response: Dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise Exception("GitHub Models returned no choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value:
                    chunks.append(text_value)
            if chunks:
                return "".join(chunks).strip()

        raise Exception("GitHub Models response did not include text content.")

    def _post_json(
        self,
        payload: Dict[str, Any],
        max_retries: int = 3,
        timeout_seconds: int = 60,
    ) -> Dict[str, Any]:
        encoded_payload = json.dumps(payload).encode("utf-8")
        ssl_context = ssl.create_default_context()

        for attempt in range(max_retries):
            req = request.Request(
                self.endpoint,
                data=encoded_payload,
                headers=self.headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=timeout_seconds, context=ssl_context) as response:
                    body = response.read().decode("utf-8")
                    return json.loads(body)
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt == max_retries - 1:
                    raise Exception(
                        f"GitHub Models request failed with status {exc.code}: {detail}"
                    ) from exc
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"GitHub Models request failed after {max_retries} attempts: {exc}"
                    ) from exc

            time.sleep(2 ** attempt)

        raise Exception("GitHub Models request failed with an unknown error.")

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        token_limit = max_tokens if max_tokens is not None else 4096
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": token_limit,
        }

        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        response = self._post_json(payload)
        return self._extract_text_content(response)

    def get_vision_completion(self, messages: List[Dict[str, Any]], max_tokens: int = 800) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        response = self._post_json(payload)
        return self._extract_text_content(response)
