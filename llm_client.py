import logging
from typing import Any, Dict, List

import httpx


class LLMClient:
    """Manages communication with the LLM provider."""

    PROVIDERS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "vocareum": "https://claude.vocareum.com/v1/chat/completions",
    }

    def __init__(self, api_key: str, provider: str = "openai") -> None:
        self.api_key: str = api_key
        self.provider: str = provider.lower()
        if self.provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.PROVIDERS.keys())}")

    def get_response(self, messages: List[Dict[str, Any]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.HTTPError: If the request to the LLM fails.
        """
        message = self.get_completion(messages)
        return message.get("content") or ""

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Return one assistant message, including any structured tool calls."""
        if self.provider == "openai":
            return self._call_openai(messages, tools)
        if self.provider == "groq":
            return self._call_groq(messages, tools)
        if self.provider == "vocareum":
            return self._call_vocareum(messages, tools)
        raise ValueError("Unknown provider")

    def _call_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        url = self.PROVIDERS["openai"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": "gpt-4o",
            "max_tokens": 8024,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._execute_request(url, headers, payload)

    def _call_groq(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        url = self.PROVIDERS["groq"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "messages": messages,
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._execute_request(url, headers, payload)

    def _call_vocareum(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        url = self.PROVIDERS["vocareum"]
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload: Dict[str, Any] = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "messages": messages
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._execute_request(url, headers, payload)

    def _execute_request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            logging.error("LLM request failed: %s", exc)
            if isinstance(exc, httpx.HTTPStatusError):
                logging.error("Status code: %s", exc.response.status_code)
                logging.error("Response details: %s", exc.response.text)
            raise

        message = data["choices"][0]["message"]
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        return assistant_message
