import logging
import httpx
from typing import Any, Dict, List

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

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        try:
            if self.provider == "openai":
                return self._call_openai(messages)
            elif self.provider == "groq":
                return self._call_groq(messages)
            elif self.provider == "vocareum":
                return self._call_vocareum(messages)
            else:
                raise ValueError("Unknown provider")
        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return f"I encountered an error: {error_message}. Please try again or rephrase your request."

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        url = self.PROVIDERS["openai"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": "gpt-4o",
            "max_tokens": 8024, # needed to be encreased of the json extraction prompt
            "messages": messages
        }
        return self._execute_request(url, headers, payload)

    def _call_groq(self, messages: List[Dict[str, str]]) -> str:
        url = self.PROVIDERS["groq"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }
        return self._execute_request(url, headers, payload)

    def _call_vocareum(self, messages: List[Dict[str, str]]) -> str:
        url = self.PROVIDERS["vocareum"]
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "messages": messages
        }
        return self._execute_request(url, headers, payload)

    def _execute_request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> str:
        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # Assuming OpenAI-compatible response structure for simplicity and consistency with original code
            # which targeted OpenAI endpoints.
            return data["choices"][0]["message"]["content"]
