import json
import os
from typing import Any
from dotenv import load_dotenv

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.provider = os.getenv("LLM_PROVIDER")
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e.msg}", e.doc, e.pos)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables. Please check your .env file.")
        return self.api_key

    @property
    def llm_provider(self) -> str:
        """Get the LLM provider.

        Returns:
            The provider as a string.

        Raises:
            ValueError: If the provider is not found in environment variables.
        """
        if not self.provider:
            raise ValueError("LLM_PROVIDER not found in environment variables. Please check your .env file.")
        return self.provider
