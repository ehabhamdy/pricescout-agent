"""Application-level pricing tools backed by MCP clients.

These narrow tools are exposed to the model and translate pricing operations
into calls to private SQLite and Firecrawl MCP tools.
"""
import ast
import json
import math
import re
from typing import Any

from fastmcp import Client as FastMCPClient


class PriceTools:
    """Narrow pricing operations backed by private MCP tools."""

    def __init__(
        self,
        sqlite_client: FastMCPClient,
        firecrawl_client: FastMCPClient,
        providers: dict[str, str],
    ) -> None:
        self.sqlite_client = sqlite_client
        self.firecrawl_client = firecrawl_client
        self.providers = {
            self._normalize(name): (name, url)
            for name, url in providers.items()
        }
        self._last_scraped_providers: set[str] = set()

    async def initialize(self) -> None:
        await self.sqlite_client.call_tool(
            "write_query",
            {
                "query": """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,
                    features TEXT,
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            },
        )
        await self.sqlite_client.call_tool(
            "write_query",
            {
                "query": (
                    "CREATE INDEX IF NOT EXISTS idx_pricing_company "
                    "ON pricing_plans(company_name COLLATE NOCASE)"
                )
            },
        )

    def reset_turn(self) -> None:
        self._last_scraped_providers.clear()

    def model_tools(self) -> list[dict[str, Any]]:
        provider_names = sorted(name for name, _ in self.providers.values())
        tools = [
            self._model_tool(
                name="find_prices",
                description=(
                    "Find cached pricing records by provider and model. Returns JSON "
                    "with ok, source='database', records, and missing_providers. "
                    "An empty records list is a successful cache miss."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "providers": {
                            "type": "array",
                            "items": {"type": "string", "enum": provider_names},
                            "minItems": 1,
                            "uniqueItems": True,
                            "description": "Providers whose cached prices are required.",
                        },
                        "model": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Model or pricing-plan name to match.",
                        },
                    },
                    "required": ["providers", "model"],
                    "additionalProperties": False,
                },
            ),
            self._model_tool(
                name="scrape_provider_pricing",
                description=(
                    "Scrape official allowlisted pricing pages for named providers. "
                    "The application selects URLs. Returns JSON with ok, "
                    "source='fresh_scrape', providers, and content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "providers": {
                            "type": "array",
                            "items": {"type": "string", "enum": provider_names},
                            "minItems": 1,
                            "uniqueItems": True,
                            "description": "Providers whose pricing pages to scrape.",
                        }
                    },
                    "required": ["providers"],
                    "additionalProperties": False,
                },
            ),
        ]

        if self._last_scraped_providers:
            tools.append(
                self._model_tool(
                    name="save_prices",
                    description=(
                        "Validate and save records extracted from the immediately "
                        "preceding scrape. Only scraped providers are accepted. Returns "
                        "JSON with ok, saved_count, and rejected records on failure."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "records": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 100,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "provider": {"type": "string"},
                                        "model": {"type": "string"},
                                        "input_price": {
                                            "type": ["number", "null"],
                                            "minimum": 0,
                                        },
                                        "output_price": {
                                            "type": ["number", "null"],
                                            "minimum": 0,
                                        },
                                        "currency": {
                                            "type": "string",
                                            "default": "USD",
                                        },
                                        "unit": {
                                            "type": "string",
                                            "description": (
                                                "Billing unit, for example "
                                                "per million tokens."
                                            ),
                                        },
                                        "features": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "limitations": {
                                            "type": ["string", "null"]
                                        },
                                    },
                                    "required": [
                                        "provider",
                                        "model",
                                        "input_price",
                                        "output_price",
                                        "unit",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["records"],
                        "additionalProperties": False,
                    },
                )
            )

        return tools

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        expected_arguments = {
            "find_prices": {"providers", "model"},
            "scrape_provider_pricing": {"providers"},
            "save_prices": {"records"},
        }
        unexpected = set(arguments) - expected_arguments.get(name, set())
        if unexpected:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Unexpected arguments: " + ", ".join(sorted(unexpected))
                    ),
                }
            )

        if name == "find_prices":
            result = await self.find_prices(
                arguments.get("providers"),
                arguments.get("model"),
            )
        elif name == "scrape_provider_pricing":
            result = await self.scrape_provider_pricing(arguments.get("providers"))
        elif name == "save_prices":
            result = await self.save_prices(arguments.get("records"))
        else:
            result = {"ok": False, "error": f"Unknown pricing tool: {name}"}
        return json.dumps(result, ensure_ascii=False)

    async def find_prices(
        self,
        providers: Any,
        model: Any,
    ) -> dict[str, Any]:
        try:
            canonical_providers = self._canonical_providers(providers)
            model_text = self._required_text(model, "model")
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        provider_values = ", ".join(
            self._sql_text(provider) for provider in canonical_providers
        )
        result = await self.sqlite_client.call_tool(
            "read_query",
            {
                "query": (
                    "SELECT id, company_name, plan_name, input_tokens, "
                    "output_tokens, currency, billing_period, features, "
                    "limitations, source_query, created_at "
                    "FROM pricing_plans WHERE company_name COLLATE NOCASE IN "
                    f"({provider_values}) ORDER BY created_at DESC, id DESC"
                )
            },
        )
        try:
            raw_result = self._mcp_value(result)
            rows = (
                raw_result
                if isinstance(raw_result, list)
                else ast.literal_eval(raw_result)
            )
            if not isinstance(rows, list):
                raise ValueError("database result was not a list")
        except (SyntaxError, TypeError, ValueError) as exc:
            return {"ok": False, "error": f"Could not parse database result: {exc}"}

        requested = {self._normalize(provider) for provider in canonical_providers}
        model_key = self._normalize(model_text)
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for row in rows:
            if not isinstance(row, dict):
                continue
            provider_key = self._normalize(str(row.get("company_name", "")))
            plan_key = self._normalize(str(row.get("plan_name", "")))
            if provider_key not in requested or model_key not in plan_key:
                continue
            record_key = (provider_key, plan_key)
            if record_key in seen:
                continue
            seen.add(record_key)
            records.append(
                {
                    "provider": row.get("company_name"),
                    "model": row.get("plan_name"),
                    "input_price": row.get("input_tokens"),
                    "output_price": row.get("output_tokens"),
                    "currency": row.get("currency"),
                    "unit": row.get("billing_period"),
                    "features": self._decode_features(row.get("features")),
                    "limitations": row.get("limitations"),
                    "source": row.get("source_query"),
                    "created_at": row.get("created_at"),
                }
            )

        found = {self._normalize(str(record["provider"])) for record in records}
        missing = [
            provider
            for provider in canonical_providers
            if self._normalize(provider) not in found
        ]
        return {
            "ok": True,
            "source": "database",
            "records": records,
            "missing_providers": missing,
        }

    async def scrape_provider_pricing(self, providers: Any) -> dict[str, Any]:
        try:
            canonical_providers = self._canonical_providers(providers)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        websites = {
            provider: self.providers[self._normalize(provider)][1]
            for provider in canonical_providers
        }
        result = await self.firecrawl_client.call_tool(
            "scrape_websites",
            {"websites": websites, "formats": ["markdown", "html"]},
        )
        try:
            raw_result = self._mcp_value(result)
            content = (
                raw_result
                if isinstance(raw_result, dict)
                else json.loads(raw_result)
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return {"ok": False, "error": f"Could not parse scrape result: {exc}"}

        if not isinstance(content, dict):
            return {"ok": False, "error": "Scrape result was not an object"}

        useful_content = {
            provider: text
            for provider, text in content.items()
            if isinstance(text, str) and text.strip()
        }
        succeeded = {
            self._normalize(provider)
            for provider in useful_content
            if self._normalize(provider) in self.providers
        }
        self._last_scraped_providers = succeeded
        failed = [
            provider
            for provider in canonical_providers
            if self._normalize(provider) not in succeeded
        ]

        return {
            "ok": bool(useful_content),
            "source": "fresh_scrape",
            "providers": [self.providers[key][0] for key in sorted(succeeded)],
            "failed_providers": failed,
            "content": useful_content,
            **(
                {}
                if useful_content
                else {"error": "No non-empty pricing content was scraped"}
            ),
        }

    async def save_prices(self, records: Any) -> dict[str, Any]:
        if not self._last_scraped_providers:
            return {"ok": False, "error": "No successful scrape is active"}
        if not isinstance(records, list) or not 1 <= len(records) <= 100:
            return {"ok": False, "error": "records must contain 1 to 100 items"}

        validated: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            try:
                validated.append(self._validate_record(record))
            except ValueError as exc:
                rejected.append({"index": index, "error": str(exc)})

        if rejected:
            return {"ok": False, "saved_count": 0, "rejected": rejected}

        values = ", ".join(self._record_values(record) for record in validated)
        query = (
            "INSERT INTO pricing_plans "
            "(company_name, plan_name, input_tokens, output_tokens, currency, "
            "billing_period, features, limitations, source_query) VALUES "
            f"{values}"
        )
        await self.sqlite_client.call_tool(
            "write_query",
            {"query": query},
        )
        return {"ok": True, "saved_count": len(validated)}

    def _validate_record(self, record: Any) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        allowed_fields = {
            "provider",
            "model",
            "input_price",
            "output_price",
            "currency",
            "unit",
            "features",
            "limitations",
        }
        unexpected = set(record) - allowed_fields
        if unexpected:
            raise ValueError(
                "unexpected record fields: " + ", ".join(sorted(unexpected))
            )
        provider_key = self._normalize(self._required_text(record.get("provider"), "provider"))
        if provider_key not in self._last_scraped_providers:
            raise ValueError("provider was not part of the successful scrape")

        input_price = self._price(record.get("input_price"), "input_price")
        output_price = self._price(record.get("output_price"), "output_price")
        if input_price is None and output_price is None:
            raise ValueError("at least one price is required")

        features = record.get("features", [])
        if not isinstance(features, list) or not all(
            isinstance(feature, str) for feature in features
        ):
            raise ValueError("features must be a list of strings")

        limitations = record.get("limitations")
        if limitations is not None and not isinstance(limitations, str):
            raise ValueError("limitations must be a string or null")

        return {
            "provider": self.providers[provider_key][0],
            "model": self._required_text(record.get("model"), "model"),
            "input_price": input_price,
            "output_price": output_price,
            "currency": self._required_text(record.get("currency", "USD"), "currency"),
            "unit": self._required_text(record.get("unit"), "unit"),
            "features": features,
            "limitations": limitations,
            "source": self.providers[provider_key][1],
        }

    def _record_values(self, record: dict[str, Any]) -> str:
        values = [
            self._sql_text(record["provider"]),
            self._sql_text(record["model"]),
            self._sql_number(record["input_price"]),
            self._sql_number(record["output_price"]),
            self._sql_text(record["currency"]),
            self._sql_text(record["unit"]),
            self._sql_text(json.dumps(record["features"], ensure_ascii=False)),
            self._sql_text(record["limitations"]),
            self._sql_text(record["source"]),
        ]
        return f"({', '.join(values)})"

    def _canonical_providers(self, providers: Any) -> list[str]:
        if not isinstance(providers, list) or not providers:
            raise ValueError("providers must be a non-empty list")
        canonical: list[str] = []
        seen: set[str] = set()
        for provider in providers:
            if not isinstance(provider, str):
                raise ValueError("every provider must be a string")
            key = self._normalize(provider)
            if key not in self.providers:
                allowed = ", ".join(sorted(name for name, _ in self.providers.values()))
                raise ValueError(f"unknown provider '{provider}'; allowed: {allowed}")
            if key not in seen:
                seen.add(key)
                canonical.append(self.providers[key][0])
        return canonical

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @staticmethod
    def _required_text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _price(value: Any, field: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a number or null")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{field} must be finite and non-negative")
        return number

    @staticmethod
    def _sql_text(value: str | None) -> str:
        if value is None:
            return "NULL"
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _sql_number(value: float | None) -> str:
        return "NULL" if value is None else repr(value)

    @staticmethod
    def _decode_features(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _model_tool(
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one OpenAI-compatible schema for a local application tool."""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema,
            },
        }

    @staticmethod
    def _mcp_value(result: Any) -> Any:
        """Read a FastMCP result without discarding structured content."""
        if getattr(result, "is_error", False):
            message = PriceTools._mcp_text(result) or "Unknown MCP tool error"
            raise ValueError(message)

        data = getattr(result, "data", None)
        if data is not None:
            return data

        structured = getattr(result, "structured_content", None)
        if structured is not None:
            if isinstance(structured, dict) and set(structured) == {"result"}:
                return structured["result"]
            return structured

        return PriceTools._mcp_text(result)

    @staticmethod
    def _mcp_text(result: Any) -> str:
        text_blocks = [
            content.text
            for content in getattr(result, "content", [])
            if getattr(content, "type", None) == "text"
        ]
        return "\n".join(text_blocks)
