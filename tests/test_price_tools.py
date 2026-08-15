import json
import unittest
from dataclasses import dataclass, field
from typing import Any

from price_tools import PriceTools


@dataclass
class FakeCallToolResult:
    data: Any = None
    structured_content: dict[str, Any] | None = None
    content: list[Any] = field(default_factory=list)
    is_error: bool = False


class FakeClient:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> FakeCallToolResult:
        self.calls.append((name, arguments))
        response = self.responses[name]
        if isinstance(response, FakeCallToolResult):
            return response
        return FakeCallToolResult(data=response)


class PriceToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.sqlite = FakeClient(
            {
                "write_query": "[{'affected_rows': 1}]",
                "read_query": (
                    "[{'id': 4, 'company_name': 'fireworks', "
                    "'plan_name': 'DeepSeek V3', 'input_tokens': 0.56, "
                    "'output_tokens': 1.68, 'currency': 'USD', "
                    "'billing_period': 'per million tokens', 'features': '[]', "
                    "'limitations': None, 'source_query': 'official page', "
                    "'created_at': '2026-08-13 00:00:00'}]"
                ),
            }
        )
        self.firecrawl = FakeClient(
            {
                "scrape_websites": (
                    '{"fireworks":"DeepSeek V3: $0.56 input, $1.68 output"}'
                )
            }
        )
        self.tools = PriceTools(
            self.sqlite,
            self.firecrawl,
            {
                "fireworks": "https://fireworks.ai/pricing",
                "deepinfra": "https://deepinfra.com/pricing",
            },
        )

    async def test_find_prices_filters_without_model_supplied_sql(self) -> None:
        result = await self.tools.find_prices(
            ["fireworks", "deepinfra"],
            "DeepSeek V3",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["records"][0]["provider"], "fireworks")
        self.assertEqual(result["missing_providers"], ["deepinfra"])
        query = self.sqlite.calls[0][1]["query"]
        self.assertNotIn("DeepSeek V3", query)
        self.assertIn("'fireworks'", query)
        self.assertIn("'deepinfra'", query)

    async def test_find_prices_accepts_native_structured_mcp_data(self) -> None:
        self.sqlite.responses["read_query"] = FakeCallToolResult(
            data=[
                {
                    "id": 4,
                    "company_name": "fireworks",
                    "plan_name": "DeepSeek V3",
                    "input_tokens": 0.56,
                    "output_tokens": 1.68,
                    "currency": "USD",
                    "billing_period": "per million tokens",
                    "features": "[]",
                    "limitations": None,
                    "source_query": "official page",
                    "created_at": "2026-08-13 00:00:00",
                }
            ]
        )

        result = await self.tools.find_prices(["fireworks"], "DeepSeek V3")

        self.assertTrue(result["ok"])
        self.assertEqual(result["records"][0]["input_price"], 0.56)

    async def test_scrape_uses_allowlisted_url_and_enables_save(self) -> None:
        result = await self.tools.scrape_provider_pricing(["fireworks"])

        self.assertTrue(result["ok"])
        arguments = self.firecrawl.calls[0][1]
        self.assertEqual(
            arguments["websites"],
            {"fireworks": "https://fireworks.ai/pricing"},
        )
        tool_names = {
            tool["function"]["name"] for tool in self.tools.model_tools()
        }
        self.assertEqual(
            tool_names,
            {"find_prices", "scrape_provider_pricing", "save_prices"},
        )

    async def test_unknown_provider_is_rejected_before_scraping(self) -> None:
        result = await self.tools.scrape_provider_pricing(["attacker.example"])

        self.assertFalse(result["ok"])
        self.assertEqual(self.firecrawl.calls, [])

    async def test_unexpected_tool_arguments_are_rejected(self) -> None:
        result = await self.tools.execute(
            "scrape_provider_pricing",
            {
                "providers": ["fireworks"],
                "url": "https://attacker.example/pricing",
            },
        )

        self.assertFalse(json.loads(result)["ok"])
        self.assertEqual(self.firecrawl.calls, [])

    async def test_save_validates_scope_and_escapes_text(self) -> None:
        await self.tools.scrape_provider_pricing(["fireworks"])
        result = await self.tools.save_prices(
            [
                {
                    "provider": "fireworks",
                    "model": "Vendor's DeepSeek V3",
                    "input_price": 0.56,
                    "output_price": 1.68,
                    "currency": "USD",
                    "unit": "per million tokens",
                    "features": ["cached input"],
                    "limitations": None,
                }
            ]
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["saved_count"], 1)
        query = self.sqlite.calls[-1][1]["query"]
        self.assertIn("Vendor''s DeepSeek V3", query)
        self.assertIn("https://fireworks.ai/pricing", query)

    async def test_save_rejects_provider_not_in_scrape(self) -> None:
        await self.tools.scrape_provider_pricing(["fireworks"])
        result = await self.tools.save_prices(
            [
                {
                    "provider": "deepinfra",
                    "model": "DeepSeek V3",
                    "input_price": 0.32,
                    "output_price": 0.89,
                    "unit": "per million tokens",
                }
            ]
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["saved_count"], 0)


if __name__ == "__main__":
    unittest.main()
