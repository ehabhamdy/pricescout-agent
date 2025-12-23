import json
import logging
import re
from typing import Any, Dict, List, Optional

from mcp_client import Client
from llm_client import LLMClient

logger = logging.getLogger(__name__)

class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""

    def __init__(self, sqlite_client: Client, llm_client: LLMClient):
        self.sqlite_client = sqlite_client
        self.llm_client = llm_client

    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:
            await self.sqlite_client.execute_tool("write_query", {
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
            })
            logging.info("✓ Data extraction tables initialized")
        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use LLM to extract structured data."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.get_response(messages)
            return response.strip()
        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'

    async def extract_and_store_data(self, user_query: str, llm_response: str) -> None:
        """Extract structured data from LLM response and store it."""
        try:
            extraction_prompt = f"""
            Analyze this text and extract pricing information in JSON format:

            Text: {llm_response}

            Extract pricing plans with this structure:
            {{
                "company_name": "company name",
                "plans": [
                    {{
                        "plan_name": "plan name",
                        "input_tokens": number or null,
                        "output_tokens": number or null,
                        "currency": "USD",
                        "billing_period": "monthly/yearly/one-time",
                        "features": ["feature1", "feature2"],
                        "limitations": "any limitations mentioned",
                        "query": "{user_query}"
                    }}
                ]
            }}

            Return only valid JSON, no other text. Do not return your response enclosed in ```json```
            """

            extraction_response = await self._get_structured_extraction(extraction_prompt)
            # Cleanup markdown code blocks if present
            extraction_response = extraction_response.replace("```json", "").replace("```", "").strip()

            try:
                pricing_data = json.loads(extraction_response)
            except json.JSONDecodeError as e:
                 logging.error(f"Failed to decode JSON from extraction response: {e}. Response was: {extraction_response}")
                 return

            company_name = pricing_data.get("company_name", "Unknown")

            for plan in pricing_data.get("plans", []):
                await self.sqlite_client.execute_tool("write_query", {
                    "query": """
                    INSERT INTO pricing_plans
                    (company_name, plan_name, input_tokens, output_tokens, currency,
                     billing_period, features, limitations, source_query)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    "args": [
                        company_name,
                        plan.get("plan_name"),
                        plan.get("input_tokens"),
                        plan.get("output_tokens"),
                        plan.get("currency", "USD"),
                        plan.get("billing_period"),
                        json.dumps(plan.get("features", [])),
                        plan.get("limitations"),
                        user_query
                    ]
                })

            logger.info(f"Stored {len(pricing_data.get('plans', []))} pricing plans for {company_name}")

        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")

    async def check_existing_prices(self, user_query: str) -> str | None:
        """Check if we already have pricing data relevant to the query."""
        extraction_prompt = f"""
        Extract the company name or service name from this query to check in the database.
        Query: {user_query}
        Return ONLY the name, nothing else.
        """
        messages = [{"role": "user", "content": extraction_prompt}]
        entity_name = self.llm_client.get_response(messages).strip()
        # Remove any quotes or extra text
        entity_name = re.sub(r'["\']', '', entity_name).strip().replace(" ", "")
        
        entities = [name.strip() for name in entity_name.split(",")]
        
        if not entity_name:
            return None

        logging.info(f"Checking database for: {entity_name}")

        try:
            query =  f"SELECT * FROM pricing_plans WHERE company_name IN ({', '.join(['"' + entity + '"' for entity in entities])}) ORDER BY created_at DESC"
            result = await self.sqlite_client.execute_tool("read_query", {
                "query": query,
            })
            
            if result and isinstance(result, str) and result.strip() != "[]":
                 return f"Found existing data for {entity_name}:\n{result}"
            
        except Exception as e:
            logging.debug(f"Failed to check existing prices (or no data found): {e}")
        
        return None
