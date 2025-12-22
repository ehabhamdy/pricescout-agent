import os
import json
from typing import Optional, List
from fastmcp import FastMCP
from firecrawl import Firecrawl
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Firecrawl MCP Server")

api_key = os.getenv("FIRECRAWL_API_KEY")
if not api_key:
    raise ValueError("FIRECRAWL_API_KEY environment variable not set")

app = Firecrawl(api_key=api_key)

class AIModelPricing(BaseModel):
    model_name: str
    context_window_k: int = Field(description="Context window in thousands (e.g., 160)")
    input_cost_per_1m: float
    input_cached_cost_per_1m: Optional[float] = None
    output_cost_per_1m: float

class PricingCatalog(BaseModel):
    models: List[AIModelPricing]

@mcp.tool
def scrape_model_prices(url: str = "https://deepinfra.com/pricing") -> str:
    """
    Scrape LLM model prices from a supported website (e.g., deepinfra.com).
    args:
        url: URL to scrape (required)
    returns:
        JSON string containing the pricing catalog
    """
    try:
        result = app.scrape(
            url=url,
            formats=[{
                "type": "json",
                "schema": PricingCatalog.model_json_schema()
            }],
            only_main_content=False,
            timeout=120000
        )
        
        # result is expected to be a dict or have a .json property depending on SDK version
        # Based on example: print(result.json)
        # Checking if result has 'json' attribute or key
        if hasattr(result, 'json'):
             return json.dumps(result.json, indent=2)
        elif isinstance(result, dict) and 'json' in result:
             return json.dumps(result['json'], indent=2)
        else:
             return str(result)

    except Exception as e:
        return f"Error scraping prices: {str(e)}"

if __name__ == "__main__":
    mcp.run()
