import os
import json
from typing import List, Dict, Optional
from fastmcp import FastMCP
from firecrawl import Firecrawl
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from urllib.parse import urlparse

load_dotenv()
logger = logging.getLogger(__name__)

mcp = FastMCP("Firecrawl MCP Server")

SCRAPE_DIR = "scraped_content"

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


@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown', 'html'],
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.
    
    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites
    """
    
    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)
    
    metadata_file = os.path.join(path, "scraped_metadata.json")
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode metadata file: {metadata_file}. Starting fresh.")
            metadata = {}

    successful_scrapes = []

    for provider_name, url in websites.items():
        logger.info(f"Scraping {provider_name} at {url}")
        
        try:
            # TODO: Check if recently scraped? For now, we overwrite or update.
            
            scrape_result = app.scrape(url, formats=formats)
       
            if not scrape_result:
                logger.error(f"No result returned for {provider_name}")
                continue

            content_files = {}
            timestamp = datetime.now().isoformat()
            
            for fmt in formats:
                content = getattr(scrape_result, fmt)
                if content:
                    filename = f"{provider_name}.{fmt}"
                    file_path = os.path.join(path, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    content_files[fmt] = filename
            
            # Update metadata
            metadata[provider_name] = {
                "provider_name": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": timestamp,
                "formats": formats,
                "content_files": content_files,
                "title": scrape_result.metadata.title,
                "description": scrape_result.metadata.description,
                "scrap_id": scrape_result.metadata.scrape_id,
                "success": "success" if scrape_result.metadata.status_code == 200 else "failure",
            }
            
            successful_scrapes.append(provider_name)
            logger.info(f"Successfully scraped {provider_name}")

        except Exception as e:
            logger.error(f"Failed to scrape {provider_name}: {e}")
            # Optionally record failure in metadata
            
    # Save metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    return successful_scrapes

@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information
    """
    
    logger.info(f"Extracting information for identifier: {identifier}")
    # logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR)}") # Optional debug

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    if not os.path.exists(metadata_file):
        return json.dumps({"error": "No scraped data found yet."}, indent=2)

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        return json.dumps({"error": f"Failed to read metadata: {str(e)}"}, indent=2)

    # Search strategy:
    # 1. Exact match on provider_name (key)
    # 2. Exact match on url
    # 3. Match on domain
    
    if identifier in metadata:
        return json.dumps(metadata[identifier], indent=2)
        
    for key, data in metadata.items():
        if data.get('url') == identifier:
            return json.dumps(data, indent=2)
        if identifier in data.get('domain', ''): # Loose match for domain
            return json.dumps(data, indent=2)
            
    return json.dumps({"error": f"No information found for identifier: {identifier}"}, indent=2)


if __name__ == "__main__":
    mcp.run()
