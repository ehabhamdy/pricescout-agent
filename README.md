# PriceScout Agent

PriceScout is a learning project that connects a chat model to local MCP servers
for SQLite and Firecrawl.

## Tool-calling loop

Each CLI query follows the native model tool-calling protocol:

1. Connect to MCP servers and verify the private capabilities the app needs.
2. Send the conversation and the currently available narrow pricing-tool schemas
   to the model. Two are initially available; `save_prices` appears only after a
   successful scrape.
3. If the model returns structured tool calls, dispatch each local pricing tool.
   The pricing tool calls the underlying FastMCP client's `call_tool()` method
   directly when it needs SQLite or Firecrawl.
4. Append each result as a `role: tool` message with the model's matching
   `tool_call_id`.
5. Send the updated messages back to the model.
6. Repeat until the model returns a normal assistant message.

Tool-call messages and results stay together for the current query. Only the
completed user and assistant messages are kept as short conversational history.
The loop is capped at ten model iterations to prevent an accidental infinite
cycle.

## Narrow pricing tools

The model never receives SQLite's generic `read_query` or `write_query` tools,
and it never receives Firecrawl's arbitrary-URL `scrape_websites` tool. Those MCP
tools are private implementation details behind:

- `find_prices(providers, model)` — searches cached records and reports which
  requested providers are missing.
- `scrape_provider_pricing(providers)` — maps provider names to allowlisted URLs
  from `providers.json`, then scrapes those pages.
- `save_prices(records)` — validates structured price records and saves them.

`save_prices` is hidden until a successful scrape exists. It accepts records only
for providers from that scrape, rejects invalid or negative prices, escapes text
before building the private SQLite write, and performs writes without retries.
The model must call `find_prices` again to verify saved data before answering.

## MCP clients

`mcp_client.py` contains only `MCPClient`, which owns the lifetime of one
stdio MCP subprocess and its FastMCP client. It does not wrap MCP discovery or
tool execution. `Agent` calls `client.list_tools()` directly during startup, and
`PriceTools` calls `client.call_tool()` directly at runtime. This preserves the
native FastMCP result object, including structured content and MCP errors.

## Run

The project expects `LLM_PROVIDER`, `LLM_API_KEY`, and `FIRECRAWL_API_KEY` in the
local `.env` file.

```bash
source .venv/bin/activate
python main.py
```

## Test

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```
