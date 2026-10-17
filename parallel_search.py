"""Anonymous Parallel Search MCP adapter for lanGen's research tools."""

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Implementation

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "lanGen/0.1.0 (https://github.com/YashasviAsthana/lanGen)"
REQUEST_TIMEOUT = 30.0


def is_web_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def _payload(result: CallToolResult) -> dict[str, Any]:
    text = "\n".join(c.text for c in result.content if c.type == "text")
    if result.isError:
        raise ValueError(text or "Parallel MCP tool failed.")
    data = result.structuredContent
    if data is None:
        data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Unexpected Parallel MCP response.")
    return data


async def _request(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # No API keys, environment credentials, cookies or OAuth are loaded here.
    try:
        async with (
            httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
            ) as http,
            streamable_http_client(MCP_URL, http_client=http) as (read, write, _),
            ClientSession(
                read,
                write,
                client_info=Implementation(name="lanGen", version="0.1.0"),
            ) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            if name not in {tool.name for tool in tools.tools}:
                raise ValueError(f"Parallel MCP does not offer {name}.")
            result = await session.call_tool(name, arguments)
    except ExceptionGroup as exc:
        # The SDK's task groups wrap HTTP/protocol failures; retain the cause.
        cause: Exception = exc
        while isinstance(cause, ExceptionGroup):
            cause = cause.exceptions[0]
        raise cause from exc
    return _payload(result)


async def _call(
    name: str, arguments: dict[str, Any], cancelled: asyncio.Event | None
) -> dict[str, Any]:
    if cancelled and cancelled.is_set():
        raise asyncio.CancelledError
    request = asyncio.create_task(_request(name, arguments))
    stop = asyncio.create_task(cancelled.wait()) if cancelled else None
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            if stop:
                done, _ = await asyncio.wait(
                    {request, stop}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop in done:
                    raise asyncio.CancelledError
            return await request
    except TimeoutError as exc:
        raise TimeoutError("Parallel MCP request timed out.") from exc
    finally:
        request.cancel()
        if stop:
            stop.cancel()
        await asyncio.gather(request, *([stop] if stop else []), return_exceptions=True)


def _metadata(session_id: str, model: str) -> dict[str, str]:
    # Both values originate in the conversation/runtime, never in tool arguments.
    return {
        key: value
        for key, value in (("session_id", session_id), ("model_name", model))
        if value and len(value) <= 100
    }


async def search(
    query: str,
    max_results: int,
    *,
    session_id: str = "",
    model: str = "",
    cancelled: asyncio.Event | None = None,
) -> list[dict[str, str]]:
    data = await _call(
        "web_search",
        {"objective": query, "search_queries": [query], **_metadata(session_id, model)},
        cancelled,
    )
    results = []
    for item in data["results"][:max_results]:
        if not isinstance(item, dict) or not is_web_url(item.get("url", "")):
            raise ValueError("Unexpected Parallel search result URL.")
        excerpts = item.get("excerpts", [])
        if not isinstance(excerpts, list) or any(
            not isinstance(e, str) for e in excerpts
        ):
            raise ValueError("Unexpected Parallel search excerpts.")
        results.append(
            {
                "title": item.get("title") or "",
                "url": item["url"],
                "snippet": "\n".join(excerpts),
            }
        )
    return results


async def fetch(
    url: str,
    *,
    session_id: str = "",
    model: str = "",
    cancelled: asyncio.Event | None = None,
) -> str:
    if not is_web_url(url):
        raise ValueError("URL must start with http:// or https://.")
    data = await _call(
        "web_fetch", {"urls": [url], **_metadata(session_id, model)}, cancelled
    )
    if not data["results"]:
        raise ValueError(
            f"Parallel could not fetch this page: {data.get('errors', [])}"
        )
    item = data["results"][0]
    if not isinstance(item, dict) or item.get("url") != url:
        raise ValueError("Unexpected Parallel fetch result URL.")
    excerpts = item.get("excerpts", [])
    if not isinstance(excerpts, list) or any(not isinstance(e, str) for e in excerpts):
        raise ValueError("Unexpected Parallel fetch excerpts.")
    text = "\n".join(excerpts).strip()
    if not text:
        raise ValueError("Parallel returned no page content.")
    return text
