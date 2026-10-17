"""Exercise the real resolver, MCP SDK transport and agent result consumption.

The model's SSE replies and remote HTTP responses are scripted; no LLM, GPU,
network access or Parallel credentials are needed for these deterministic tests.
"""

import asyncio
import configparser
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import agent
import comfy
import parallel_search
import setup

PAGE_URL = "https://example.org/research"
SESSION_ID = "d5728991-54b3-4a31-a1ba-31c269f4a9ad"
MODEL = "local-model.gguf"
HTTP_CLIENT = httpx.AsyncClient


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = configparser.ConfigParser()
        self.cfg["search"] = {"provider": "parallel"}
        self.enterContext(patch.object(agent, "_cfg", self.cfg))
        self.enterContext(patch.dict(os.environ, {"PARALLEL_API_KEY": "unused-key"}))
        self.requests = []
        self.model_requests = []
        self.model_replies = []
        self.payload = None
        self.tool_error = False
        self.http_error = None
        self.block = False
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.enterContext(patch.object(httpx, "AsyncClient", self.client))
        self.ddg = self.enterContext(
            patch.object(
                agent, "_ddgs_search", side_effect=AssertionError("DDG fallback")
            )
        )

    def client(self, **kwargs):
        return HTTP_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)

    async def handle(self, request):
        if request.url.host == "localhost":
            self.model_requests.append(json.loads(request.content))
            delta = self.model_replies.pop(0)
            content = "data: " + json.dumps({"choices": [{"delta": delta}]})
            return httpx.Response(
                200,
                text=content + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        self.requests.append(request)
        if str(request.url) == PAGE_URL:
            return httpx.Response(
                200,
                text="<p>Direct page content</p>",
                headers={"content-type": "text/html"},
            )
        self.assertEqual(str(request.url), parallel_search.MCP_URL)
        self.assertEqual(request.headers["user-agent"], parallel_search.USER_AGENT)
        self.assertNotIn("authorization", request.headers)
        self.assertNotIn("cookie", request.headers)
        self.assertIn("timeout", request.extensions)
        message = json.loads(request.content)
        method = message["method"]
        if "id" not in message:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        elif method == "tools/call":
            self.started.set()
            if self.http_error:
                return httpx.Response(self.http_error, text="Service unavailable")
            if self.block:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.closed.set()
            params = message["params"]
            data = self.payload
            if data is None:
                data = {
                    "results": [
                        {
                            "title": "Research",
                            "url": PAGE_URL,
                            "excerpts": ["Useful research evidence"],
                        }
                    ]
                }
            result = {
                "isError": self.tool_error,
                "content": [{"type": "text", "text": json.dumps(data)}],
            }
            # Cover both MCP structured output and JSON text compatibility.
            if params["name"] == "web_search" and not self.tool_error:
                result["structuredContent"] = data
        else:
            self.fail(f"Unexpected MCP method: {method}")
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    def calls(self):
        return [
            json.loads(r.content)["params"]
            for r in self.requests
            if r.content and json.loads(r.content).get("method") == "tools/call"
        ]

    async def test_agent_loop_search_fetch_and_sources(self):
        def call(name, args, number):
            return {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call-{number}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ]
            }

        self.model_replies = [
            call("web_search", {"query": "current research evidence"}, 1),
            call("fetch_page", {"url": PAGE_URL}, 2),
            {"content": "An answer based on the research evidence."},
        ]
        events = [
            json.loads(event)
            async for event in agent.agent_stream(
                [{"role": "user", "content": "Research this topic"}],
                tools=comfy.build_tools_from_workflows({}),
                session_id=SESSION_ID,
                model=MODEL,
            )
        ]
        search_result = self.model_requests[1]["messages"][-1]
        self.assertEqual(search_result["role"], "tool")
        normalized = json.loads(search_result["content"])[0]
        self.assertEqual(
            normalized,
            {
                "title": "Research",
                "url": PAGE_URL,
                "snippet": "Useful research evidence",
            },
        )
        fetched = [
            m for m in self.model_requests[2]["messages"] if m["role"] == "tool"
        ][-1]
        self.assertIn(f"Source: {PAGE_URL}", fetched["content"])
        self.assertIn("Useful research evidence", fetched["content"])
        self.assertTrue(any(e.get("type") == "text" for e in events))
        self.assertTrue(any(PAGE_URL in e.get("urls", []) for e in events))
        calls = self.calls()
        self.assertEqual([c["name"] for c in calls], ["web_search", "web_fetch"])
        self.assertEqual(
            calls[0]["arguments"]["search_queries"], ["current research evidence"]
        )
        self.assertEqual(calls[1]["arguments"]["urls"], [PAGE_URL])
        for c in calls:
            self.assertEqual(c["arguments"]["session_id"], SESSION_ID)
            self.assertEqual(c["arguments"]["model_name"], MODEL)
        self.ddg.assert_not_called()

    async def test_default_and_saved_duckduckgo_ignore_ambient_key(self):
        self.ddg.side_effect = None
        self.ddg.return_value = [{"title": "Default", "href": PAGE_URL, "body": "DDG"}]
        for cfg in (configparser.ConfigParser(), self.cfg):
            if cfg is self.cfg:
                cfg["search"] = {"provider": "duckduckgo"}
            with patch.object(agent, "_cfg", cfg):
                result = json.loads(await agent.exec_web_search("query"))
                self.assertEqual(result[0]["snippet"], "DDG")
                async with self.client() as http:
                    self.assertIn(
                        "Direct page content", await agent.fetch_page(PAGE_URL, http)
                    )
        self.assertFalse(self.calls())

    async def test_disabled_and_invalid_provider_make_no_requests(self):
        for provider in ("disabled", "invalid"):
            self.cfg["search"]["provider"] = provider
            self.assertTrue(await agent.exec_web_search("query"))
            async with self.client() as http:
                self.assertTrue(await agent.fetch_page(PAGE_URL, http))
        self.assertFalse(self.requests)
        self.ddg.assert_not_called()

    async def test_unadvertised_tool_is_denied(self):
        self.model_replies = [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "denied",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"query"}',
                        },
                    }
                ]
            },
            {"content": "Tool unavailable."},
        ]
        _ = [event async for event in agent.agent_stream([], tools=[])]
        content = self.model_requests[1]["messages"][-1]["content"]
        self.assertIn("Tool is not enabled", content)
        self.assertFalse(self.requests)

    async def test_mcp_errors_do_not_fall_back(self):
        for status in (401, 429, 500):
            self.http_error = status
            result = await agent.exec_web_search("query")
            self.assertTrue(result.startswith("Search error:"), result)
        self.http_error = None
        self.tool_error = True
        self.payload = {"error": "anonymous quota exceeded"}
        self.assertIn("anonymous quota exceeded", await agent.exec_web_search("query"))
        self.ddg.assert_not_called()

    async def test_malformed_results_and_empty_search(self):
        for payload in ({"unexpected": []}, {"results": [{"url": "file:///secret"}]}):
            self.payload = payload
            self.assertIn("Search error:", await agent.exec_web_search("query"))
        self.payload = {"results": []}
        self.assertEqual(await agent.exec_web_search("query"), "No results found.")

    async def test_limits_and_fetch_errors(self):
        self.payload = {
            "results": [{"url": PAGE_URL, "title": None, "excerpts": ["x" * 7000]}] * 15
        }
        results = json.loads(await agent.exec_web_search("query", 99))
        self.assertEqual(len(results), 10)
        async with self.client() as http:
            text = await agent.fetch_page(PAGE_URL, http)
            self.assertTrue(text.endswith("x" * 6000 + "\n\n[Content truncated]"))
            count = len(self.requests)
            self.assertIn("URL must", await agent.fetch_page("file:///secret", http))
            self.assertEqual(len(self.requests), count)
            self.payload = {"results": [], "errors": [{"error_type": "not_found"}]}
            self.assertIn("not_found", await agent.fetch_page(PAGE_URL, http))

    async def test_timeout_closes_transport(self):
        self.block = True
        with patch.object(parallel_search, "REQUEST_TIMEOUT", 0.1):
            result = await agent.exec_web_search("query")
        self.assertIn("timed out", result)
        self.assertTrue(self.closed.is_set())

    async def test_user_cancellation_closes_transport(self):
        self.block = True
        cancelled = asyncio.Event()
        task = asyncio.create_task(agent.exec_web_search("query", cancelled=cancelled))
        await asyncio.wait_for(self.started.wait(), 2)
        cancelled.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(self.closed.is_set())

    async def test_task_cancellation_closes_transport(self):
        self.block = True
        task = asyncio.create_task(agent.exec_web_search("query"))
        await asyncio.wait_for(self.started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.closed.is_set())


class SetupTests(unittest.TestCase):
    def test_wizard_preserves_search_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lanGen.ini"
            path.write_text("[search]\nprovider = parallel\n")
            with patch.object(setup, "CONFIG_FILE", path):
                setup._write_config(Path("llama-server"), Path("model.gguf"), 0, "cpu")
            cfg = configparser.ConfigParser()
            cfg.read(path)
            self.assertEqual(cfg["search"]["provider"], "parallel")


if __name__ == "__main__":
    unittest.main()
