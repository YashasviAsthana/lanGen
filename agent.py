"""
Personal LAN Agent
==================
• Connects to a local llama.cpp server (OpenAI-compatible API)
• Tools: DuckDuckGo web search + Stable Diffusion image generation

Run:
    python agent.py
    Then open  https://<your-ip>:7000  from any device on the LAN
"""

import asyncio
import base64
import configparser
import json
import logging
import mimetypes
import os
import random
import socket
import subprocess
import sys
import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import IO, Any

import html2text
import httpx
import trafilatura
import uvicorn
import websockets
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import comfy
from comfy import (
    _SKIP_NODE_CLASSES,
    IMAGE_SERVER,
    _extract_pdf_text,
    _gen_dir,
    _input_dir,
    _resize_image,
    _static_dir,
    build_tools_from_workflows,
    exec_generate_image,
    init_comfy,
    load_comfy_workflows,
)

log = logging.getLogger("agent")
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
_log_level_num = getattr(logging, _log_level, logging.INFO)
if not log.handlers:
    log.setLevel(_log_level_num)
    _h = logging.StreamHandler()
    _h.setLevel(_log_level_num)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(_h)

try:
    from ddgs import DDGS
except ImportError:
    print("Missing dependency. Run:  pip install duckduckgo-search")
    sys.exit(1)

try:
    from PIL import Image as PILImage
except ImportError:
    print("Missing dependency. Run:  pip install pillow")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent

# ── Session / chat persistence ────────────────────────────────────────────────

CHAT_DIR = ROOT / "chats"


def _ensure_chat_dir() -> None:
    CHAT_DIR.mkdir(exist_ok=True)


def _session_file(session_id: str) -> Path:
    return CHAT_DIR / f"{session_id}.json"


def get_session_id(request: Request) -> str:
    """Read session_id cookie, generate UUID if missing. Sets cookie on first visit."""
    cookie = request.cookies.get("session_id")
    if cookie:
        return cookie
    return str(uuid.uuid4())


def load_session(session_id: str) -> dict:
    _ensure_chat_dir()
    f = _session_file(session_id)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return {"id": session_id, "history": []}


def save_session(session_id: str, history: list[Any]) -> None:
    _ensure_chat_dir()
    f = _session_file(session_id)
    f.write_text(
        json.dumps({"id": session_id, "history": history}, ensure_ascii=False, indent=2)
    )


def _extract_preview(history: list) -> str:
    """Extract a short text preview from session history."""
    for m in history:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part["text"][:80]
        return str(content)[:80]
    return ""


def list_sessions() -> list:
    _ensure_chat_dir()
    sessions = []
    for f in CHAT_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            preview = _extract_preview(data.get("history", []))
            sessions.append(
                {
                    "id": data.get("id", f.stem),
                    "preview": preview or "(empty)",
                    "timestamp": f.stat().st_mtime,
                    "time": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                        "%b %d, %H:%M"
                    ),
                }
            )
        except (json.JSONDecodeError, OSError):
            pass
    sessions.sort(key=lambda s: s["timestamp"], reverse=True)
    return sessions


def create_session() -> str:
    sid = str(uuid.uuid4())
    save_session(sid, [])
    return sid


def delete_session(session_id: str) -> None:
    f = _session_file(session_id)
    if f.exists():
        f.unlink()


# ── Configuration ─────────────────────────────────────────────────────────────

LLAMA_SERVER = "http://localhost:8080"  # llama.cpp OpenAI-compat API
BIND_HOST = "0.0.0.0"  # 0.0.0.0 = accessible on LAN
BIND_PORT = 7000
MAX_ITERATIONS = 50  # max tool-call rounds per turn

# ── lanGen.ini config (written by setup.py) ───────────────────────────────────

_CONFIG_FILE = ROOT / "lanGen.ini"
_cfg = configparser.ConfigParser()
_cfg.read(_CONFIG_FILE)

if "llama" in _cfg:
    _llama_port = _cfg["llama"].get("port", "8080")
    LLAMA_SERVER = f"http://localhost:{_llama_port}"

if "comfyui" in _cfg:
    _comfy_port = _cfg["comfyui"].get("port", "7860")
    IMAGE_SERVER = f"http://localhost:{_comfy_port}"
    _workspace = _cfg["comfyui"].get("workspace", "")
    _comfy_models_dir = Path(_workspace) / "models" if _workspace else None
    init_comfy(_comfy_models_dir)


# ── Process handles (set by lifespan if managed=true) ─────────────────────────


@dataclass
class _ServerState:
    proc: "subprocess.Popen | None" = None
    log_fh: IO[str] | None = None


_TaskVal = dict[str, Any]


@dataclass
class AppState:
    """Runtime application state initialised once in lifespan and stored on app.state."""

    model: str
    registry: dict[str, Any]
    tools: list[dict[str, Any]]
    models_dir: Path | None
    task_store: dict[str, _TaskVal] = field(default_factory=dict)
    task_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ── SSL ────────────────────────────────────────────────────────────────────────

CERT_DIR = os.path.join(os.path.dirname(__file__), "certs")
CERT_FILE = os.path.join(CERT_DIR, "cert.pem")
KEY_FILE = os.path.join(CERT_DIR, "key.pem")


def _get_san_list() -> list[str]:
    """Collect SANs: localhost, 127.0.0.1, ::1, and all LAN IPs."""
    sans = ["DNS:localhost", "IP:127.0.0.1", "IP:::1"]
    try:
        proc = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                ip_cidr = parts[3]
                ip = ip_cidr.split("/")[0]
                if not ip.startswith("127.") and ip not in sans:
                    sans.append(f"IP:{ip}")
    except (OSError, subprocess.SubprocessError):
        pass
    seen: set[str] = set()
    deduped: list[str] = []
    for s in sans:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _ensure_ssl() -> tuple[str, str]:
    """Generate a self-signed cert with proper SANs, return (certfile, keyfile)."""
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return CERT_FILE, KEY_FILE
    os.makedirs(CERT_DIR, exist_ok=True)
    san_str = ",".join(_get_san_list())
    os.system(
        f'openssl req -x509 -newkey rsa:2048 -keyout "{KEY_FILE}" '
        f'-out "{CERT_FILE}" -days 365 -nodes -subj "/CN=localhost" '
        f'-addext "subjectAltName={san_str}" 2>/dev/null'
    )
    return CERT_FILE, KEY_FILE


_WORKFLOW_TIPS: dict[str, str] = {
    "generate_image": (
        "[generate_image — Z-Image Turbo, 4-step text-to-image]\n"
        "WHEN: user asks to create, draw, paint, render, or generate any new image from scratch with no existing image as input.\n"
        "PROMPT — write full natural sentences; the Qwen 3 4B text encoder understands grammar, not just comma-separated tags.\n"
        "  Structure every prompt in this order:\n"
        '    1. SUBJECT + ACTION   "A red fox perched on a mossy boulder in an autumn forest"\n'
        '    2. VISUAL STYLE       "photorealistic, shot on 35mm film" | "Studio Ghibli watercolour" | '
        '"oil painting, thick impasto" | "cinematic concept art" | "anime cel-shaded"\n'
        '    3. LIGHTING           "golden hour backlight" | "dramatic chiaroscuro" | "soft overcast diffuse" | '
        '"neon signs reflecting on wet pavement at night" | "volumetric god rays"\n'
        '    4. COMPOSITION        "extreme close-up, shallow DoF" | "wide-angle establishing shot" | '
        '"low angle looking up" | "bird\'s-eye view" | "rule of thirds, negative space"\n'
        '    5. COLOUR PALETTE     "warm amber and rust tones" | "muted desaturated earth tones" | '
        '"high-contrast neon" | "cool blues and steel greys" | "pastel watercolour wash"\n'
        '    6. QUALITY TAIL       always end with: "sharp focus, intricate detail, high resolution"\n'
        '  AVOID keyword dumps ("beautiful, amazing, gorgeous") — use specific descriptive phrases instead.\n'
        "  SIZES: 1024×1024 square | 1344×768 landscape | 768×1344 portrait | 1024×576 cinematic 16:9"
    ),
    "image_flux2_klein_image_edit": (
        "[image_flux2_klein_image_edit — Flux 2 Klein 4B, ReferenceLatent image editing]\n"
        "WHEN: user has an existing image (attached or previously generated) and wants to modify it —\n"
        "  style transfer, colour change, adding/removing elements, background swap, texture change, or any targeted edit.\n"
        "PROMPT — describe ONLY THE CHANGE, not the full scene.\n"
        "  ReferenceLatent preserves the source image's structure, identity, and pose automatically.\n"
        '  GOOD: "Change the jacket to bright red leather with visible stitching"\n'
        '  GOOD: "Replace the background with a foggy pine forest at dusk, keep the subject unchanged"\n'
        '  GOOD: "Render the entire image as an oil painting with thick impasto brushstrokes and visible canvas texture"\n'
        '  GOOD: "Add a glowing neon sign reading OPEN on the wall behind the subject"\n'
        '  GOOD: "Make it look like a vintage 1970s Kodachrome photograph — faded colours, slight grain"\n'
        "  BAD: re-describing the unchanged parts of the original scene.\n"
        "  CFG: default 5 (balanced). Use 7–9 for aggressive/creative edits; use 2–3 for subtle, blended adjustments.\n"
        "  STEPS: 20 default. Raise to 30 for fine detail; lower to 12–15 for quick previews.\n"
        "  Source image: auto-read from the user's attachment or last visible image — never ask for a filename."
    ),
    "video_wan2_2_14B_i2v_subgraphed": (
        "[video_wan2_2_14B_i2v_subgraphed — WAN 2.2 I2V 14B FP8 + LightX2V 4-step, 720×720 @ 16 fps]\n"
        "WHEN: user wants to animate a still image into a short video clip with realistic, physics-accurate motion.\n"
        "  Requires a source image (attached or last generated). Do NOT use for text-only video requests.\n"
        "PROMPT — describe MOTION ONLY. The image defines all visual appearance; the prompt drives what moves and how.\n"
        "  Structure:\n"
        "    1. PRIMARY SUBJECT MOTION\n"
        '         "The hawk spreads its wings and lifts off the branch, gaining altitude"\n'
        '         "The waterfall surges powerfully and crashes into the pool below, spray misting upward"\n'
        '         "The woman slowly turns her head toward the camera, a faint smile crossing her face"\n'
        '         "The campfire flickers and pulses, embers drifting upward"\n'
        "    2. CAMERA MOTION (optional but effective)\n"
        '         "Camera slowly dollies forward" | "Smooth slow pan from left to right"\n'
        '         "Gentle handheld shake" | "Camera tilts upward following the subject"\n'
        '         "Static locked-off shot" | "Slow push-in on the subject\'s face"\n'
        "    3. SECONDARY / ENVIRONMENTAL MOTION\n"
        '         "Leaves rustle gently in the wind" | "Ripples spread across the water surface"\n'
        '         "Hair billows softly" | "Smoke drifts upward and disperses" | "Grass sways in the breeze"\n'
        '         "Clouds drift slowly across the sky" | "Dust motes float in a shaft of light"\n'
        "    4. TEMPORAL PACING\n"
        '         Use: "slowly", "gradually", "suddenly", "rhythmically", "in gentle slow motion"\n'
        '  FULL EXAMPLE: "The fox slowly raises its head and looks directly at the camera. Its ears perk '
        "forward attentively. The surrounding foliage rustles gently in a soft breeze. "
        'Camera holds steady with a subtle breathing motion."\n'
        "  AVOID: describing colours, style, lighting, or visual content already in the image.\n"
        "  AVOID: multiple scene cuts, transitions, or complex multi-step choreography — WAN excels at single continuous shots.\n"
        "  NEGATIVE PROMPT: keep the default Chinese negative — it is fine-tuned for WAN 2.2 motion artifacts.\n"
        "  LENGTH: 81 frames ≈ 5 s (default). Use num_frames=121 for ~7.5 s, 161 for ~10 s."
    ),
}


def get_system_prompt(registry: dict) -> str:
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%H:%M")

    gen_wfs: list[str] = []  # text → image/video, no required input image
    img_edit_wfs: list[str] = []  # requires/accepts image, outputs image
    i2v_wfs: list[str] = []  # requires image, outputs video

    for wf_name, entry in registry.items():
        sc = entry["sidecar"]
        params = list(sc.get("parameters", {}).values())
        has_img_src = any(p.get("image_source") for p in params)
        img_required = any(p.get("image_source") and p.get("required") for p in params)
        is_video = sc.get("output_type") == "video"
        if img_required and is_video:
            i2v_wfs.append(wf_name)
        elif has_img_src and not is_video:
            img_edit_wfs.append(wf_name)
        else:
            gen_wfs.append(wf_name)

    routing = (
        "TOOL ROUTING:\n"
        f"  • No input image + create image from text → {', '.join(gen_wfs) or 'generate_image'}\n"
        f"  • Have an image + edit / transform / restyle it → {', '.join(img_edit_wfs) or 'none'}\n"
        f"  • Have an image + animate it into video → {', '.join(i2v_wfs) or 'none'}\n"
        "  • Never ask the user for an image filename or URL — auto-read from attachment or last generated image.\n"
        "  • web_search: use for any current events, recent news, prices, or time-sensitive facts.\n"
    )

    all_wfs = gen_wfs + img_edit_wfs + i2v_wfs
    tips = "\n\n".join(_WORKFLOW_TIPS[wf] for wf in all_wfs if wf in _WORKFLOW_TIPS)
    tips_block = f"\n\nPER-TOOL PROMPT GUIDANCE:\n{tips}" if tips else ""

    research = """\
    DEEP RESEARCH WORKFLOW:
    For any question requiring up-to-date, comprehensive, or factual answers:
    1. SEARCH PHASE — call web_search with 3-6 queries simultaneously, each attacking
        a different angle (e.g. one factual, one comparative, one recent-news).
        You may emit multiple web_search tool calls in a single response; they run in parallel.
    2. ASSESS PHASE — read the returned snippets. Identify the 5-15 most promising URLs.
    3. FETCH PHASE — call fetch_page on those URLs simultaneously (all in one response).
        Prefer authoritative sources: official docs, Wikipedia, major news outlets, academic pages.
        Skip URLs that look like forums, ads, or paywalled content.
    4. SYNTHESISE — AFTER fetching, write a comprehensive answer citing the sources you read.
        Each fetched page is prefixed with "Source: <url>". Use these URLs as citations.

    WORKFLOW RULES (follow strictly):
    - After web_search results ONLY (no fetched pages yet) → you MUST call fetch_page on URLs from the search results. Do NOT answer from snippets alone.
    - After fetch_page results → your NEXT response MUST be a text answer. Do NOT call any more tools.
    - If unsure whether to answer or fetch: fetch more. Better to read 3 good pages than answer from snippets.

    CONTEXT BUDGET:
    Each fetch_page result is capped at ~6000 characters. You can fetch up to ~20 pages
    before your context fills up. Prioritise quality over quantity.
    Results may be excerpts rather than full pages. Do not assume missing details are absent from the source.
    If a page returns an error or irrelevant content, discard it and move on.

    WHEN NOT TO USE DEEP RESEARCH:
    - Simple factual questions you know with confidence — answer directly.
    - Image/video generation requests — use the generation tools, not search unless user asks to.
    - Personal/hypothetical questions — no search needed.
    """

    return (
        f"You are a capable personal AI assistant running locally. "
        f"Today is {date_str} and the current time is {time_str}. "
        f"Always reason with this date in mind — when researching, "
        f"prioritize recent information and consider whether sources may be outdated. "
        f"If asked about current events, news, prices, or anything time-sensitive, "
        f"use web_search and interpret results relative to today's date.\n\n"
        f"{routing}"
        f"{tips_block}\n\n"
        f"{research}"
        f"Be concise but thorough."
    )


# ── Tool implementations ──────────────────────────────────────────────────────

_FETCH_SEMAPHORE = asyncio.Semaphore(8)
_FETCH_TIMEOUT = httpx.Timeout(connect=8.0, read=20.0, write=5.0, pool=5.0)
_FETCH_MAX_CHARS = 6_000

_PARALLEL_TOOLS = {"web_search", "fetch_page"}
_SESSION_RESEARCH_BUDGET = 80_000  # ~20k tokens of tool results per turn


def _search_provider() -> str:
    provider = _cfg.get("search", "provider", fallback="duckduckgo").strip().lower()
    if provider not in {"duckduckgo", "parallel", "disabled"}:
        raise ValueError(f"Unknown search provider: {provider}")
    return provider


def _ddgs_search(query: str, max_results: int) -> list[dict]:
    """Synchronous DuckDuckGo search — run via asyncio.to_thread."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


async def exec_web_search(
    query: str,
    max_results: int = 10,
    *,
    session_id: str = "",
    model: str = "",
    cancelled: asyncio.Event | None = None,
) -> str:
    """
    Search with the configured provider and return normalized JSON results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        JSON string of results or an error string.
    """
    try:
        provider = _search_provider()
        if provider == "disabled":
            return "Search is disabled in lanGen.ini."
        if not isinstance(query, str) or not query.strip():
            return "Search error: query must not be empty."
        max_results = max(1, min(10, int(max_results)))
        if provider == "parallel":
            import parallel_search

            results = await parallel_search.search(
                query,
                max_results,
                session_id=session_id,
                model=model,
                cancelled=cancelled,
            )
        else:
            raw = await asyncio.to_thread(_ddgs_search, query, max_results)
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
        return (
            json.dumps(results, ensure_ascii=False, indent=2)
            if results
            else "No results found."
        )
    except Exception as e:
        return f"Search error: {e}"


async def fetch_page(
    url: str,
    http: httpx.AsyncClient,
    *,
    session_id: str = "",
    model: str = "",
    cancelled: asyncio.Event | None = None,
) -> str:
    """
    Fetch a URL and extract its main text content.

    Args:
        url: The URL to fetch (must be http/https).
        http: Shared async HTTP client.

    Returns:
        Extracted plain text capped at _FETCH_MAX_CHARS, or an error string.
    """
    try:
        provider = _search_provider()
        if provider == "disabled":
            return "Page fetching is disabled in lanGen.ini."
        if provider == "parallel":
            import parallel_search

            async with _FETCH_SEMAPHORE:
                parallel_text = await parallel_search.fetch(
                    url, session_id=session_id, model=model, cancelled=cancelled
                )
            if len(parallel_text) > _FETCH_MAX_CHARS:
                parallel_text = (
                    parallel_text[:_FETCH_MAX_CHARS] + "\n\n[Content truncated]"
                )
            return f"Source: {url}\n{'=' * 40}\n{parallel_text}"
    except Exception as e:
        return f"Error: {e}"

    async with _FETCH_SEMAPHORE:
        try:
            resp = await http.get(
                url,
                timeout=_FETCH_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text" not in content_type and "html" not in content_type:
                return f"Skipped: non-text content ({content_type})"
            html = resp.text
        except httpx.TimeoutException:
            return "Error: request timed out."
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error: {e}"

    # Primary: trafilatura (boilerplate/nav/ads stripping)
    try:
        text = trafilatura.extract(
            html, include_comments=False, include_tables=True, no_fallback=False
        )
        if text and len(text) > 200:
            header = f"Source: {url}\n{'=' * 40}\n"
            if len(text) > _FETCH_MAX_CHARS:
                text = text[:_FETCH_MAX_CHARS] + "\n\n[Content truncated]"
            return header + text
    except Exception:
        pass

    # Fallback: html2text
    try:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        text = h.handle(html).strip()
        if text:
            header = f"Source: {url}\n{'=' * 40}\n"
            if len(text) > _FETCH_MAX_CHARS:
                text = text[:_FETCH_MAX_CHARS] + "\n\n[Content truncated]"
            return header + text
    except Exception:
        pass

    # Last resort: lxml text extraction
    try:
        from lxml import etree

        tree = etree.fromstring(html.encode(), parser=etree.HTMLParser())
        text = " ".join(tree.itertext()).strip()
        if text:
            header = f"Source: {url}\n{'=' * 40}\n"
            if len(text) > _FETCH_MAX_CHARS:
                text = text[:_FETCH_MAX_CHARS] + "\n\n[Content truncated]"
            return header + text
    except Exception as e:
        return f"Error: content extraction failed ({e})"

    return "Error: could not extract any content."


# ── Agentic loop ──────────────────────────────────────────────────────────────


async def _parse_sse_response(resp: httpx.Response) -> dict:
    """Parse SSE stream from llama.cpp and return accumulated assistant message."""
    msg_content = ""
    msg_reasoning = ""
    msg_tool_calls = None
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            # Newer llama.cpp builds expose reasoning in a separate field
            if delta.get("reasoning_content"):
                msg_reasoning += delta["reasoning_content"]
            if "content" in delta and delta["content"]:
                msg_content += delta["content"]
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if msg_tool_calls is None:
                    msg_tool_calls = {}
                if idx not in msg_tool_calls:
                    msg_tool_calls[idx] = {
                        "id": tc.get("id", f"tool_{idx}"),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                if "function" in tc:
                    fn = tc["function"]
                    if fn.get("name"):
                        msg_tool_calls[idx]["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        msg_tool_calls[idx]["function"]["arguments"] += fn["arguments"]

    # Qwen3 / DeepSeek-style inline <think>...</think> in content
    if not msg_reasoning and "<think>" in msg_content:
        start = msg_content.find("<think>")
        end = msg_content.find("</think>")
        if start != -1 and end != -1:
            msg_reasoning = msg_content[start + 7 : end].strip()
            msg_content = msg_content[end + 8 :].strip()

    msg: dict[str, Any] = {"role": "assistant"}
    if msg_content:
        msg["content"] = msg_content
    if msg_tool_calls:
        msg["tool_calls"] = list(msg_tool_calls.values())
    if msg_reasoning:
        msg["reasoning"] = msg_reasoning
    return msg


@dataclass
class ComfyResult:
    progress_events: list = field(default_factory=list)
    result_text: str | None = None
    file_url: str | None = None
    output_type: str = "image"


async def _execute_comfy_workflow(
    name: str,
    workflow: dict,
    sc: dict,
    http: httpx.AsyncClient,
    client_id: str,
) -> ComfyResult:
    """
    Execute a ComfyUI workflow from the registry.

    Returns a ComfyResult with progress events, result text, file URL, and output type.
    """
    result = ComfyResult()
    output_type = sc.get("output_type", "image")

    # Health check
    try:
        await http.get(f"{IMAGE_SERVER}/")
    except httpx.RequestError as e:
        result.result_text = f"ComfyUI not reachable: {e}"
        return result

    # Submit workflow
    resp = await http.post(
        f"{IMAGE_SERVER}/prompt",
        json={"prompt": workflow, "client_id": client_id},
    )
    data = resp.json()
    if "error" in data or data.get("node_errors"):
        errs = data.get("node_errors") or data.get("error")
        result.result_text = f"ComfyUI error: {errs}"
        return result
    prompt_id: str = data["prompt_id"]

    # WebSocket progress tracking
    output_file_meta = None
    ws_url = (
        IMAGE_SERVER.replace("http://", "ws://").replace("https://", "wss://")
        + f"/ws?clientId={client_id}"
    )

    log.debug("WS url=%s", ws_url)
    try:
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                wmsg = json.loads(raw)
                wt = wmsg.get("type", "")
                wd = wmsg.get("data", {})
                log.debug(
                    "WS msg type=%r pid=%r node=%r",
                    wt,
                    wd.get("prompt_id", ""),
                    wd.get("node", ""),
                )
                if wd.get("prompt_id") and wd["prompt_id"] != prompt_id:
                    continue
                if wt == "progress":
                    result.progress_events.append(
                        json.dumps(
                            {
                                "type": "progress",
                                "tool": name,
                                "value": wd.get("value", 0),
                                "max": wd.get("max", 1),
                                "node": wd.get("node", ""),
                            }
                        )
                    )
                elif wt == "executed":
                    node_id_ex = wd.get("node", "")
                    node_cls = workflow.get(node_id_ex, {}).get("class_type", "")
                    log.debug(
                        "executed node=%r class=%r skip=%s",
                        node_id_ex,
                        node_cls,
                        node_cls in _SKIP_NODE_CLASSES,
                    )
                    if node_cls in _SKIP_NODE_CLASSES:
                        continue
                    out = wd.get("output", {})
                    for key in ("images", "gifs", "video", "audio"):
                        if key in out and out[key]:
                            output_file_meta = out[key][0]
                            log.info(
                                "captured output_file_meta=%s key=%r",
                                output_file_meta,
                                key,
                            )
                            if key in ("gifs", "video"):
                                output_type = "video"
                            elif key == "audio":
                                output_type = "audio"
                            break
                elif wt == "execution_success":
                    log.info(
                        "execution_success → breaking WS loop, output_file_meta=%s",
                        output_file_meta,
                    )
                    break
                elif wt == "execution_error":
                    result.result_text = f"ComfyUI execution error: {wd.get('exception_message', 'unknown')}"
                    log.error("execution_error: %s", result.result_text)
                    break
    except httpx.RequestError as e:
        log.warning("WebSocket error: %s; falling back to polling", e)
        # Polling fallback
        for _i in range(600):
            await asyncio.sleep(0.5)
            hr = await http.get(f"{IMAGE_SERVER}/history/{prompt_id}")
            history_resp = hr.json()
            if prompt_id in history_resp:
                outputs = history_resp[prompt_id].get("outputs", {})
                for _nid, node_out in outputs.items():
                    node_cls = workflow.get(_nid, {}).get("class_type", "")
                    if node_cls in _SKIP_NODE_CLASSES:
                        continue
                    for key in ("images", "gifs", "video", "audio"):
                        if key in node_out and node_out[key]:
                            output_file_meta = node_out[key][0]
                            if key in ("gifs", "video"):
                                output_type = "video"
                            elif key == "audio":
                                output_type = "audio"
                            break
                    if output_file_meta:
                        break
                if output_file_meta:
                    break
        else:
            result.result_text = "ComfyUI timed out."

    result.output_type = output_type

    # Download the output file after the loop
    log.info(
        "post-WS: output_file_meta=%s result_text=%r",
        output_file_meta,
        result.result_text,
    )
    if output_file_meta and not result.result_text:
        file_url, output_type = await _download_comfy_output(
            output_file_meta, output_type, http
        )
        result.file_url = file_url
        result.output_type = output_type

    if result.file_url is None and not result.result_text:
        result.result_text = "ComfyUI finished but no output file found."

    return result


async def _download_comfy_output(
    output_file_meta: dict,
    output_type: str,
    http: httpx.AsyncClient,
) -> tuple[str | None, str]:
    """Download a ComfyUI output file and save it to the generated directory."""
    try:
        view_resp = await http.get(
            f"{IMAGE_SERVER}/view",
            params={
                "filename": output_file_meta["filename"],
                "subfolder": output_file_meta.get("subfolder", ""),
                "type": output_file_meta.get("type", "output"),
            },
        )
        view_resp.raise_for_status()
        raw_bytes = view_resp.content
        if output_type == "video":
            ext = ".mp4"
        elif output_type == "audio":
            fname_out = output_file_meta["filename"]
            ext = ".flac" if fname_out.endswith(".flac") else ".wav"
        else:
            ext = ".jpg"
        out_filename = f"gen_{uuid.uuid4().hex[:12]}{ext}"
        out_path = _gen_dir / out_filename
        if output_type == "image":
            img: PILImage.Image = PILImage.open(BytesIO(raw_bytes))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            if max(img.size) > 1280:
                ratio = 1280 / max(img.size)
                img = img.resize(
                    (
                        int(img.width * ratio),
                        int(img.height * ratio),
                    ),
                    PILImage.Resampling.LANCZOS,
                )
            img.save(out_path, format="JPEG", quality=88)
        else:
            out_path.write_bytes(raw_bytes)
        file_url = f"/static/generated/{out_filename}"
        log.info("saved → %s (%d bytes)", file_url, out_path.stat().st_size)
        return file_url, output_type
    except Exception as dl_err:
        log.error("download FAILED: %s", dl_err)
        return None, output_type


async def agent_stream(
    messages: list[dict[str, Any]],
    attachment: Any = None,
    registry: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    last_image: str = "",
    last_image_url: str = "",
    models_dir: Path | None = None,
    cancelled: asyncio.Event | None = None,
    session_id: str = "",
    model: str = "",
) -> AsyncGenerator[str, None]:
    """
    Runs the tool-calling loop, yielding newline-delimited JSON events:

      {"type": "status",  "tool": "web_search",    "args": {...}}
      {"type": "status",  "tool": "generate_image", "args": {...}}
      {"type": "image",   "url": "/static/generated/...", "prompt": "..."}
      {"type": "text",    "content": "final reply"}
      {"type": "error",   "msg": "..."}

    Client disconnection (AbortController / browser tab close) cancels the
    in-flight HTTP request to llama.cpp immediately.
    """
    history = list(messages)
    search_session_id = session_id or str(uuid.uuid4())
    allowed_tools = {tool["function"]["name"] for tool in tools or []}

    async with httpx.AsyncClient(timeout=720) as client:
        has_fetch_results = False
        search_rounds = 0
        sources: list[str] = []
        for _iteration in range(MAX_ITERATIONS):
            if cancelled and cancelled.is_set():
                yield json.dumps({"type": "error", "msg": "Cancelled by user."})
                return

            # After fetch_page results exist, add a steering message to force synthesis
            if has_fetch_results:
                history.append(
                    {
                        "role": "system",
                        "content": "You have fetched page content. Write your final answer now. Do not call any more tools.",
                    }
                )

            # After N search rounds without fetching, force synthesis
            if search_rounds >= 2 and not has_fetch_results:
                history.append(
                    {
                        "role": "system",
                        "content": (
                            "You have already searched multiple times. STOP searching. "
                            "Write your final answer now using the search results you already have. "
                            "Do NOT call any more tools."
                        ),
                    }
                )
                search_rounds += 1  # count the steering round too
                log.info(
                    "steering: forcing synthesis after %d search rounds (no fetch)",
                    search_rounds - 1,
                )

            req_body: dict = {
                "messages": history,
                "tools": tools,
                "tool_choice": "auto",
                "stream": True,
            }
            if model:
                req_body["model"] = model

            try:
                async with client.stream(
                    "POST", f"{LLAMA_SERVER}/v1/chat/completions", json=req_body
                ) as resp:
                    resp.raise_for_status()
                    msg = await _parse_sse_response(resp)

            except httpx.ConnectError:
                yield json.dumps(
                    {
                        "type": "error",
                        "msg": f"Cannot reach llama server at {LLAMA_SERVER}. Is it running?",
                    }
                )
                return
            except httpx.RemoteProtocolError:
                # Client disconnected — this is expected on stop
                yield json.dumps({"type": "error", "msg": "Cancelled by user."})
                return
            except httpx.ReadTimeout:
                yield json.dumps({"type": "error", "msg": "LLM request timed out."})
                return
            except (httpx.HTTPError, OSError, ValueError) as e:
                log.error("LLM request failed: %s", e)
                yield json.dumps({"type": "error", "msg": f"LLM request failed: {e}"})
                return

            # Yield reasoning before anything else; strip from msg so it
            # never lands in the LLM history (models reject unknown fields)
            reasoning = msg.pop("reasoning", None)
            if reasoning:
                yield json.dumps({"type": "thinking", "content": reasoning})

            calls = msg.get("tool_calls")

            # ── Model wants to call tools ─────────────────────────────────
            if calls:
                log.info(
                    "iteration=%d tool_calls=%d names=[%s]",
                    _iteration,
                    len(calls),
                    ", ".join(tc["function"]["name"] for tc in calls),
                )
                # Yield any text the model sent alongside tool calls (e.g. "Let me search...")
                partial_text = msg.get("content", "").strip()
                if partial_text:
                    yield json.dumps({"type": "text", "content": partial_text})
                history.append(msg)

                # ── Classify: parallel vs serial ─────────────────────────────
                parallel_calls = [
                    tc for tc in calls if tc["function"]["name"] in _PARALLEL_TOOLS
                ]
                serial_calls = [
                    tc for tc in calls if tc["function"]["name"] not in _PARALLEL_TOOLS
                ]

                # ── Pass 1: run parallel tools concurrently ──────────────────
                for tc in parallel_calls:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    if name == "web_search":
                        search_rounds += 1
                    log.info("tool call (parallel): name=%r args=%r", name, args)
                    yield json.dumps({"type": "status", "tool": name, "args": args})

                async def _run_parallel_tool(
                    tc: dict, fetch_http: httpx.AsyncClient
                ) -> tuple[str, str]:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    if name not in allowed_tools:
                        return tc["id"], f"Tool is not enabled: {name}"
                    if name == "web_search":
                        result = await exec_web_search(
                            args.get("query", ""),
                            args.get("max_results", 10),
                            session_id=search_session_id,
                            model=model,
                            cancelled=cancelled,
                        )
                    elif name == "fetch_page":
                        result = await fetch_page(
                            args.get("url", ""),
                            fetch_http,
                            session_id=search_session_id,
                            model=model,
                            cancelled=cancelled,
                        )
                    else:
                        result = f"Unknown parallel tool: {name}"
                    return tc["id"], result

                async with httpx.AsyncClient(timeout=30) as fetch_http:
                    outcomes = await asyncio.gather(
                        *[_run_parallel_tool(tc, fetch_http) for tc in parallel_calls],
                        return_exceptions=True,
                    )

                research_chars_used = 0
                for tc, outcome in zip(parallel_calls, outcomes, strict=False):
                    if isinstance(outcome, (tuple, list)) and len(outcome) == 2:
                        tool_call_id, result_text = outcome
                    else:
                        result_text = f"Tool error: {outcome}"
                        tool_call_id = tc["id"]
                    if (
                        research_chars_used + len(result_text)
                        > _SESSION_RESEARCH_BUDGET
                    ):
                        result_text = "[Budget exceeded — skipping this result to preserve context.]"
                    else:
                        research_chars_used += len(result_text)
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result_text,
                        }
                    )
                    if "Source:" in result_text:
                        has_fetch_results = True
                    # Collect URLs for sources display
                    name = tc["function"]["name"]
                    if name == "web_search":
                        try:
                            results = json.loads(result_text)
                            if isinstance(results, list):
                                for r in results:
                                    u = r.get("url", "")
                                    if u and u not in sources:
                                        sources.append(u)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    elif name == "fetch_page":
                        args = json.loads(tc["function"]["arguments"] or "{}")
                        u = args.get("url", "")
                        if u and u not in sources:
                            sources.append(u)

                # ── Pass 2: run serial tools (ComfyUI workflows, generate_image) ──
                for tc in serial_calls:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    log.info("tool call (serial): name=%r args=%r", name, args)

                    yield json.dumps({"type": "status", "tool": name, "args": args})

                    has_visible_image = False

                    if registry is not None and name in registry:
                        # Dynamic workflow execution
                        sc = registry[name]["sidecar"]

                        # Build workflow from registry entry
                        workflow = json.loads(json.dumps(registry[name]["workflow"]))
                        seed = random.randint(0, 2**32 - 1)

                        # Patch parameters
                        for param_name, p in sc.get("parameters", {}).items():
                            node_id = str(p.get("node", ""))
                            fld = p.get("field", "")
                            value = args.get(param_name, p.get("default"))
                            log.debug(
                                "patch param=%r node=%r field=%r value=%r",
                                param_name,
                                node_id,
                                fld,
                                value,
                            )
                            if value is None:
                                continue
                            if node_id in workflow and fld:
                                workflow[node_id]["inputs"][fld] = value
                                log.debug(
                                    "  patched → %r",
                                    workflow[node_id]["inputs"].get(fld, ""),
                                )
                            else:
                                log.debug(
                                    "  SKIP: node_id_in_workflow=%s, field_empty=%s",
                                    node_id in workflow,
                                    not fld,
                                )

                        # Patch seed into sampler nodes (model can override via "seed" param)
                        model_seed = args.get("seed")
                        if model_seed is not None:
                            seed = int(model_seed)
                            log.info("using model-provided seed=%d", seed)
                        for _nid, node in workflow.items():
                            ct = node.get("class_type", "")
                            if ct in (
                                "KSampler",
                                "KSamplerAdvanced",
                                "SamplerCustomAdvanced",
                                "WanVideoSampler",
                                "LTXVSampler",
                                "WanVideoSamplerGGUF",
                                "LTXVSamplerGGUF",
                            ):
                                node["inputs"]["seed"] = seed

                        # Disable ComfyUI's own SaveImage/VHS save to avoid double-writes
                        if sc.get("disable_comfy_save"):
                            for node in workflow.values():
                                ct = node.get("class_type", "")
                                if ct == "SaveImage":
                                    node["class_type"] = "PreviewImage"
                                elif ct == "VHS_VideoCombine":
                                    node["inputs"]["save_output"] = False

                        # Resolve image source for img2img/i2v workflows.
                        # Falls back through: LLM url_arg → attachment → history → last_image_url → last_image.
                        img_resolve_failed = False
                        for pname, p in sc.get("parameters", {}).items():
                            if not p.get("image_source"):
                                continue
                            node_id = str(p.get("node", ""))
                            fld = p.get("field", "")
                            if not (node_id in workflow and fld):
                                continue
                            url_arg = args.get(pname)
                            async with httpx.AsyncClient(timeout=30) as up_http:
                                comfy_fname = await comfy.resolve_image_for_comfy(
                                    up_http,
                                    url_arg,
                                    attachment,
                                    last_image,
                                    history,
                                    last_image_url=last_image_url,
                                )
                            if comfy_fname:
                                workflow[node_id]["inputs"][fld] = comfy_fname
                                log.info("img2img image resolved → %r", comfy_fname)
                            else:
                                log.warning(
                                    "no image source resolved for param %r", pname
                                )
                                img_resolve_failed = True

                        if img_resolve_failed:
                            yield json.dumps(
                                {
                                    "type": "error",
                                    "msg": "No image found to edit. Please generate an image first or attach one.",
                                }
                            )
                            result_text = ""
                            file_url = None
                            has_visible_image = False
                            history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": "Error: no image source available.",
                                }
                            )
                            continue

                        # Abort if any required models are missing
                        missing_models = False
                        async for dl_evt in comfy.ensure_models(sc, models_dir):
                            yield dl_evt
                            missing_models = True
                        if missing_models:
                            return

                        client_id = str(uuid.uuid4())

                        try:
                            async with httpx.AsyncClient(timeout=600) as http:
                                cr = await _execute_comfy_workflow(
                                    name, workflow, sc, http, client_id
                                )
                                # Yield progress events
                                for pe in cr.progress_events:
                                    yield json.dumps(json.loads(pe))
                                if cr.file_url:
                                    yield json.dumps(
                                        {
                                            "type": cr.output_type,
                                            "url": cr.file_url,
                                            "prompt": args.get("prompt", ""),
                                        }
                                    )
                                    result_text = f"{cr.output_type.capitalize()} generated: {args.get('prompt', '')}"
                                    has_visible_image = cr.output_type == "image"
                                    # Make generated images available for follow-up img2img editing
                                    if cr.output_type == "image":
                                        local_path = _gen_dir / Path(cr.file_url).name
                                        if local_path.exists():
                                            try:
                                                b64 = base64.b64encode(
                                                    local_path.read_bytes()
                                                ).decode()
                                                async with httpx.AsyncClient(
                                                    timeout=30
                                                ) as up_http:
                                                    last_image = await comfy.upload_image_to_comfy(
                                                        up_http,
                                                        b64,
                                                        local_path.name,
                                                        last_image,
                                                    )
                                                log.info(
                                                    "registered generated image in ComfyUI input for future editing"
                                                )
                                            except Exception as up_err:
                                                log.warning(
                                                    "could not upload generated image to ComfyUI: %s",
                                                    up_err,
                                                )
                                else:
                                    result_text = (
                                        cr.result_text
                                        or "ComfyUI workflow failed with no error."
                                    )
                                    has_visible_image = False
                                file_url = cr.file_url

                        except httpx.HTTPError as e:
                            log.error(
                                "EXCEPTION in '%s': %s: %s", name, type(e).__name__, e
                            )
                            result_text = f"Workflow '{name}' failed: {e}"
                            file_url = None
                            has_visible_image = False

                    elif name == "generate_image":
                        # Legacy shim for backwards compat
                        result_text, file_url = await exec_generate_image(
                            args.get("prompt", ""),
                            args.get("negative_prompt", ""),
                            int(args.get("steps", 22)),
                            int(args.get("width", 512)),
                            int(args.get("height", 512)),
                            last_image,
                        )
                        if file_url:
                            log.debug("yielding image event, url=%s", file_url)
                            yield json.dumps(
                                {
                                    "type": "image",
                                    "url": file_url,
                                    "prompt": args.get("prompt", ""),
                                }
                            )
                            has_visible_image = True

                    else:
                        result_text = f"Unknown tool: {name}"
                        file_url = None
                        has_visible_image = False

                    # Build tool result — include generated image as base64 so the
                    # LLM can see what it produced on the next iteration.
                    tool_content: str | list[dict[str, Any]] = result_text
                    if has_visible_image and file_url is not None:
                        local_path = _gen_dir / Path(file_url).name
                        if local_path.exists():
                            try:
                                resized = _resize_image(
                                    base64.b64encode(local_path.read_bytes()).decode()
                                )
                                tool_content = [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{resized}",
                                            "detail": "low",
                                        },
                                    },
                                    {"type": "text", "text": result_text},
                                ]
                            except Exception:
                                pass

                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_content,
                        }
                    )

            # Log tool results so we can see what the model receives next
            tool_msgs = [h for h in history if h.get("role") == "tool"]
            log.info(
                "iteration=%d tool_results=%d last_result_len=%d",
                _iteration,
                len(tool_msgs),
                len(tool_msgs[-1].get("content", "")) if tool_msgs else 0,
            )

            # ── Save checkpoint after tool results ─────────────────────────
            # Strip system messages (prompt + steering) — they are regenerated
            # fresh on every request and must not be re-sent by the frontend.
            if session_id:
                save_session(
                    session_id, [m for m in history if m.get("role") != "system"]
                )

            # ── Check cancellation between tool calls ─────────────────────
            if cancelled and cancelled.is_set():
                yield json.dumps({"type": "error", "msg": "Cancelled by user."})
                return

            # ── Final text response ───────────────────────────────────────
            else:
                content = msg.get("content") or ""
                log.info(
                    "iteration=%d text_response len=%d preview=%r",
                    _iteration,
                    len(content),
                    content[:100],
                )
                if not content.strip():
                    log.warning(
                        "iteration=%d empty response — continuing loop", _iteration
                    )
                    continue
                if sources:
                    yield json.dumps({"type": "sources", "urls": sources})
                yield json.dumps({"type": "text", "content": content})
                # ── Persist final assistant turn ───────────────────────────
                final_msg: dict[str, Any] = {"role": "assistant", "content": content}
                if sources:
                    final_msg["sources"] = sources
                history.append(final_msg)
                if session_id:
                    save_session(session_id, history)
                return

        yield json.dumps(
            {
                "type": "error",
                "msg": f"Reached max tool iterations ({MAX_ITERATIONS}).",
            }
        )


# ── Static assets ─────────────────────────────────────────────────────────────

_SW_JS = """\
const CACHE = "local-agent-v1";
const SHELL = ["/", "/manifest.json"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.pathname === "/chat" || url.pathname.startsWith("/api/")) return;
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
"""

_MANIFEST = {
    "name": "Local Agent",
    "short_name": "Agent",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#07080d",
    "theme_color": "#07080d",
    "icons": [{"src": "/static/icon.png", "sizes": "192x192", "type": "image/png"}],
}

# ── llama-server spawning ─────────────────────────────────────────────────────


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def _wait_for_llama(url: str, timeout: int = 60) -> bool:
    """Poll url/health until 200 or timeout (seconds)."""
    async with httpx.AsyncClient(timeout=3) as client:
        for _ in range(timeout):
            try:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
    return False


def _stream(pipe: IO[str], fh: IO[str]) -> None:
    """Read a subprocess pipe line-by-line, print to stdout and write to file handle."""
    try:
        for line in pipe:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
    except ValueError:
        pass


def _drain(pipe: IO[str], fh: IO[str]) -> None:
    """Read a subprocess pipe and write to file handle only (no stdout)."""
    try:
        for line in pipe:
            fh.write(line)
            fh.flush()
    except ValueError:
        pass


async def _wait_for_comfyui(url: str, timeout: int = 60) -> bool:
    """Poll url/object_info until 200 or timeout (seconds)."""
    async with httpx.AsyncClient(timeout=3) as client:
        for _ in range(timeout):
            try:
                r = await client.get(f"{url}/object_info")
                if r.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
    return False


def _spawn_llama_server() -> _ServerState | None:
    """
    Spawn llama-server using paths from lanGen.ini [llama] section.
    Returns a _ServerState with the Popen handle, or None if already running or config absent.
    """
    if "llama" not in _cfg:
        return None
    if _cfg["llama"].get("managed", "false").lower() != "true":
        return None

    port = int(_cfg["llama"].get("port", "8080"))
    if _port_open(port):
        log.info("llama-server already listening on port %d — not spawning", port)
        return None

    sec = _cfg["llama"]
    binary = sec.get("binary", "")
    model = sec.get("model", "")
    mmproj = sec.get("mmproj", "")
    gpu_layers = sec.get("gpu_layers", "0")
    ctx_size = sec.get("ctx_size", "8000")
    n_parallel = sec.get("n_parallel", "1")
    flash_attn = sec.get("flash_attn", "true")
    cache_type_k = sec.get("cache_type_k", "q8_0")
    cache_type_v = sec.get("cache_type_v", "q8_0")
    temp = sec.get("temp", "0.6")
    top_p = sec.get("top_p", "0.95")
    top_k = sec.get("top_k", "20")
    min_p = sec.get("min_p", "0.0")
    presence_penalty = sec.get("presence_penalty", "1.0")
    repeat_penalty = sec.get("repeat_penalty", "1.0")
    reasoning_budget = sec.get("reasoning_budget", "-1")
    image_max_tokens = sec.get("image_max_tokens", "1120")

    if not binary or not Path(binary).exists():
        log.warning("llama binary not found at %r — skipping spawn", binary)
        return None
    if not model or not Path(model).exists():
        log.warning("llama model not found at %r — skipping spawn", model)
        return None

    cmd = [
        binary,
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--n-gpu-layers",
        gpu_layers,
        "--ctx-size",
        ctx_size,
        "-np",
        n_parallel,
        "--temp",
        temp,
        "--top-p",
        top_p,
        "--top-k",
        top_k,
        "--min-p",
        min_p,
        "--presence-penalty",
        presence_penalty,
        "--repeat-penalty",
        repeat_penalty,
        "--reasoning-budget",
        reasoning_budget,
        "--log-disable",
    ]
    if flash_attn.lower() == "true":
        cmd += ["-fa", "on"]
    if cache_type_k:
        cmd += ["-ctk", cache_type_k]
    if cache_type_v:
        cmd += ["-ctv", cache_type_v]
    if mmproj and Path(mmproj).exists():
        cmd += ["--mmproj", mmproj, "--image-max-tokens", image_max_tokens]
        log.info("multimodal mode: mmproj=%s", mmproj)

    log.info("spawning llama-server: %s", " ".join(cmd))
    _llama_log_path = ROOT / "logs" / "llama-server.log"
    _llama_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(_llama_log_path, "a")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    threading.Thread(
        target=_stream, args=(proc.stderr, log_fh), daemon=True, name="llama-stderr"
    ).start()
    threading.Thread(
        target=_drain, args=(proc.stdout, log_fh), daemon=True, name="llama-stdout"
    ).start()
    return _ServerState(proc=proc, log_fh=log_fh)


def _spawn_comfyui() -> _ServerState | None:
    """
    Spawn ComfyUI using paths from lanGen.ini [comfyui] section.
    Returns a _ServerState with the Popen handle, or None if already running or config absent.
    """
    if "comfyui" not in _cfg:
        return None
    if _cfg["comfyui"].get("managed", "false").lower() != "true":
        return None
    port = int(_cfg["comfyui"].get("port", "7860"))
    if _port_open(port):
        log.info("ComfyUI already listening on port %d — not spawning", port)
        return None
    sec = _cfg["comfyui"]
    workspace = sec.get("workspace", "")
    python_bin = sec.get("python", "python3")
    if not workspace:
        log.warning("ComfyUI workspace not configured — skipping spawn")
        return None
    main_py = Path(workspace) / "main.py"
    if not main_py.exists():
        log.warning("ComfyUI main.py not found at %r — skipping spawn", main_py)
        return None
    cmd = [
        python_bin,
        str(main_py),
        "--port",
        str(port),
        "--listen",
        "127.0.0.1",
        "--disable-auto-launch",
    ]
    log.info("spawning ComfyUI: %s (workspace=%s)", " ".join(cmd), workspace)
    _comfy_log_path = ROOT / "logs" / "comfyui.log"
    _comfy_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(_comfy_log_path, "a")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    threading.Thread(
        target=_stream, args=(proc.stderr, log_fh), daemon=True, name="comfy-stderr"
    ).start()
    threading.Thread(
        target=_drain, args=(proc.stdout, log_fh), daemon=True, name="comfy-stdout"
    ).start()
    return _ServerState(proc=proc, log_fh=log_fh)


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

ssl_cert, ssl_key = _ensure_ssl()


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:  # pylint: disable=unused-argument
    print(f"  SSL cert → {ssl_cert}")
    print(f"  SSL key  → {ssl_key}")

    # Spawn llama-server if lanGen.ini says managed=true
    llama_state = _spawn_llama_server()
    if llama_state and llama_state.proc is not None:
        print("  Waiting for llama-server to load model (up to 60 s)...")
        ready = await _wait_for_llama(LLAMA_SERVER, timeout=60)
        if ready:
            log.info("llama-server ready at %s", LLAMA_SERVER)
        else:
            log.warning("llama-server did not respond within 60 s — continuing anyway")

    # Fetch active model name from llama server
    model = ""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{LLAMA_SERVER}/v1/models")
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    model = models[0]["id"]
                    log.info("llama model: %s", model)
    except httpx.RequestError as e:
        log.warning("could not fetch model list from llama server: %s", e)

    # Spawn ComfyUI if lanGen.ini says managed=true
    comfy_state = _spawn_comfyui()
    if comfy_state and comfy_state.proc is not None:
        print("  Waiting for ComfyUI to start (up to 60 s)...")
        comfy_ready = await _wait_for_comfyui(IMAGE_SERVER, timeout=60)
        if comfy_ready:
            log.info("ComfyUI ready at %s", IMAGE_SERVER)
        else:
            log.warning("ComfyUI did not respond within 60 s — continuing anyway")

    # Load workflow registry
    registry = load_comfy_workflows()

    # Introspect ComfyUI for model lists and validation
    object_info = None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{IMAGE_SERVER}/object_info")
            if r.status_code == 200:
                object_info = r.json()
                log.info("loaded %d node types", len(object_info))
                comfy.validate_workflow_models(registry, object_info)
    except httpx.RequestError as e:
        log.warning("ComfyUI not reachable at startup: %s", e)

    # Build dynamic TOOLS list
    tools = build_tools_from_workflows(registry, object_info)
    models_dir = comfy.comfy_cfg.models_dir if comfy.comfy_cfg else None
    log.info(
        "registered %d tools: %s",
        len(tools),
        [t["function"]["name"] for t in tools],
    )

    fastapi_app.state.app_state = AppState(
        model=model,
        registry=registry,
        tools=tools,
        models_dir=models_dir,
    )

    # Log VRAM
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{IMAGE_SERVER}/system_stats")
            if r.status_code == 200:
                stats = r.json()
                for dev in stats.get("devices", []):
                    free_gb = dev.get("vram_free", 0) / 1e9
                    total_gb = dev.get("vram_total", 0) / 1e9
                    log.info("%s: %.1f/%.1f GB free", dev["name"], free_gb, total_gb)
    except (httpx.RequestError, KeyError):
        pass

    yield

    # Shutdown: terminate llama-server if we spawned it
    if llama_state and llama_state.proc is not None:
        log.info("shutting down llama-server (pid %d)", llama_state.proc.pid)
        llama_state.proc.terminate()
        try:
            llama_state.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            llama_state.proc.kill()
    if llama_state and llama_state.log_fh is not None:
        llama_state.log_fh.close()

    # Shutdown: terminate ComfyUI if we spawned it
    if comfy_state and comfy_state.proc is not None:
        log.info("shutting down ComfyUI (pid %d)", comfy_state.proc.pid)
        comfy_state.proc.terminate()
        try:
            comfy_state.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            comfy_state.proc.kill()
    if comfy_state and comfy_state.log_fh is not None:
        comfy_state.log_fh.close()


app = FastAPI(title="Personal LAN Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Pydantic models ───────────────────────────────────────────────────────────


class Attachment(BaseModel):
    type: str  # "image" | "pdf"
    data: str  # base64-encoded file content
    filename: str
    mime_type: str


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    system: str = ""
    attachment: Attachment | None = None
    last_image_url: str = ""
    session_id: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream an agent response over SSE for a single chat turn."""
    app_state: AppState = request.app.state.app_state
    system = req.system if req.system else get_system_prompt(app_state.registry)
    messages = list(req.messages)

    if req.attachment and messages:
        last = messages[-1]
        if last.get("role") == "user":
            att = req.attachment
            if att.type == "image":
                resized = _resize_image(att.data)
                # Save attachment to disk for URL reference
                ext = ".jpg" if att.mime_type and "jpg" in att.mime_type else ".png"
                fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(att)}{ext}"
                save_path = _input_dir / fname
                save_path.write_bytes(base64.b64decode(att.data))
                img_url = f"/static/input/{fname}"
                # Update message content to include the image URL
                original_content = last.get("content", "")
                if original_content and f"![]({img_url})" not in original_content:
                    last["content"] = f"![]({img_url})\n\n{original_content}"
                messages[-1] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{resized}"},
                        },
                        {"type": "text", "text": last.get("content", "")},
                    ],
                }
            elif att.type == "pdf":
                pdf_text = _extract_pdf_text(att.data)
                original = last.get("content", "")
                messages[-1] = {
                    "role": "user",
                    "content": (
                        f"[Attached document: {att.filename}]\n\n{pdf_text}\n\n[User message]\n{original}"
                    ),
                }

    full = [{"role": "system", "content": system}] + messages

    async def sse() -> AsyncGenerator[str, None]:
        try:
            async for event in agent_stream(
                full,
                attachment=req.attachment,
                registry=app_state.registry,
                tools=app_state.tools,
                last_image_url=req.last_image_url,
                models_dir=app_state.models_dir,
                session_id=req.session_id,
                model=app_state.model,
            ):
                yield f"data: {event}\n\n"
            yield "data: [DONE]\n\n"
        except httpx.RemoteProtocolError:
            # Client disconnected — stop streaming
            pass
        except GeneratorExit:
            pass

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Background task store (keeps generation alive across tab switches) ─────────


async def _start_task(req: ChatRequest, app_state: AppState) -> str:
    """Start agent generation as a background task. Returns task_id."""
    task_id = uuid.uuid4().hex[:12]
    cancelled = asyncio.Event()
    task: _TaskVal = {
        "status": "starting",
        "text": "",
        "reasoning": "",
        "events": [],
        "done": False,
        "error": None,
        "cancelled": cancelled,
        "session_id": req.session_id,
    }
    async with app_state.task_lock:
        app_state.task_store[task_id] = task

    async def _run() -> None:
        system = req.system if req.system else get_system_prompt(app_state.registry)
        messages = list(req.messages)

        if req.attachment and messages:
            last = messages[-1]
            if last.get("role") == "user":
                att = req.attachment
                if att.type == "image":
                    resized = _resize_image(att.data)
                    messages[-1] = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{resized}"
                                },
                            },
                            {"type": "text", "text": last.get("content", "")},
                        ],
                    }
                elif att.type == "pdf":
                    pdf_text = _extract_pdf_text(att.data)
                    original = last.get("content", "")
                    messages[-1] = {
                        "role": "user",
                        "content": (
                            f"[Attached document: {att.filename}]\n\n{pdf_text}\n\n[User message]\n{original}"
                        ),
                    }

        full = [{"role": "system", "content": system}] + messages

        try:
            async for event_json in agent_stream(
                full,
                attachment=req.attachment,
                registry=app_state.registry,
                tools=app_state.tools,
                last_image_url=req.last_image_url,
                models_dir=app_state.models_dir,
                cancelled=task["cancelled"],
                session_id=req.session_id,
                model=app_state.model,
            ):
                evt = json.loads(event_json)
                if evt.get("type") == "text":
                    task["text"] = evt["content"]
                if evt.get("type") == "thinking":
                    task["reasoning"] = evt.get("content", "")
                if evt.get("type") in ("image", "error", "text"):
                    log.debug(
                        "event: type=%s url=%s text_len=%d",
                        evt["type"],
                        evt.get("url", ""),
                        len(evt.get("content", "")),
                    )
                task["events"].append(event_json)
        except Exception as e:
            log.error("background task failed: %s", e)
            task["error"] = str(e)
        finally:
            task["status"] = "done"
            task["done"] = True

    asyncio.create_task(_run())
    return task_id


@app.post("/api/chat")
async def api_start_chat(req: ChatRequest, request: Request) -> JSONResponse:
    """Start a background agent task and return its task_id."""
    app_state: AppState = request.app.state.app_state
    task_id = await _start_task(req, app_state)
    async with app_state.task_lock:
        session_id = app_state.task_store.get(task_id, {}).get("session_id", "")
    return JSONResponse({"task_id": task_id, "session_id": session_id})


@app.post("/api/chat/cancel/{task_id}")
async def api_cancel_chat(task_id: str, request: Request) -> JSONResponse:
    """Signal cancellation for a running background task."""
    app_state: AppState = request.app.state.app_state
    async with app_state.task_lock:
        task = app_state.task_store.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    task["cancelled"].set()
    return JSONResponse({"ok": True})


@app.get("/api/sessions/active")
async def api_active_sessions(request: Request) -> JSONResponse:
    """List all in-progress background tasks with their session IDs."""
    app_state: AppState = request.app.state.app_state
    active = []
    async with app_state.task_lock:
        for task_id, task in app_state.task_store.items():
            if not task["done"] and not task.get("cancelled", asyncio.Event()).is_set():
                active.append({"session_id": task["session_id"], "task_id": task_id})
    return JSONResponse({"active": active})


@app.get("/api/chat/status/{task_id}")
async def api_chat_status(task_id: str, request: Request) -> JSONResponse:
    """Return current status and accumulated events for a background task."""
    app_state: AppState = request.app.state.app_state
    async with app_state.task_lock:
        task = app_state.task_store.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(
        {
            "status": task["status"],
            "text": task["text"],
            "reasoning": task.get("reasoning", ""),
            "events": task["events"],
            "done": task["done"],
            "error": task["error"],
        }
    )


@app.get("/api/chat/stream/{task_id}", response_model=None)
async def api_chat_stream(
    task_id: str, request: Request
) -> StreamingResponse | JSONResponse:
    """Stream accumulated and future SSE events for a background task."""
    app_state: AppState = request.app.state.app_state
    async with app_state.task_lock:
        task = app_state.task_store.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    async def stream_events() -> AsyncGenerator[str, None]:
        log.debug(
            "stream_events started, task done=%s events=%d",
            task["done"],
            len(task["events"]),
        )
        # Yield any events already accumulated
        for event_json in task["events"]:
            yield f"data: {event_json}\n\n"
        if task["done"]:
            yield "data: [DONE]\n\n"
            return
        # Yield events as they arrive
        last_len = len(task["events"])
        while not task["done"]:
            if task.get("cancelled") and task["cancelled"].is_set():
                return
            await asyncio.sleep(0.2)
            new_events = task["events"][last_len:]
            for event_json in new_events:
                yield f"data: {event_json}\n\n"
            last_len = len(task["events"])
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/info")
async def api_info(request: Request) -> JSONResponse:
    """Return runtime info including the active model name."""
    app_state: AppState = request.app.state.app_state
    return JSONResponse({"model": app_state.model})


@app.get("/api/sessions")
async def api_list_sessions() -> list:
    return list_sessions()


@app.post("/api/session")
async def api_create_session(_request: Request) -> JSONResponse:
    sid = create_session()
    response = JSONResponse({"id": sid})
    response.set_cookie(
        key="session_id",
        value=sid,
        max_age=31536000,
        path="/",
        samesite="lax",
    )
    return response


@app.get("/api/session")
async def api_get_session(request: Request) -> JSONResponse:
    sid = get_session_id(request)
    response = JSONResponse(load_session(sid))
    response.set_cookie(
        key="session_id",
        value=sid,
        max_age=31536000,
        path="/",
        samesite="lax",
    )
    return response


@app.put("/api/session")
async def api_save_session(request: Request) -> JSONResponse:
    body = await request.json()
    sid = get_session_id(request)
    save_session(sid, body.get("history", []))
    response = JSONResponse({"ok": True})
    response.set_cookie(
        key="session_id",
        value=sid,
        max_age=31536000,
        path="/",
        samesite="lax",
    )
    return response


@app.delete("/api/session")
async def api_delete_session(request: Request) -> JSONResponse:
    sid = get_session_id(request)
    delete_session(sid)
    response = JSONResponse({"ok": True})
    response.delete_cookie(key="session_id", path="/")
    return response


@app.get("/api/session/{session_id}")
async def api_get_specific_session(session_id: str) -> dict:
    data = load_session(session_id)
    return data


@app.delete("/api/session/{session_id}")
async def api_delete_specific_session(session_id: str) -> JSONResponse:
    delete_session(session_id)
    return JSONResponse({"ok": True})


@app.get("/manifest.json")
async def manifest() -> JSONResponse:
    return JSONResponse(_MANIFEST)


@app.get("/sw.js")
async def service_worker() -> Response:
    return Response(content=_SW_JS, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = ROOT / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/favicon.ico")
async def favicon() -> HTMLResponse:
    return HTMLResponse("", status_code=204)


# Ensure video MIME types for <video> to work in Safari
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/webm", ".webm")

app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════╗")
    print("║         Personal LAN Agent               ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  UI     →  https://{BIND_HOST}:{BIND_PORT}         ║")
    print(f"║  LLM    →  {LLAMA_SERVER:<30} ║")
    print(f"║  ImGen  →  {IMAGE_SERVER:<30} ║")
    print("╚══════════════════════════════════════════╝")
    print("  (self-signed cert — browser will warn; click 'Advanced → Proceed')")
    print()
    uvicorn.run(
        app,
        host=BIND_HOST,
        port=BIND_PORT,
        log_level="info",
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
    )
