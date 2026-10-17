"""ComfyUI multi-workflow registry and execution."""

import asyncio
import base64
import contextlib
import json
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import httpx
import yaml
from PIL import Image as PILImage
from pypdf import PdfReader

log = logging.getLogger("agent.comfy")

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_SERVER = "http://localhost:7860"  # ComfyUI local server
COMFY_MODEL = ""  # checkpoint (only for hardcoded workflow)
COMFY_WORKFLOW_FILE = (
    "text_to_image_zturbo.json"  # deprecated, kept for backwards compat
)
COMFY_WORKFLOWS_DIR = Path(__file__).parent / "comfy_workflows"

# Static files directory
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)

# User-uploaded images (input)
_input_dir = _static_dir / "input"
_input_dir.mkdir(exist_ok=True)

# Generated images (output)
_gen_dir = _static_dir / "generated"
_gen_dir.mkdir(exist_ok=True)

# ── Workflow Registry ─────────────────────────────────────────────────────────

_DSL_PATTERN = re.compile(
    r"\$(?P<name>[a-zA-Z_]\w*)\.(?P<field>[a-zA-Z_]\w*)(?:=(?P<default>[^!\s]+))?(?P<required>!)?"
)
_OUTPUT_PATTERN = re.compile(r"\$OUTPUT(?:\.(?P<key>\w+))?")


def parse_dsl_title(node_id: str, node: dict) -> list | None:
    """Parse DSL annotations from a node's _meta.title field."""
    title = node.get("_meta", {}).get("title", "")
    results = []

    output_match = _OUTPUT_PATTERN.search(title)
    if output_match:
        results.append(
            {
                "node": node_id,
                "is_output": True,
                "output_key": output_match.group("key"),
            }
        )

    for m in _DSL_PATTERN.finditer(title):
        results.append(
            {
                "param_name": m.group("name"),
                "node": node_id,
                "field": m.group("field"),
                "required": bool(m.group("required")),
                "default": m.group("default"),
                "is_output": False,
            }
        )

    return results if results else None


def _infer_type(node: dict, fld: str) -> str:
    """Guess JSON schema type from the current value in node inputs."""
    val = node.get("inputs", {}).get(fld)
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int):
        return "integer"
    if isinstance(val, float):
        return "number"
    return "string"


OUTPUT_NODE_CLASSES = {
    "SaveImage",
    "PreviewImage",
    "VHS_VideoCombine",
    "SaveAudio",
    "VHS_SaveAudio",
    "WanVideoSave",
}


def _detect_output_nodes(workflow: dict) -> list[str]:
    """Fallback: find SaveImage / VHS_VideoCombine / SaveAudio nodes."""
    return [
        nid for nid, n in workflow.items() if n.get("class_type") in OUTPUT_NODE_CLASSES
    ]


def dsl_to_sidecar(workflow_path: Path, workflow: dict) -> dict:
    """Build a sidecar dict from DSL annotations in _meta.title fields."""
    parameters = {}
    output_nodes = []
    output_type = "image"

    for node_id, node in workflow.items():
        parsed = parse_dsl_title(node_id, node)
        if parsed is None:
            continue
        if not isinstance(parsed, list):
            continue
        for entry in parsed:  # pylint: disable=not-an-iterable
            if entry["is_output"]:
                output_nodes.append(node_id)
                ct = node.get("class_type", "")
                if "video" in ct.lower() or ct in ("VHS_VideoCombine",):
                    output_type = "video"
                elif "audio" in ct.lower():
                    output_type = "audio"
            else:
                parameters[entry["param_name"]] = {
                    "node": entry["node"],
                    "field": entry["field"],
                    "type": _infer_type(node, entry["field"]),
                    "required": entry["required"],
                    "default": entry["default"],
                    "description": "",
                }

    stem = workflow_path.stem
    return {
        "name": stem,
        "description": f"Run ComfyUI workflow '{stem}'",
        "output_type": output_type,
        "output_nodes": output_nodes or _detect_output_nodes(workflow),
        "vram_hint": "unknown",
        "parameters": parameters,
        "models": [],
    }


def load_comfy_workflows() -> dict:
    """Auto-discover comfy_workflows/*.json and load their sidecars."""
    registry = {}

    COMFY_WORKFLOWS_DIR.mkdir(exist_ok=True)

    for wf_path in sorted(COMFY_WORKFLOWS_DIR.glob("*.json")):
        stem = wf_path.stem
        try:
            workflow = json.loads(wf_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("skip %s: JSON parse error: %s", wf_path.name, e)
            continue

        yaml_path = wf_path.with_suffix(".yaml")
        if yaml_path.exists():
            try:
                sidecar = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as e:
                log.warning("YAML parse error %s: %s", yaml_path.name, e)
                sidecar = dsl_to_sidecar(wf_path, workflow)
        else:
            sidecar = dsl_to_sidecar(wf_path, workflow)

        sidecar.setdefault("name", stem)
        sidecar.setdefault("description", f"Run workflow '{stem}'")
        sidecar.setdefault("output_type", "image")
        sidecar.setdefault("output_nodes", _detect_output_nodes(workflow))
        sidecar.setdefault("parameters", {})
        sidecar.setdefault("models", [])

        registry[sidecar["name"]] = {
            "sidecar": sidecar,
            "workflow": workflow,
            "path": wf_path,
        }
        log.info(
            "loaded '%s' (%d nodes, output_type=%s)",
            sidecar["name"],
            len(workflow),
            sidecar["output_type"],
        )

    # Backwards-compat: legacy COMFY_WORKFLOW_FILE
    if COMFY_WORKFLOW_FILE:
        old = Path(COMFY_WORKFLOW_FILE)
        if old.exists() and old.stem not in registry:
            try:
                wf = json.loads(old.read_text(encoding="utf-8"))
                sc = dsl_to_sidecar(old, wf)
                sc["name"] = "generate_image"
                sc["description"] = (
                    "Generate an image using Stable Diffusion. "
                    "Use for illustrations, artwork, diagrams, or any visual content."
                )
                registry["generate_image"] = {
                    "sidecar": sc,
                    "workflow": wf,
                    "path": old,
                }
                log.info("registered legacy generate_image from %s", old)
            except json.JSONDecodeError as e:
                log.warning("legacy workflow load failed: %s", e)

    return registry


def _enrich_schema_with_enum(
    schema: dict, param: dict, entry: dict, object_info: dict
) -> None:
    """Enrich a parameter schema with valid enum choices from /object_info."""
    node_id = param.get("node")
    fld = param.get("field")
    wf = entry["workflow"]
    if not (node_id and fld and node_id in wf):
        return
    ct = wf[node_id].get("class_type")
    if not (ct and ct in object_info):
        return
    node_schema = object_info[ct]
    for section in ("required", "optional"):
        inputs = node_schema.get("input", {}).get(section, {})
        if fld in inputs:
            spec = inputs[fld]
            if isinstance(spec, list) and len(spec) > 0 and isinstance(spec[0], list):
                schema["enum"] = spec[0]
            break


def _build_tool_schema(name: str, sc: dict, props: dict, required: list) -> dict:
    """Build a single OpenAI tool schema dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": sc["description"],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def build_tools_from_workflows(registry: dict, object_info: dict | None = None) -> list:
    """Generate OpenAI tool schema list from the workflow registry."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for current information, news, facts, prices, "
                    "recent events, or anything that might have changed recently."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Number of results (1-10, default 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": (
                    "Fetch extracted text from a web page. "
                    "Use this after web_search to read promising URLs. "
                    "Returns main text or page excerpts, depending on the search "
                    "provider (up to ~6000 characters). "
                    "The result may not cover the entire page. "
                    "Do not use for PDFs, images, or binary file downloads."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL to fetch (must start with http:// or https://).",
                        }
                    },
                    "required": ["url"],
                },
            },
        },
    ]

    for name, entry in registry.items():
        sc = entry["sidecar"]
        props = {}
        required = []

        for pname, p in sc.get("parameters", {}).items():
            schema = {"type": p.get("type", "string")}
            if p.get("description"):
                schema["description"] = p["description"]
            if "default" in p and p["default"] is not None:
                schema["default"] = p["default"]
            if "minimum" in p:
                schema["minimum"] = p["minimum"]
            if "maximum" in p:
                schema["maximum"] = p["maximum"]

            if object_info:
                _enrich_schema_with_enum(schema, p, entry, object_info)

            props[pname] = schema
            if p.get("required"):
                required.append(pname)

        tools.append(_build_tool_schema(name, sc, props, required))

    return tools


def _check_model_enum(
    wf_name: str, loader_node: str, fld: str, current_val: str, oi_node: dict
) -> None:
    """Check if a model value is in the object_info enum; warn if not."""
    for section in ("required", "optional"):
        inputs = oi_node.get("input", {}).get(section, {})
        if fld in inputs:
            spec = inputs[fld]
            if isinstance(spec, list) and len(spec) > 0 and isinstance(spec[0], list):
                valid = spec[0]
                if current_val and current_val not in valid:
                    log.warning(
                        "workflow '%s' node '%s' field '%s' = '%s' not in installed models: %s...",
                        wf_name,
                        loader_node,
                        fld,
                        current_val,
                        valid[:5],
                    )
            break


def validate_workflow_models(registry: dict, object_info: dict):
    """Warn if any sidecar model name is not in the /object_info enum list."""
    for wf_name, entry in registry.items():
        sc = entry["sidecar"]
        wf = entry["workflow"]
        for model_spec in sc.get("models", []):
            loader_node = str(model_spec.get("loader_node", ""))
            fld = model_spec.get("field", "")
            ct = model_spec.get("class_type", "")
            if not (loader_node and fld and ct):
                continue
            current_val = wf.get(loader_node, {}).get("inputs", {}).get(fld, "")
            oi_node = object_info.get(ct, {})
            _check_model_enum(wf_name, loader_node, fld, current_val, oi_node)


# ── Runtime config (set once by init_comfy(), read-only after) ──────────────────


@dataclass
class _ComfyConfig:
    tools: list = field(default_factory=list)
    last_image: str = ""
    models_dir: Path | None = None


comfy_cfg = _ComfyConfig()


def init_comfy(models_dir: Path | None) -> _ComfyConfig:
    """Call once at startup. Returns the config object for use by other functions."""
    comfy_cfg.models_dir = models_dir
    return comfy_cfg


# ── Model auto-download ───────────────────────────────────────────────────────


def check_missing_models(sidecar: dict, models_dir: Path | None) -> list[str]:
    """Return filenames of models declared in sidecar['downloads'] that are not on disk."""
    if models_dir is None:
        return []
    missing = []
    for spec in sidecar.get("downloads", []):
        filename = spec.get("filename", "")
        dest = spec.get("dest", "")
        if not (filename and dest):
            continue
        if not (models_dir / dest / filename).exists():
            missing.append(filename)
    return missing


async def ensure_models(sidecar: dict, models_dir: Path | None = None):
    """Yield an error SSE event if any required models are missing, telling user to run setup.py."""
    missing = check_missing_models(sidecar, models_dir)
    if missing:
        names = ", ".join(missing)
        yield json.dumps(
            {
                "type": "error",
                "msg": (
                    f"Missing model files: {names}\n\n"
                    "Run **`python setup.py --models`** to download the models "
                    "required for this workflow, then try again."
                ),
            }
        )


# ── Legacy workflow loader ────────────────────────────────────────────────────


def _load_workflow(
    prompt: str,
    *,
    negative_prompt: str = "",
    steps: int = 22,
    width: int = 512,
    height: int = 512,
    seed: int = 0,
) -> dict:
    """
    Load a ComfyUI workflow from COMFY_WORKFLOW_FILE (or use the built-in
    minimal workflow), then patch runtime values.

    Patched fields:
      KSampler:     seed, steps
      CLIPTextEncode: text (positive via _meta title detection; negative via ConditioningZeroOut)
      EmptyLatentImage / EmptySD3LatentImage: width, height, batch_size
      ConditioningZeroOut: negative conditioning passthrough
    """
    wf_path = COMFY_WORKFLOW_FILE
    if wf_path:
        wf_file = Path(wf_path)
        if wf_file.exists():
            wf = json.loads(wf_file.read_text(encoding="utf-8"))
        else:
            wf = None
    else:
        wf = None
    if wf is None:
        log.info("workflow file not found, using hardcoded fallback")
        # Minimal hardcoded fallback
        wf = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": COMFY_MODEL or "v1-5-pruned.ckpt"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "width": width, "height": height},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["4", 1]},
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 7.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "agent", "images": ["8", 0]},
            },
        }

    # Patch runtime values
    for _nid, node in wf.items():
        ct = node.get("class_type", "")
        inp = node.get("inputs", {})

        if ct == "KSampler":
            inp["seed"] = seed
            inp["steps"] = steps
        elif ct in ("EmptyLatentImage", "EmptySD3LatentImage"):
            inp["width"] = width
            inp["height"] = height
            inp["batch_size"] = 1
        elif ct == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "").lower()
            if "neg" in title:
                inp["text"] = negative_prompt
            else:
                inp["text"] = prompt
        elif ct == "ConditioningZeroOut":
            # Negative conditioning — already wired to zero out;
            # the positive prompt in the referenced node is patched above.
            pass

    return wf


# ── ComfyUI execution ─────────────────────────────────────────────────────────

_SKIP_NODE_CLASSES = {
    "LoadImage",
    "LoadImageMask",
    "LoadLatent",
    "LoadAudio",
    "VHS_LoadVideo",
    "LoadImagePath",
}


async def exec_generate_image(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, watermark",
    steps: int = 22,
    width: int = 512,
    height: int = 512,
    _last_image: str = "",
) -> tuple[str, str | None]:
    """
    Generate an image via the ComfyUI local API (no cloud, fully local).
    Returns (status_text, url_or_None).
    """
    log.info(
        "exec_generate_image prompt=%r neg=%r steps=%d size=%dx%d",
        prompt,
        negative_prompt,
        steps,
        width,
        height,
    )

    seed = random.randint(0, 2**32 - 1)

    workflow = _load_workflow(
        prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
    )
    client_id = str(uuid.uuid4())

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Health check (non-blocking)
            with contextlib.suppress(httpx.RequestError):
                await client.get(f"{IMAGE_SERVER}/")

            # Submit workflow
            resp = await client.post(
                f"{IMAGE_SERVER}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            data = resp.json()
            if "error" in data or data.get("node_errors"):
                errs = data.get("node_errors") or data.get("error")
                return f"ComfyUI workflow error: {errs}", None
            prompt_id: str = data["prompt_id"]

            # Poll for completion
            image_meta = await _poll_history(client, prompt_id)
            if image_meta is None:
                return "ComfyUI timed out waiting for image.", None

            # Download and save
            return await _save_generated_image(client, image_meta, prompt)

    except httpx.HTTPError as e:
        log.error("HTTP error: %s: %s", type(e).__name__, e)
        return f"Image generation failed: {e}", None
    except Exception as e:
        log.error("EXCEPTION: %s: %s", type(e).__name__, e)
        return f"Image generation failed: {e}", None


async def _poll_history(client: httpx.AsyncClient, prompt_id: str) -> dict | None:
    """Poll ComfyUI /history until the prompt completes. Returns image meta or None."""
    log.info("polling /history/%s ...", prompt_id)
    for _i in range(600):
        await asyncio.sleep(0.5)
        hist_resp = await client.get(f"{IMAGE_SERVER}/history/{prompt_id}")
        hist_resp.raise_for_status()
        history = hist_resp.json()
        if _i % 20 == 0:
            log.debug("poll #%d: history keys=%s", _i, list(history.keys()))
        if prompt_id in history:
            log.info("job completed after %d polls", _i + 1)
            break
    else:
        log.error("TIMEOUT after 600 polls")
        return None

    outputs = history[prompt_id].get("outputs", {})
    for node_output in outputs.values():
        images = node_output.get("images", [])
        if images:
            return images[0]
    log.error("NO image found in outputs")
    return None


async def _save_generated_image(
    client: httpx.AsyncClient,
    image_meta: dict,
    prompt: str,
) -> tuple[str, str | None]:
    """Download a ComfyUI output image, resize, save, and return (status_text, url)."""
    log.info("GET /view filename=%s", image_meta["filename"])
    view_resp = await client.get(
        f"{IMAGE_SERVER}/view",
        params={
            "filename": image_meta["filename"],
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        },
    )
    view_resp.raise_for_status()

    filename = f"gen_{uuid.uuid4().hex[:12]}.jpg"
    img_path = _gen_dir / filename
    img: PILImage.Image = PILImage.open(BytesIO(view_resp.content))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > 768:
        ratio = 768 / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            PILImage.Resampling.LANCZOS,
        )
    img.save(img_path, format="JPEG", quality=85)
    url = f"/static/generated/{filename}"
    log.info("saved image: %s (%d bytes)", url, img_path.stat().st_size)
    return f"Image generated for prompt: {prompt}", url


# ── Attachment helpers ────────────────────────────────────────────────────────


def _resize_image(b64_data: str, max_side: int = 768) -> str:
    """Decode base64 image, resize longest side to max_side, re-encode as JPEG."""
    try:
        img: PILImage.Image = PILImage.open(BytesIO(base64.b64decode(b64_data)))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize(
                (int(w * ratio), int(h * ratio)), PILImage.Resampling.LANCZOS
            )
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return b64_data  # fall back to original if processing fails


def _extract_pdf_text(b64_data: str) -> str:
    """Decode base64 PDF and extract all page text."""
    try:
        reader = PdfReader(BytesIO(base64.b64decode(b64_data)))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(t for t in pages if t.strip())
    except Exception as e:
        return f"[PDF text extraction failed: {e}]"


async def upload_image_to_comfy(
    http: httpx.AsyncClient, b64_data: str, filename: str, _last_image: str = ""
) -> str:
    """Upload a base64-encoded image to ComfyUI /upload/image. Returns the stored filename."""
    raw = base64.b64decode(b64_data)
    resp = await http.post(
        f"{IMAGE_SERVER}/upload/image",
        files={
            "image": (filename, raw, "image/jpeg"),
            "type": (None, "input"),
            "overwrite": (None, "true"),
        },
    )
    resp.raise_for_status()
    name = resp.json()["name"]
    return name


_image_url_re = re.compile(r"!\[.*?\]\((/static/(?:input|generated)/[^)]+)\)")


def _extract_image_urls_from_history(messages: list[dict]) -> list[str]:
    """Scan message history for markdown image URLs, newest first."""
    urls: list[str] = []
    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, str):
            urls.extend(_image_url_re.findall(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    urls.extend(_image_url_re.findall(part))
                elif isinstance(part, dict):
                    text = part.get("text", "")
                    if isinstance(text, str):
                        urls.extend(_image_url_re.findall(text))
    return urls


async def resolve_image_for_comfy(
    http: httpx.AsyncClient,
    url_arg: str | None,
    attachment=None,
    last_image: str = "",
    history: list[dict] | None = None,
    last_image_url: str = "",
) -> str | None:
    """
    Resolve the best available image to a ComfyUI input filename.
    Priority: LLM url_arg → attachment → history URLs → last_image_url → last uploaded image.
    """
    _search_dirs = [_input_dir, _gen_dir]

    async def _try_upload(b64: str, fname: str) -> str | None:
        try:
            return await upload_image_to_comfy(http, b64, fname, last_image)
        except Exception as e:
            log.warning("ComfyUI upload failed for %r: %s", fname, e)
            return None

    if url_arg and url_arg.startswith("/static/"):
        local = Path(__file__).parent / url_arg.lstrip("/")
        if local.exists():
            result = await _try_upload(
                base64.b64encode(local.read_bytes()).decode(), local.name
            )
            if result:
                return result
        else:
            log.warning("image_source URL not found locally: %s", url_arg)

    if url_arg and "/" not in url_arg and "\\" not in url_arg:
        for d in _search_dirs:
            candidate = d / url_arg
            if candidate.exists():
                log.info("using image from %s: %s", d.name, url_arg)
                result = await _try_upload(
                    base64.b64encode(candidate.read_bytes()).decode(), candidate.name
                )
                if result:
                    return result

    if attachment and getattr(attachment, "type", None) == "image":
        result = await _try_upload(attachment.data, attachment.filename)
        if result:
            return result

    if history:
        for url in _extract_image_urls_from_history(history):
            local = Path(__file__).parent / url.lstrip("/")
            if local.exists():
                b64 = base64.b64encode(local.read_bytes()).decode()
                log.info("using image from history: %s", url)
                result = await _try_upload(b64, local.name)
                if result:
                    return result

    if last_image_url and last_image_url.startswith("/static/"):
        local = Path(__file__).parent / last_image_url.lstrip("/")
        if local.exists():
            b64 = base64.b64encode(local.read_bytes()).decode()
            log.info("using last_image_url: %s", last_image_url)
            result = await _try_upload(b64, local.name)
            if result:
                return result

    if last_image:
        log.info("using last ComfyUI image %r as fallback", last_image)
        return last_image

    return None
