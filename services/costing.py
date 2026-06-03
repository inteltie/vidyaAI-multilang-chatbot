"""Cost calculation helpers for per-request token usage."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from config import settings
from state import merge_usage_by_component


MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

EMBEDDING_PRICING_PER_1M: dict[str, float] = {
    "text-embedding-3-large": 0.13,
}

REASONING_MODEL_PREFIXES = ("o1", "o3", "o4-mini", "o4", "gpt-5")


def _normalize_model_name(model_name: str) -> str:
    return (model_name or "").strip().lower()


def _usd(tokens: int, price_per_million: float) -> float:
    return (int(tokens or 0) * float(price_per_million or 0.0)) / 1_000_000


def _resolve_chat_pricing(model_name: str) -> dict[str, float]:
    norm = _normalize_model_name(model_name)
    if norm in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[norm]
    return MODEL_PRICING_PER_1M["gpt-4o-mini"]


def _resolve_embedding_pricing(model_name: str) -> float:
    norm = _normalize_model_name(model_name)
    if norm in EMBEDDING_PRICING_PER_1M:
        return EMBEDDING_PRICING_PER_1M[norm]
    return EMBEDDING_PRICING_PER_1M["text-embedding-3-large"]


def web_search_content_tokens_billed(web_search_model_name: str) -> bool:
    """Whether web-search content tokens are billed (reasoning models) or free."""
    explicit = getattr(settings, "web_search_content_tokens_billed", None)
    if explicit is not None:
        return bool(explicit)
    norm = _normalize_model_name(web_search_model_name)
    return any(norm.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)


def _component_token_cost(component_data: Dict[str, Any]) -> float:
    total = 0.0
    for key, usage in (component_data or {}).items():
        if key in ("calls", "images") or not isinstance(usage, dict):
            continue
        pricing = _resolve_chat_pricing(str(key))
        total += _usd(int(usage.get("input", 0) or 0), pricing["input"])
        total += _usd(int(usage.get("output", 0) or 0), pricing["output"])
    return total


def parse_web_search_result(
    result: Any,
) -> Tuple[str, int, int, bool]:
    """Parse WebSearchTool result: (text, input_tokens, output_tokens, billable_call)."""
    if isinstance(result, tuple):
        if len(result) >= 4:
            return (
                str(result[0]),
                int(result[1] or 0),
                int(result[2] or 0),
                bool(result[3]),
            )
        if len(result) == 3:
            text, i_t, o_t = result
            billable = bool(i_t or o_t)
            return str(text), int(i_t or 0), int(o_t or 0), billable
    return str(result), 0, 0, False


def patch_component_usage(
    component: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    billable_web_call: bool = False,
    images_analyzed: int = 0,
    billable_image_generation: bool = False,
) -> Dict[str, Any]:
    """Build a LangGraph state patch for component usage + legacy token_usage_by_model."""
    patch: Dict[str, Any] = {}
    comp_entry: Dict[str, Any] = {}

    if input_tokens or output_tokens:
        comp_entry[model] = {
            "input": int(input_tokens or 0),
            "output": int(output_tokens or 0),
        }
        patch["token_usage_by_model"] = {
            model: {"input": int(input_tokens or 0), "output": int(output_tokens or 0)},
        }

    if component == "web_search" and billable_web_call:
        comp_entry["calls"] = 1
        patch["web_search_calls"] = 1

    if component == "vision" and images_analyzed:
        comp_entry["images"] = int(images_analyzed)

    if comp_entry:
        patch["usage_by_component"] = {component: comp_entry}

    if component == "vision" and images_analyzed:
        patch["images_analyzed"] = int(images_analyzed)

    if component == "image_generation" and billable_image_generation:
        comp_entry["calls"] = 1
        patch["image_generation_calls"] = 1

    return patch


def merge_patches(*patches: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple patch_component_usage dicts into one state update."""
    merged: Dict[str, Any] = {}
    usage: Dict[str, Dict[str, Any]] = {}
    token_usage: Dict[str, Dict[str, int]] = {}
    web_calls = 0
    images = 0
    image_gen_calls = 0

    for patch in patches:
        if not patch:
            continue
        usage = merge_usage_by_component(usage, patch.get("usage_by_component") or {})
        token_usage = _merge_token_maps(token_usage, patch.get("token_usage_by_model") or {})
        web_calls += int(patch.get("web_search_calls", 0) or 0)
        images += int(patch.get("images_analyzed", 0) or 0)
        image_gen_calls += int(patch.get("image_generation_calls", 0) or 0)
        for key, val in patch.items():
            if key in (
                "usage_by_component",
                "token_usage_by_model",
                "web_search_calls",
                "images_analyzed",
                "image_generation_calls",
            ):
                continue
            if key in merged and isinstance(merged[key], (int, float)) and isinstance(val, (int, float)):
                merged[key] = merged[key] + val
            else:
                merged[key] = val

    if usage:
        merged["usage_by_component"] = usage
    if token_usage:
        merged["token_usage_by_model"] = token_usage
    if web_calls:
        merged["web_search_calls"] = web_calls
    if images:
        merged["images_analyzed"] = images
    if image_gen_calls:
        merged["image_generation_calls"] = image_gen_calls
    return merged


def _merge_token_maps(
    left: Dict[str, Dict[str, int]],
    right: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    merged: Dict[str, Dict[str, int]] = dict(left or {})
    for model_name, usage in (right or {}).items():
        if not model_name:
            continue
        row = merged.setdefault(model_name, {"input": 0, "output": 0})
        row["input"] += int((usage or {}).get("input", 0) or 0)
        row["output"] += int((usage or {}).get("output", 0) or 0)
    return merged


def _normalize_usage_for_cost(
    usage_by_component: Dict[str, Dict[str, Any]] | None,
    token_usage_by_model: Dict[str, Dict[str, int]] | None,
) -> Dict[str, Dict[str, Any]]:
    if usage_by_component:
        return usage_by_component
    if token_usage_by_model:
        return {"chat": dict(token_usage_by_model)}
    return {}


def compute_request_cost(
    usage_by_component: Dict[str, Dict[str, Any]] | None = None,
    web_search_calls: int = 0,
    image_generation_calls: int = 0,
    embedding_tokens: int = 0,
    embedding_model_name: str = "text-embedding-3-large",
    background_input_tokens: int = 0,
    background_output_tokens: int = 0,
    background_model_name: str = "gpt-4.1-nano",
    *,
    web_search_model_name: str = "gpt-4o-mini",
    token_usage_by_model: Dict[str, Dict[str, int]] | None = None,
) -> Dict[str, Any]:
    """Compute detailed USD cost for one request."""
    usage = _normalize_usage_for_cost(usage_by_component, token_usage_by_model)
    web_model = web_search_model_name or settings.web_search_model_name or "gpt-4o-mini"
    fee_per_1k = float(getattr(settings, "web_search_fee_per_1k_calls_usd", 25.0) or 25.0)

    web_calls = int(web_search_calls or 0)
    web_comp = usage.get("web_search") or {}
    if not web_calls:
        web_calls = int(web_comp.get("calls", 0) or 0)

    cost_web_tool = web_calls * (fee_per_1k / 1000.0)
    cost_web_tokens = 0.0
    if web_search_content_tokens_billed(web_model):
        cost_web_tokens = _component_token_cost(web_comp)

    cost_vision = _component_token_cost(usage.get("vision") or {})
    image_gen_comp = usage.get("image_generation") or {}
    gen_calls = int(image_generation_calls or 0)
    if not gen_calls:
        gen_calls = int(image_gen_comp.get("calls", 0) or 0)
    gen_fee = float(getattr(settings, "image_generation_fee_per_call_usd", 0.041) or 0.041)
    fee_override = image_gen_comp.get("fee_usd")
    if fee_override is not None and gen_calls:
        cost_image_generation = float(fee_override)
    else:
        cost_image_generation = gen_calls * gen_fee
    cost_chat = _component_token_cost(usage.get("chat") or {})
    cost_image_gate = _component_token_cost(usage.get("image_gate") or {})
    cost_llm = cost_chat + cost_image_gate

    bg_pricing = _resolve_chat_pricing(background_model_name)
    background_cost = _usd(background_input_tokens, bg_pricing["input"]) + _usd(
        background_output_tokens, bg_pricing["output"]
    )
    embedding_cost = _usd(embedding_tokens, _resolve_embedding_pricing(embedding_model_name))
    cost_web_search = cost_web_tool + cost_web_tokens
    total_cost = (
        cost_llm
        + cost_web_search
        + cost_vision
        + cost_image_generation
        + background_cost
        + embedding_cost
    )

    by_model: dict[str, dict[str, float | int]] = {}
    for component in ("chat", "vision", "image_gate", "web_search"):
        comp_data = usage.get(component) or {}
        for model_name, model_usage in comp_data.items():
            if model_name in ("calls", "images") or not isinstance(model_usage, dict):
                continue
            input_tokens = int(model_usage.get("input", 0) or 0)
            output_tokens = int(model_usage.get("output", 0) or 0)
            if not input_tokens and not output_tokens:
                continue
            pricing = _resolve_chat_pricing(model_name)
            input_cost = _usd(input_tokens, pricing["input"])
            output_cost = _usd(output_tokens, pricing["output"])
            row = by_model.setdefault(model_name, {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            })
            row["input_tokens"] = int(row["input_tokens"]) + input_tokens
            row["output_tokens"] = int(row["output_tokens"]) + output_tokens
            row["input_cost_usd"] = float(row["input_cost_usd"]) + input_cost
            row["output_cost_usd"] = float(row["output_cost_usd"]) + output_cost
            row["total_cost_usd"] = float(row["total_cost_usd"]) + input_cost + output_cost

    images_analyzed = 0
    vision_comp = usage.get("vision") or {}
    images_analyzed = int(vision_comp.get("images", 0) or 0)

    return {
        "cost_usd_total": total_cost,
        "cost_usd_llm": cost_llm,
        "cost_usd_web_search": cost_web_search,
        "cost_usd_vision": cost_vision,
        "cost_usd_image_generation": cost_image_generation,
        "cost_usd_background": background_cost,
        "cost_usd_embedding": embedding_cost,
        "cost_by_component_usd": {
            "chat": cost_chat,
            "image_gate": cost_image_gate,
            "web_search": cost_web_search,
            "vision": cost_vision,
            "image_generation": cost_image_generation,
            "background": background_cost,
            "embedding": embedding_cost,
        },
        "cost_by_model_usd": by_model,
        "web_search_calls": web_calls,
        "images_analyzed": images_analyzed,
        "image_generation_calls": gen_calls,
    }
