from typing import Optional

# Known multimodal model markers per provider. The match is case-insensitive and
# only used to guard tools that attempt to send screenshots.
VISION_MODEL_HINTS = {
    "OpenAI": ["gpt-4o", "gpt-4.1", "omni", "vision"],
    "Azure OpenAI": ["gpt-4o", "gpt-4.1", "omni", "vision"],
    "OpenRouter": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1",
        "google/gemini",
        "anthropic/claude-3.5",
        "anthropic/claude-3-opus",
        "meta-llama/llama-3.2-vision",
    ],
}

# Fallback short list for OpenRouter if dynamic catalog fetch fails
DEFAULT_OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/o3-mini",
    "google/gemini-2.0-flash-exp:free",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
]


def model_supports_images(source: str, model: Optional[str] = None) -> bool:
    """Return True if the selected model is known to accept image input.

    Heuristic check using provider-specific substrings; conservative by default.
    """
    provider_hints = VISION_MODEL_HINTS.get(source, [])
    normalized = (model or "").lower().strip() if model else ""

    if not provider_hints:
        return False
    if not normalized:
        # Fall back to provider defaults if the user did not specify a model name.
        return source in ("OpenAI", "Azure OpenAI")

    return any(hint in normalized for hint in provider_hints)
