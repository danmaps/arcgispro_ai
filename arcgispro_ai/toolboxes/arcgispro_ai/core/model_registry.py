from typing import Optional

# Known multimodal model markers per provider. The match is case-insensitive and
# only used to guard tools that attempt to send screenshots.
VISION_MODEL_HINTS = {
    "OpenAI": ["gpt-4o", "gpt-4.1", "omni", "vision"],
    "Azure OpenAI": ["gpt-4o", "gpt-4.1", "omni", "vision"],
    "OpenRouter": [
        "openai/gpt-4o",
        "openai/gpt-4.1",
        "openai/gpt-5",
        "google/gemini-2.5",
        "google/gemini-3",
        "anthropic/claude-4",
        "anthropic/claude-4.5",
        "meta-llama/llama-3.2-vision",
    ],
}

# Curated OpenRouter model list aligned with the Copilot-style picker.
OPENROUTER_CURATED_MODELS = [
    "openai/gpt-4.1",
    "openai/gpt-4o",
    "openai/gpt-5-mini",
    "anthropic/claude-4.5-haiku",
    "anthropic/claude-4.5-opus",
    "anthropic/claude-4-sonnet",
    "anthropic/claude-4.5-sonnet",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "openai/gpt-5",
    "openai/gpt-5-codex-preview",
    "openai/gpt-5.1",
    "openai/gpt-5.1-codex",
    "openai/gpt-5.1-codex-max",
    "openai/gpt-5.1-codex-mini-preview",
    "openai/gpt-5.2",
    "openai/gpt-5.2-codex",
]

# Fallback short list for OpenRouter if dynamic catalog fetch fails
DEFAULT_OPENROUTER_MODELS = OPENROUTER_CURATED_MODELS


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
