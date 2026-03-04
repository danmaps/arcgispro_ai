import os
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_DIR = Path.home() / ".arcgispro_ai"
DEFAULT_GITHUB_MODELS_TOKEN_FILE = DEFAULT_CONFIG_DIR / "github_models_token.txt"
TOKEN_SETUP_GUIDANCE = (
    "GitHub Models token not found. Run tools\\auth\\auth_github_models.bat "
    "to configure authentication."
)


def get_github_models_token_path() -> Path:
    """Return the configured GitHub Models token file path."""
    override = os.environ.get("ARCGISPRO_AI_GITHUB_MODELS_TOKEN_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_GITHUB_MODELS_TOKEN_FILE


def load_github_models_token(token_path: Optional[str] = None) -> str:
    """Load the locally stored GitHub Models token."""
    path = Path(token_path).expanduser() if token_path else get_github_models_token_path()

    try:
        token_value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(TOKEN_SETUP_GUIDANCE) from exc
    except OSError as exc:
        raise ValueError(f"Unable to read GitHub Models token file at {path}: {exc}") from exc

    if not token_value:
        raise ValueError(TOKEN_SETUP_GUIDANCE)

    return token_value
