import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from arcgispro_ai.toolboxes.arcgispro_ai.auth import (
    get_github_models_token_path,
    load_github_models_token,
)


class TestGitHubModelsAuth(unittest.TestCase):
    @patch("pathlib.Path.read_text", return_value="ghp_test_token\n")
    def test_load_token_from_explicit_path(self, mock_read_text):
        token = load_github_models_token("C:\\fake\\github_models_token.txt")
        self.assertEqual(token, "ghp_test_token")
        mock_read_text.assert_called_once()

    def test_missing_token_file_raises_helpful_error(self):
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
            with self.assertRaises(ValueError) as exc:
                load_github_models_token("C:\\fake\\missing_token.txt")
            self.assertIn("auth_github_models.bat", str(exc.exception))

    def test_env_override_path(self):
        token_path = Path("C:/temp/github_models_token.txt")
        with patch.dict(os.environ, {"ARCGISPRO_AI_GITHUB_MODELS_TOKEN_PATH": str(token_path)}):
            resolved = get_github_models_token_path()
            self.assertEqual(resolved, token_path)


if __name__ == "__main__":
    unittest.main()
