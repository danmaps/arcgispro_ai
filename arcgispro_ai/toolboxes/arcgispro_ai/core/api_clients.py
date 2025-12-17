import requests
import time
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Union, Optional, Any

def log_message(message: str):
    """Log message to both console and ArcGIS Pro if available."""
    print(message)  # Always print to console for testing
    try:
        import arcpy
        arcpy.AddMessage(message)  # Log to ArcGIS Pro if available
    except ImportError:
        pass  # Not running in ArcGIS Pro

class APIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def make_request(self, endpoint: str, data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Make an API request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        # log_message(f"Making request to: {url}")
        for attempt in range(max_retries):
            try:
                # log_message(f"\nAttempt {attempt + 1} - Making request to: {url}")
                # log_message(f"Request data: {json.dumps(data, indent=2)}")
                
                response = requests.post(url, headers=self.headers, json=data, verify=False)
                
                # log_message(f"Response status: {response.status_code}")
                # if response.status_code != 200:
                #     log_message(f"Error response: {response.text}")
                
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    if hasattr(e.response, 'text'):
                        error_detail = e.response.text
                    else:
                        error_detail = str(e)
                    raise Exception(f"Failed to get response after {max_retries} retries. Status: {e.response.status_code if hasattr(e, 'response') else 'Unknown'}, Error: {error_detail}")
                # log_message(f"Retrying request due to: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def get_vision_completion(self, messages: List[Dict[str, Any]], max_tokens: int = 800) -> str:
        """Default multimodal handler falls back to text completion."""
        return self.get_completion(messages, response_format=None)

class OpenAIClient(APIClient):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, "https://api.openai.com/v1")
        self.model = model

        # log_message(f"OpenAI Client initialized with model: {self.model}")

    def get_available_models(self) -> List[str]:
        """Get list of available models from OpenAI API."""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self.headers, verify=False)
            response.raise_for_status()
            models = response.json()["data"]
            # Filter for chat models only
            chat_models = [
                model["id"] for model in models 
                if model["id"].startswith(("gpt-4", "gpt-3.5"))
            ]
            return sorted(chat_models)
        except Exception as e:
            # If API call fails, return default models
            return ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get completion from OpenAI API."""
        token_limit = max_tokens if max_tokens is not None else 4096
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": token_limit,
        }
        
        # For GPT-3.5-turbo, ensure we're using the latest model version
        if self.model == "gpt-3.5-turbo":
            data["model"] = "gpt-3.5-turbo-0125"
        
        # Only add response_format for GPT-4 models
        if response_format == "json_object" and self.model.startswith("gpt-4"):
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request("chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

    def get_vision_completion(self, messages: List[Dict[str, Any]], max_tokens: int = 800) -> str:
        """Handle image + text prompts for OpenAI models."""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        response = self.make_request("chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

class AzureOpenAIClient(APIClient):
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        super().__init__(api_key, endpoint)
        self.deployment_name = deployment_name
        self.headers["api-key"] = api_key

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get completion from Azure OpenAI API."""
        token_limit = max_tokens if max_tokens is not None else 5000
        data = {
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": token_limit,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request(f"openai/deployments/{self.deployment_name}/chat/completions?api-version=2023-12-01-preview", data)
        return response["choices"][0]["message"]["content"].strip()

    def get_vision_completion(self, messages: List[Dict[str, Any]], max_tokens: int = 800) -> str:
        """Handle image + text prompts for Azure OpenAI deployments."""
        data = {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        response = self.make_request(f"openai/deployments/{self.deployment_name}/chat/completions?api-version=2023-12-01-preview", data)
        return response["choices"][0]["message"]["content"].strip()

class ClaudeClient(APIClient):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key, "https://api.anthropic.com/v1")
        self.model = model
        self.headers["anthropic-version"] = "2023-06-01"
        self.headers["x-api-key"] = api_key

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get completion from Claude API."""
        token_limit = max_tokens if max_tokens is not None else 5000
        data = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": 0.5,
            "max_tokens": token_limit,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json"}
        
        response = self.make_request("messages", data)
        return response["content"][0]["text"].strip()

class DeepSeekClient(APIClient):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        super().__init__(api_key, "https://api.deepseek.com/v1")
        self.model = model

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get completion from DeepSeek API."""
        token_limit = max_tokens if max_tokens is not None else 5000
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": token_limit,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request("chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

class OpenRouterClient(APIClient):
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        super().__init__(api_key, "https://openrouter.ai/api/v1")
        self.model = model
        # Add OpenRouter-specific headers
        self.headers.update({
            "HTTP-Referer": "https://github.com/danmaps/arcgispro_ai",
            "X-Title": "ArcGIS Pro AI Toolbox"
        })

    def get_available_models(self) -> List[str]:
        """Get list of available models from OpenRouter API."""
        fallback_models = [
            "openai/gpt-4o-mini",
            "openai/o3-mini",
            "google/gemini-2.0-flash-exp:free",
            "anthropic/claude-3.5-sonnet",
            "deepseek/deepseek-chat"
        ]
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=15,
                verify=False
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            models_with_meta = []
            for model in data:
                model_id = model.get("id")
                if not model_id:
                    continue
                pricing = model.get("pricing", {})
                prompt_price = pricing.get("prompt")
                completion_price = pricing.get("completion")

                def _parse_price(value: Any) -> float:
                    if value in (None, "", "N/A"):
                        return float("inf")
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return float("inf")

                models_with_meta.append(
                    (
                        model_id,
                        _parse_price(prompt_price),
                        _parse_price(completion_price)
                    )
                )

            if not models_with_meta:
                return fallback_models

            # Sort with free models first, then by price, then alphabetically
            models_with_meta.sort(key=lambda item: (item[1], item[2], item[0]))
            return [model_id for model_id, _, _ in models_with_meta]
        except Exception:
            return fallback_models

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get completion from OpenRouter API."""
        token_limit = max_tokens if max_tokens is not None else 5000
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": token_limit,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request("chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

    def get_vision_completion(self, messages: List[Dict[str, Any]], max_tokens: int = 800) -> str:
        """Handle image + text prompts for OpenRouter models."""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        response = self.make_request("chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

class LocalLLMClient(APIClient):
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:8000"):
        super().__init__(api_key, base_url)
        # Local LLMs typically don't need auth
        self.headers = {"Content-Type": "application/json"}

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get completion from local LLM API."""
        token_limit = max_tokens if max_tokens is not None else 5000
        data = {
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": token_limit,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request("v1/chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

class WolframAlphaClient(APIClient):
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.wolframalpha.com/v2")
        self.headers = {"Content-Type": "application/x-www-form-urlencoded"}

    def get_result(self, query: str) -> str:
        """Get result from Wolfram Alpha API."""
        data = {"appid": self.api_key, "input": query}
        response = self.make_request("query", data)
        
        root = ET.fromstring(response.content)
        if root.attrib.get('success') == 'true':
            for pod in root.findall(".//pod[@title='Result']"):
                for subpod in pod.findall('subpod'):
                    plaintext = subpod.find('plaintext')
                    if plaintext is not None and plaintext.text:
                        return plaintext.text.strip()
            print("Result pod not found in the response")
        else:
            print("Query was not successful")
        raise Exception("Failed to get Wolfram Alpha response")

class GeoJSONUtils:
    @staticmethod
    def infer_geometry_type(geojson_data: Dict[str, Any]) -> str:
        """Infer geometry type from GeoJSON data."""
        geometry_type_map = {
            "Point": "Point",
            "MultiPoint": "Multipoint",
            "LineString": "Polyline",
            "MultiLineString": "Polyline",
            "Polygon": "Polygon",
            "MultiPolygon": "Polygon"
        }

        geometry_types = set()
        features = geojson_data.get("features", [geojson_data])
        
        for feature in features:
            geometry_type = feature["geometry"]["type"]
            geometry_types.add(geometry_type_map.get(geometry_type))

        if len(geometry_types) == 1:
            return geometry_types.pop()
        raise ValueError("Multiple geometry types found in GeoJSON")

def parse_numeric_value(text_value: str) -> Union[float, int]:
    """Parse numeric value from text."""
    if "," in text_value:
        text_value = text_value.replace(",", "")
    try:
        value = float(text_value)
        return int(value) if value.is_integer() else value
    except ValueError:
        raise ValueError(f"Could not parse numeric value from: {text_value}")

def get_env_var(var_name: str = "OPENROUTER_API_KEY") -> str:
    """Get environment variable value."""
    return os.environ.get(var_name, "")

def get_client(source: str, api_key: str, **kwargs) -> APIClient:
    """Get the appropriate AI client based on the source."""
    clients = {
        "OpenAI": lambda: OpenAIClient(api_key, model=kwargs.get('model', 'gpt-4')),
        "Azure OpenAI": lambda: AzureOpenAIClient(
            api_key,
            kwargs.get('endpoint', ''),
            kwargs.get('deployment_name', '')
        ),
        "Claude": lambda: ClaudeClient(api_key, model=kwargs.get('model', 'claude-3-opus-20240229')),
        "DeepSeek": lambda: DeepSeekClient(api_key, model=kwargs.get('model', 'deepseek-chat')),
        "OpenRouter": lambda: OpenRouterClient(api_key, model=kwargs.get('model', 'openai/gpt-4o-mini')),
        "Local LLM": lambda: LocalLLMClient(base_url=kwargs.get('base_url', 'http://localhost:8000')),
        "Wolfram Alpha": lambda: WolframAlphaClient(api_key)
    }
    
    if source not in clients:
        raise ValueError(f"Unsupported AI provider: {source}")
    
    return clients[source]() 
