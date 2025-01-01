import requests
import time
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Union, Optional, Any

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
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=data, verify=False)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to get response after {max_retries} retries: {str(e)}")
                print(f"Retrying request due to: {e}")
                time.sleep(1)

class OpenAIClient(APIClient):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, "https://api.openai.com/v1")
        self.model = model

    def get_completion(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
        """Get completion from OpenAI API."""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 5000,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request("chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

class AzureOpenAIClient(APIClient):
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        super().__init__(api_key, endpoint)
        self.deployment_name = deployment_name
        self.headers["api-key"] = api_key

    def get_completion(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
        """Get completion from Azure OpenAI API."""
        data = {
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 5000,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request(f"openai/deployments/{self.deployment_name}/chat/completions?api-version=2023-12-01-preview", data)
        return response["choices"][0]["message"]["content"].strip()

class ClaudeClient(APIClient):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key, "https://api.anthropic.com/v1")
        self.model = model
        self.headers["anthropic-version"] = "2023-06-01"
        self.headers["x-api-key"] = api_key

    def get_completion(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
        """Get completion from Claude API."""
        data = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": 0.5,
            "max_tokens": 5000,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json"}
        
        response = self.make_request("messages", data)
        return response["content"][0]["text"].strip()

class DeepSeekClient(APIClient):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        super().__init__(api_key, "https://api.deepseek.com/v1")
        self.model = model

    def get_completion(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
        """Get completion from DeepSeek API."""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 5000,
        }
        
        if response_format == "json_object":
            data["response_format"] = {"type": "json_object"}
        
        response = self.make_request("chat/completions", data)
        return response["choices"][0]["message"]["content"].strip()

class LocalLLMClient(APIClient):
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:8000"):
        super().__init__(api_key, base_url)
        # Local LLMs typically don't need auth
        self.headers = {"Content-Type": "application/json"}

    def get_completion(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
        """Get completion from local LLM API."""
        data = {
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 5000,
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

def get_env_var(var_name: str = "OPENAI_API_KEY") -> str:
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
        "Local LLM": lambda: LocalLLMClient(base_url=kwargs.get('base_url', 'http://localhost:8000')),
        "Wolfram Alpha": lambda: WolframAlphaClient(api_key)
    }
    
    if source not in clients:
        raise ValueError(f"Unsupported AI provider: {source}")
    
    return clients[source]() 