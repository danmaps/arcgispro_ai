# -*- coding: utf-8 -*-
"""
arcgispro_ai.pyt - Monolithic Python Toolbox
This file is auto-generated from arcgispro_ai_tools.pyt with inlined dependencies.
Do not edit directly - regenerate using build_monolithic_pyt.py
"""

import arcpy
import json
import os

# --- INLINED UTILITY CODE ---
import time
from datetime import datetime
import arcpy
import json
import os
import tempfile
import re
from typing import Dict, List, Union, Optional, Any

from .core.api_clients import get_client, GeoJSONUtils, parse_numeric_value, get_env_var

class MapUtils:
    @staticmethod
    def metadata_to_dict(metadata: Any) -> Dict[str, Any]:
        """Convert metadata object to dictionary."""
        if metadata is None:
            return "No metadata"

        extent_dict = {}
        for attr in ["XMax", "XMin", "YMax", "YMin"]:
            if hasattr(metadata, attr):
                extent_dict[attr.lower()] = getattr(metadata, attr)

        meta_dict = {
            "title": getattr(metadata, "title", "No title"),
            "tags": getattr(metadata, "tags", "No tags"),
            "summary": getattr(metadata, "summary", "No summary"),
            "description": getattr(metadata, "description", "No description"),
            "credits": getattr(metadata, "credits", "No credits"),
            "access_constraints": getattr(
                metadata, "accessConstraints", "No access constraints"
            ),
            "extent": extent_dict,
        }
        return meta_dict

    @staticmethod
    def expand_extent(extent: arcpy.Extent, factor: float = 1.1) -> arcpy.Extent:
        """Expand the given extent by a factor."""
        width = extent.XMax - extent.XMin
        height = extent.YMax - extent.YMin
        expansion = {"x": width * (factor - 1) / 2, "y": height * (factor - 1) / 2}
        return arcpy.Extent(
            extent.XMin - expansion["x"],
            extent.YMin - expansion["y"],
            extent.XMax + expansion["x"],
            extent.YMax + expansion["y"],
        )

class FeatureLayerUtils:
    @staticmethod
    def get_top_n_records(
        feature_class: str, fields: List[str], n: int
    ) -> List[Dict[str, Any]]:
        """Get top N records from a feature class."""
        records = []
        try:
            with arcpy.da.SearchCursor(feature_class, fields) as cursor:
                for i, row in enumerate(cursor):
                    if i >= n:
                        break
                    records.append({field: value for field, value in zip(fields, row)})
        except Exception as e:
            arcpy.AddError(f"Error retrieving records: {e}")
        return records

    @staticmethod
    def get_layer_info(input_layers: List[str]) -> Dict[str, Any]:
        """Get layer information including sample data."""
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        active_map = aprx.activeMap
        layers_info = {}

        if input_layers:
            for layer_name in input_layers:
                layer = active_map.listLayers(layer_name)[0]
                if layer.isFeatureLayer:
                    dataset = arcpy.Describe(layer.dataSource)
                    layers_info[layer.name] = {
                        "name": layer.name,
                        "path": layer.dataSource,
                        "data": FeatureLayerUtils.get_top_n_records(
                            layer, [f.name for f in dataset.fields], 5
                        ),
                    }
        return layers_info

def map_to_json(
    in_map: Optional[str] = None, output_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a JSON object containing information about a map."""
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        if not in_map:
            active_map = aprx.activeMap
            if not active_map:
                # Return an empty map structure instead of raising an error
                return {
                    "map_name": "No active map",
                    "title": "No map open",
                    "description": "No map is currently open in the project",
                    "spatial_reference": "",
                    "layers": [],
                    "properties": {}
                }
        else:
            maps = aprx.listMaps(in_map)
            if not maps:
                # Return an empty map structure if the named map doesn't exist
                return {
                    "map_name": f"Map '{in_map}' not found",
                    "title": "Map not found",
                    "description": f"No map named '{in_map}' found in the project",
                    "spatial_reference": "",
                    "layers": [],
                    "properties": {}
                }
            active_map = maps[0]
            
        map_info = {
            "map_name": active_map.name,
            "title": getattr(active_map, "title", "No title"),
            "description": getattr(active_map, "description", "No description"),
            "spatial_reference": active_map.spatialReference.name,
            "layers": [],
            "properties": {
                "rotation": getattr(active_map, "rotation", "No rotation"),
                "units": getattr(active_map, "units", "No units"),
                "time_enabled": getattr(active_map, "isTimeEnabled", "No time enabled"),
                "metadata": (
                    MapUtils.metadata_to_dict(active_map.metadata)
                    if hasattr(active_map, "metadata")
                    else "No metadata"
                ),
            },
        }
        
        for layer in active_map.listLayers():
            layer_info = {
                "name": layer.name,
                "feature_layer": layer.isFeatureLayer,
                "raster_layer": layer.isRasterLayer,
                "web_layer": layer.isWebLayer,
                "visible": layer.visible,
                "metadata": (
                    MapUtils.metadata_to_dict(layer.metadata)
                    if hasattr(layer, "metadata")
                    else "No metadata"
                ),
            }
    
            if layer.isFeatureLayer:
                dataset = arcpy.Describe(layer.dataSource)
                layer_info.update(
                    {
                        "spatial_reference": getattr(
                            dataset.spatialReference, "name", "Unknown"
                        ),
                        "extent": (
                            {
                                "xmin": dataset.extent.XMin,
                                "ymin": dataset.extent.YMin,
                                "xmax": dataset.extent.XMax,
                                "ymax": dataset.extent.YMax,
                            }
                            if hasattr(dataset, "extent")
                            else "Unknown"
                        ),
                        "fields": (
                            [
                                {
                                    "name": field.name,
                                    "type": field.type,
                                    "length": field.length,
                                }
                                for field in dataset.fields
                            ]
                            if hasattr(dataset, "fields")
                            else []
                        ),
                        "record_count": (
                            int(arcpy.management.GetCount(layer.dataSource)[0])
                            if dataset.dataType in ["FeatureClass", "Table"]
                            else 0
                        ),
                        "source_type": getattr(dataset, "dataType", "Unknown"),
                        "geometry_type": getattr(dataset, "shapeType", "Unknown"),
                        "renderer": (
                            layer.symbology.renderer.type
                            if hasattr(layer, "symbology")
                            and hasattr(layer.symbology, "renderer")
                            else "Unknown"
                        ),
                        "labeling": getattr(layer, "showLabels", "Unknown"),
                    }
                )
    
            map_info["layers"].append(layer_info)
    
        if output_json_path:
            with open(output_json_path, "w") as json_file:
                json.dump(map_info, json_file, indent=4)
            print(f"Map information has been written to {output_json_path}")
    
        return map_info
        
    except Exception as e:
        # Handle any other exceptions like not being in an ArcGIS Pro session
        return {
            "map_name": "Error accessing map",
            "title": "Error",
            "description": f"Error accessing map: {str(e)}",
            "spatial_reference": "",
            "layers": [],
            "properties": {}
        }

def create_feature_layer_from_geojson(
    geojson_data: Dict[str, Any], output_layer_name: str
) -> None:
    """Create a feature layer in ArcGIS Pro from GeoJSON data."""
    geometry_type = GeoJSONUtils.infer_geometry_type(geojson_data)

    # Create temporary file
    temp_dir = tempfile.gettempdir()
    geojson_file = os.path.join(temp_dir, f"{output_layer_name}.geojson")

    if os.path.exists(geojson_file):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        geojson_file = os.path.join(
            temp_dir, f"{output_layer_name}_{timestamp}.geojson"
        )

    with open(geojson_file, "w") as f:
        json.dump(geojson_data, f)
        arcpy.AddMessage(f"GeoJSON file saved to: {geojson_file}")

    time.sleep(1)
    arcpy.AddMessage(f"Converting GeoJSON to feature layer: {output_layer_name}")
    arcpy.conversion.JSONToFeatures(
        geojson_file, output_layer_name, geometry_type=geometry_type
    )

    aprx = arcpy.mp.ArcGISProject("CURRENT")
    if aprx.activeMap:
        active_map = aprx.activeMap
        output_layer_path = os.path.join(aprx.defaultGeodatabase, output_layer_name)
        arcpy.AddMessage(f"Adding layer from: {output_layer_path}")

        try:
            active_map.addDataFromPath(output_layer_path)
            layer = active_map.listLayers(output_layer_name)[0]
            desc = arcpy.Describe(layer.dataSource)

            if desc.extent:
                expanded_extent = MapUtils.expand_extent(desc.extent)
                active_view = aprx.activeView

                if hasattr(active_view, "camera"):
                    active_view.camera.setExtent(expanded_extent)
                    arcpy.AddMessage(
                        f"Layer '{output_layer_name}' added and extent set successfully."
                    )
                else:
                    arcpy.AddWarning(
                        "The active view is not a map view, unable to set the extent."
                    )
            else:
                arcpy.AddWarning(
                    f"Unable to get extent for layer '{output_layer_name}'."
                )
        except Exception as e:
            arcpy.AddError(f"Error processing layer: {str(e)}")
    else:
        arcpy.AddWarning("No active map found in the current project.")

def fetch_geojson(
    api_key: str, query: str, output_layer_name: str, source: str = "OpenAI", **kwargs
) -> Optional[Dict[str, Any]]:
    """Fetch GeoJSON data using AI response and create a feature layer."""
    client = get_client(source, api_key, **kwargs)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that always only returns valid GeoJSON in response to user queries. "
            "Don't use too many vertices. Include somewhat detailed geometry and any attributes you think might be relevant. "
            "Include factual information. If you want to communicate text to the user, you may use a message property "
            "in the attributes of geometry objects. For compatibility with ArcGIS Pro, avoid multiple geometry types "
            "in the GeoJSON output. For example, don't mix points and polygons.",
        },
        {"role": "user", "content": query},
    ]

    try:
        geojson_str = client.get_completion(messages, response_format="json_object")
        arcpy.AddMessage(f"Raw GeoJSON data:\n{geojson_str}")

        geojson_data = json.loads(geojson_str)
        create_feature_layer_from_geojson(geojson_data, output_layer_name)
        return geojson_data
    except Exception as e:
        arcpy.AddError(str(e))
        return None

def generate_python(
    api_key: str,
    map_info: Dict[str, Any],
    prompt: str,
    source: str = "OpenAI",
    explain: bool = False,
    **kwargs,
) -> Optional[str]:
    """Generate Python code using AI response."""
    if not prompt:
        return None

    client = get_client(source, api_key, **kwargs)

    # # Load prompts from config
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # config_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'config', 'prompts.json')
    # with open(config_path, 'r', encoding='utf-8') as f:
    #     prompts = json.load(f)    # define prompts directly instead of loading from config
    prompts = {
        "python": [
            {
                "role": "system",
                "content": "You are an AI assistant that writes Python code for ArcGIS Pro based on the user's map information and prompt. Only return markdown formatted python code. Avoid preamble like \"Here is a Python script that uses ArcPy to automate your workflow:\". Don't include anything after the code sample.",
            },
            {
                "role": "user",
                "content": "I have a map in ArcGIS Pro. I've gathered information about this map and will give it to you in JSON format containing information about the map, including the map name, title, description, spatial reference, layers, and properties. Based on this information, write Python code that performs the user specified task(s) in ArcGIS Pro. If you need to write a SQL query to select features based on an attribute, keep in mind that the ORDER BY clause is not supported in attribute queries for selection operations. ArcGIS Pro SQL expressions have specific limitations and capabilities that vary depending on the underlying data source (e.g., file geodatabases, enterprise geodatabases). Notably, the ORDER BY clause is not supported in selection queries, and aggregate functions like SUM and COUNT have limited use outside definition queries. Subqueries and certain string functions may also face restrictions. Additionally, field names and values must match exactly in terms of case sensitivity. If zooming to a layer, make sure to use a layer object with active_map.listLayers(<layer_name>)[0] NOT just a layer name string. Use a combination of SQL expressions and ArcPy functions, such as search cursors with sql_clause for sorting, to achieve desired results. Understanding these constraints is crucial for effective data querying and manipulation in ArcGIS Pro.",
            },
            {
                "role": "user",
                "content": "The Python code should: 1. Use the arcpy module. 2. Select the features using arcpy.management.SelectLayerByAttribute. 3. Zoom the map to the selected features using the arcpy.mapping module. 4. Use arcpy.AddMessage to communicate with the user. 5. If the user asks about features within a distance of another, use arcpy.SelectLayerByLocation_management in addition to arcpy.SelectLayerByAttribute_management. Provide only the complete Python code, without any additional text or comments.",
            },
            {
                "role": "user",
                "content": "Here are some example prompts and corresponding Python code:",
            },
            {                "role": "user",
                "content": 'Prompt: show me the largest polygon in states\nYour response:\n```python\nimport arcpy\n\n# User prompt: "show me the largest polygon in states"\n\n# Inputs\nlayer_name = "states"\nattribute_query = "shape_area = (SELECT MAX(shape_area) FROM states)"\n\n# Get the current project and the active view\naprx = arcpy.mp.ArcGISProject("CURRENT")\nactive_map = aprx.activeMap\nactive_view = aprx.activeView\n\n# Get the layer\nlayer = active_map.listLayers(layer_name)[0]\n\n# Select features based on the attribute query\narcpy.management.SelectLayerByAttribute(layer, "NEW_SELECTION", attribute_query)\n\n# Zoom to the extent of the selected features\nactive_view.camera.setExtent(active_view.getLayerExtent(layer))\n```',
            },
            {
                "role": "user",
                "content": 'Prompt: ca counties with the lowest population density\nYour response:\n```python\nimport arcpy\n\n# User prompt: "CA counties with the lowest population density"\n\n# Define the name of the counties layer\ncounties_fc = "counties"\n\n# Select counties in California\nquery = "STATE_ABBR = \'CA\'"\narcpy.management.SelectLayerByAttribute(counties_fc, "NEW_SELECTION", query)\n\n# Create a list to store county names and population densities\ncounty_density_list = []\n\n# Use a search cursor to get the names and population densities of the counties\nwith arcpy.da.SearchCursor(counties_fc, ["NAME", "POPULATION", "SQMI"]) as cursor:\n    for row in cursor:\n        population_density = row[1] / row[2] if row[2] > 0 else 0  # Avoid division by zero\n        county_density_list.append((row[0], population_density))\n\n# Sort the list by population density in ascending order and get the top 3\nlowest_density_counties = sorted(county_density_list, key=lambda x: x[1])[:3]\narcpy.AddMessage(f"Top 3 counties with the lowest population density: {lowest_density_counties}")\n\n# Create a query to select the lowest density counties\nlowest_density_names = [county[0] for county in lowest_density_counties]\nlowest_density_query = "NAME IN ({})".format(", ".join(["\'{}\'".format(name) for name in lowest_density_names]))\n\n# Select the lowest density counties\narcpy.management.SelectLayerByAttribute(counties_fc, "NEW_SELECTION", lowest_density_query + " AND " + query)\n\n# Zoom to the selected counties\naprx = arcpy.mp.ArcGISProject("CURRENT")\nactive_map = aprx.activeMap\nactive_view = aprx.activeView\nlayer = active_map.listLayers(counties_fc)[0]\nactive_view.camera.setExtent(active_view.getLayerExtent(layer))\narcpy.AddMessage("Zoomed to selected counties")\n```',
            },
        ],
        # "geojson": [
        #     {
        #         "role": "system",
        #         "content": "You are a helpful assistant that always only returns valid GeoJSON in response to user queries. Don't use too many vertices. Include somewhat detailed geometry and any attributes you think might be relevant. Include factual information. If you want to communicate text to the user, you may use a message property in the attributes of geometry objects. For compatibility with ArcGIS Pro, avoid multiple geometry types in the GeoJSON output. For example, don't mix points and polygons.",
        #     }
        # ],
        # "field": [
        #     {
        #         "role": "system",
        #         "content": "Respond breifly without any other information, not even a complete sentence. No need for any punctuation, decorations, or other verbage. This response is going to be in a field value.",
        #     }
        # ],
    }
    messages = prompts["python"] + [
        {"role": "system", "content": json.dumps(map_info, indent=4)},
        {"role": "user", "content": prompt},
    ]

    try:
        code_snippet = client.get_completion(messages)

        def trim_code_block(code_block: str) -> str:
            """Remove language identifier and triple backticks from code block."""
            code_block = re.sub(r"^```[a-zA-Z]*\n", "", code_block)
            code_block = re.sub(r"\n```$", "", code_block)
            return code_block.strip()

        code_snippet = trim_code_block(code_snippet)
        line = "<html><hr></html>"
        arcpy.AddMessage(line)
        arcpy.AddMessage(code_snippet)
        arcpy.AddMessage(line)

        return code_snippet
    except Exception as e:
        arcpy.AddError(str(e))
        return None

def add_ai_response_to_feature_layer(
    api_key: str,
    source: str,
    in_layer: str,
    out_layer: Optional[str],
    field_name: str,
    prompt_template: str,
    sql_query: Optional[str] = None,
    **kwargs,
) -> None:
    """Enrich feature layer with AI-generated responses."""
    if out_layer:
        arcpy.CopyFeatures_management(in_layer, out_layer)
        layer_to_use = out_layer
    else:
        layer_to_use = in_layer

    # Add new field for AI responses
    existing_fields = [f.name for f in arcpy.ListFields(layer_to_use)]
    if field_name in existing_fields:
        field_name += "_AI"

    arcpy.management.AddField(layer_to_use, field_name, "TEXT")

    def generate_ai_responses_for_feature_class(
        source: str,
        feature_class: str,
        field_name: str,
        prompt_template: str,
        sql_query: Optional[str] = None,
    ) -> None:
        """Generate AI responses for features and update the field."""
        desc = arcpy.Describe(feature_class)
        oid_field_name = desc.OIDFieldName
        fields = [field.name for field in arcpy.ListFields(feature_class)]

        # Store prompts and their corresponding OIDs
        prompts_dict = {}
        with arcpy.da.SearchCursor(feature_class, fields[:-1], sql_query) as cursor:
            for row in cursor:
                row_dict = {field: value for field, value in zip(fields[:-1], row)}
                formatted_prompt = prompt_template.format(**row_dict)
                oid = row_dict[oid_field_name]
                prompts_dict[oid] = formatted_prompt

        if prompts_dict:
            sample_oid, sample_prompt = next(iter(prompts_dict.items()))
            arcpy.AddMessage(f"{oid_field_name} {sample_oid}: {sample_prompt}")
        else:
            arcpy.AddMessage("prompts_dict is empty.")

        # Get AI responses
        client = get_client(source, api_key, **kwargs)
        responses_dict = {}

        if source == "Wolfram Alpha":
            for oid, prompt in prompts_dict.items():
                responses_dict[oid] = client.get_result(prompt)
        else:
            role = "Respond without any other information, not even a complete sentence. No need for any other decoration or verbage."
            for oid, prompt in prompts_dict.items():
                messages = [
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt},
                ]
                responses_dict[oid] = client.get_completion(messages)

        # Update feature class with responses
        with arcpy.da.UpdateCursor(
            feature_class, [oid_field_name, field_name]
        ) as cursor:
            for row in cursor:
                oid = row[0]
                if oid in responses_dict:
                    row[1] = responses_dict[oid]
                    cursor.updateRow(row)

    generate_ai_responses_for_feature_class(
        source, layer_to_use, field_name, prompt_template, sql_query
    )

import requests
import xml.etree.ElementTree as ET

def log_message(message: str):
    """Log message to both console and ArcGIS Pro if available."""
    print(message)  # Always print to console for testing
    try:
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

    def get_completion(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> str:
        """Get completion from OpenAI API."""
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 4096,
        }
        
        # For GPT-3.5-turbo, ensure we're using the latest model version
        if self.model == "gpt-3.5-turbo":
            data["model"] = "gpt-3.5-turbo-0125"
        
        # Only add response_format for GPT-4 models
        if response_format == "json_object" and self.model.startswith("gpt-4"):
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
# --- END INLINED UTILITY CODE ---

def update_model_parameters(source: str, parameters: list, current_model: str = None) -> None:
    """Update model parameters based on the selected source.
    
    Args:
        source: The selected AI source (e.g., 'OpenAI', 'Azure OpenAI', etc.)
        parameters: List of arcpy.Parameter objects [source, model, endpoint, deployment]
        current_model: Currently selected model, if any
    """
    model_configs = {
        "Azure OpenAI": {
            "models": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
            "default": "gpt-4o-mini",
            "endpoint": True,
            "deployment": True
        },
        "OpenAI": {
            "models": [],  # Will be populated dynamically
            "default": "gpt-4o-mini",
            "endpoint": False,
            "deployment": False
        },
        "Claude": {
            "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
            "default": "claude-3-opus-20240229",
            "endpoint": False,
            "deployment": False
        },
        "DeepSeek": {
            "models": ["deepseek-chat", "deepseek-coder"],
            "default": "deepseek-chat",
            "endpoint": False,
            "deployment": False
        },
        "Local LLM": {
            "models": [],
            "default": None,
            "endpoint": True,
            "deployment": False,
            "endpoint_value": "http://localhost:8000"
        }
    }

    config = model_configs.get(source, {})
    if not config:
        return

    # If OpenAI is selected, fetch available models
    if source == "OpenAI":
        try:
            api_key = get_env_var("OPENAI_API_KEY")
            client = OpenAIClient(api_key)
            config["models"] = client.get_available_models()
            
        except Exception:
            # If fetching fails, use default hardcoded models
            config["models"] = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]

    # Model parameter
    parameters[1].enabled = bool(config["models"])
    if config["models"]:
        parameters[1].filter.type = "ValueList"
        parameters[1].filter.list = config["models"]
        if not current_model or current_model not in config["models"]:
            parameters[1].value = config["default"]

    # Endpoint parameter
    parameters[2].enabled = config["endpoint"]
    if config.get("endpoint_value"):
        parameters[2].value = config["endpoint_value"]

    # Deployment parameter
    parameters[3].enabled = config["deployment"]

class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file). This is important because the tools can be called like
        `arcpy.mytoolbox.mytool()` where mytoolbox is the name of the .pyt
        file and mytool is the name of the class in the toolbox."""
        self.label = "ai"
        self.alias = "ai"

        # List of tool classes associated with this toolbox
        self.tools = [FeatureLayer,
                      Field,
                      GetMapInfo,
                      Python,
                      ConvertTextToNumeric,
                      GenerateTool]

class FeatureLayer(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Create AI Feature Layer"
        self.description = "Create AI Feature Layer"
        self.params = arcpy.GetParameterInfo()
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define the tool parameters."""
        source = arcpy.Parameter(
            displayName="Source",
            name="source",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        source.filter.type = "ValueList"
        source.filter.list = ["OpenAI", "Azure OpenAI", "Claude", "DeepSeek", "Local LLM"]
        source.value = "OpenAI"

        model = arcpy.Parameter(
            displayName="Model",
            name="model",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        model.value = ""
        model.enabled = True

        endpoint = arcpy.Parameter(
            displayName="Endpoint",
            name="endpoint",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        endpoint.value = ""
        endpoint.enabled = False

        deployment = arcpy.Parameter(
            displayName="Deployment Name",
            name="deployment",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        deployment.value = ""
        deployment.enabled = False

        prompt = arcpy.Parameter(
            displayName="Prompt",
            name="prompt",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        prompt.description = "The prompt to generate a feature layer for. Try literally anything you can think of."

        output_layer = arcpy.Parameter(
            displayName="Output Layer",
            name="output_layer",    
            datatype="GPFeatureLayer",
            parameterType="Derived",
            direction="Output",
        )
        output_layer.description = "The output feature layer."

        params = [source, model, endpoint, deployment, prompt, output_layer]
        return params

    def isLicensed(self):   
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        source = parameters[0].value
        current_model = parameters[1].value

        update_model_parameters(source, parameters, current_model)

        import re
        parameters[5].value = re.sub(r'[^\w]', '_', parameters[4].valueAsText)
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        source = parameters[0].valueAsText
        model = parameters[1].valueAsText
        endpoint = parameters[2].valueAsText
        deployment = parameters[3].valueAsText
        prompt = parameters[4].valueAsText
        output_layer_name = parameters[5].valueAsText

        # Get the appropriate API key
        api_key_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Azure OpenAI": "AZURE_OPENAI_API_KEY",
            "Claude": "ANTHROPIC_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "Local LLM": None
        }
        api_key = get_env_var(api_key_map.get(source, "OPENAI_API_KEY"))

        # Fetch GeoJSON and create feature layer
        try:
            kwargs = {}
            if model:
                kwargs["model"] = model
            if endpoint:
                kwargs["endpoint"] = endpoint
            if deployment:
                kwargs["deployment_name"] = deployment

            geojson_data = fetch_geojson(api_key, prompt, output_layer_name, source, **kwargs)
            if not geojson_data:
                raise ValueError("Received empty GeoJSON data.")
        except Exception as e:
            arcpy.AddError(f"Error fetching GeoJSON: {str(e)}")
            return

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

class Field(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Field"
        self.description = "Adds a new attribute field to feature layers with AI-generated text. It uses AI APIs to create responses based on user-defined prompts that can reference existing attributes."
        self.getParameterInfo()

    def getParameterInfo(self):
        """Define the tool parameters."""
        source = arcpy.Parameter(
            displayName="Source",
            name="source",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        source.filter.type = "ValueList"
        source.filter.list = ["OpenAI", "Azure OpenAI", "Claude", "DeepSeek", "Local LLM", "Wolfram Alpha"]
        source.value = "OpenAI"

        model = arcpy.Parameter(
            displayName="Model",
            name="model",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        model.value = ""
        model.enabled = True

        endpoint = arcpy.Parameter(
            displayName="Endpoint",
            name="endpoint",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        endpoint.value = ""
        endpoint.enabled = False

        deployment = arcpy.Parameter(
            displayName="Deployment Name",
            name="deployment",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        deployment.value = ""
        deployment.enabled = False

        in_layer = arcpy.Parameter(
            displayName="Input Layer",
            name="in_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        out_layer = arcpy.Parameter(
            displayName="Output Layer",
            name="out_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output"
        )

        field_name = arcpy.Parameter(
            displayName="Field Name",
            name="field_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        prompt = arcpy.Parameter(
            displayName="Prompt",
            name="prompt",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        sql = arcpy.Parameter(
            displayName="SQL Query",
            name="sql",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )

        params = [source, model, endpoint, deployment, in_layer, out_layer, field_name, prompt, sql]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        source = parameters[0].value
        current_model = parameters[1].value

        update_model_parameters(source, parameters, current_model)

        if source == "Wolfram Alpha":
            parameters[1].enabled = False
            parameters[2].enabled = False
            parameters[3].enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        source = parameters[0].valueAsText
        model = parameters[1].valueAsText
        endpoint = parameters[2].valueAsText
        deployment = parameters[3].valueAsText
        in_layer = parameters[4].valueAsText
        out_layer = parameters[5].valueAsText
        field_name = parameters[6].valueAsText
        prompt = parameters[7].valueAsText
        sql = parameters[8].valueAsText

        # Get the appropriate API key
        api_key_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Azure OpenAI": "AZURE_OPENAI_API_KEY",
            "Claude": "ANTHROPIC_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "Local LLM": None,
            "Wolfram Alpha": "WOLFRAM_ALPHA_API_KEY"
        }
        api_key = get_env_var(api_key_map.get(source, "OPENAI_API_KEY"))

        # Add AI response to feature layer
        kwargs = {}
        if model:
            kwargs["model"] = model
        if endpoint:
            kwargs["endpoint"] = endpoint
        if deployment:
            kwargs["deployment_name"] = deployment

        add_ai_response_to_feature_layer(
            api_key,
            source,
            in_layer,
            out_layer,
            field_name,
            prompt,
            sql,
            **kwargs
        )

        arcpy.AddMessage(f"{out_layer} created with AI-generated field {field_name}.")
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

class GetMapInfo(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Get Map Info"
        self.description = "Get Map Info"
        self.params = arcpy.GetParameterInfo()
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define the tool parameters."""
        in_map = arcpy.Parameter(
            displayName="Map",
            name="map",
            datatype="Map",
            parameterType="Optional",
            direction="Input",
        )

        in_map.description = "The map to get info from."

        output_json_path = arcpy.Parameter(
            displayName="Output JSON Path",
            name="output_json_path",
            datatype="GPString",
            parameterType="Required",
            direction="Output",
        )

        output_json_path.description = "The path to the output JSON file."

        params = [in_map, output_json_path]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        if parameters[0].value:
            # If a map is selected, set the output path to the project home folder with the map name and json extension
            parameters[1].value = os.path.join(os.path.dirname(aprx.homeFolder), os.path.basename(parameters[0].valueAsText) + ".json")
        else:
            # otherwise, set the output path to the current project home folder with the current map name and json extension
            parameters[1].value = os.path.join(os.path.dirname(aprx.homeFolder), os.path.basename(aprx.activeMap.name) + ".json")
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        in_map = parameters[0].valueAsText
        out_json = parameters[1].valueAsText
        map_info = map_to_json(in_map)
        with open(out_json, "w") as f:
            json.dump(map_info, f, indent=4)

        arcpy.AddMessage(f"Map info saved to {out_json}")
        return
    
class Python(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Python"
        self.description = "Python"
        self.params = arcpy.GetParameterInfo()
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define the tool parameters."""
        source = arcpy.Parameter(
            displayName="Source",
            name="source",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        source.filter.type = "ValueList"
        source.filter.list = ["OpenAI", "Azure OpenAI", "Claude", "DeepSeek", "Local LLM"]
        source.value = "OpenAI"

        model = arcpy.Parameter(
            displayName="Model",
            name="model",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        model.value = ""
        model.enabled = True

        endpoint = arcpy.Parameter(
            displayName="Endpoint",
            name="endpoint",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        endpoint.value = ""
        endpoint.enabled = False

        deployment = arcpy.Parameter(
            displayName="Deployment Name",
            name="deployment",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        deployment.value = ""
        deployment.enabled = False

        layers = arcpy.Parameter(
            displayName="Layers for context",
            name="layers_for_context",
            datatype="GPFeatureRecordSetLayer",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
        )

        prompt = arcpy.Parameter(
            displayName="Prompt",
            name="prompt",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        # Temporarily disabled eval parameter
        # eval = arcpy.Parameter(
        #     displayName="Execute Generated Code",
        #     name="eval",
        #     datatype="Boolean",
        #     parameterType="Required",
        #     direction="Input",
        # )
        # eval.value = False

        context = arcpy.Parameter(
            displayName="Context (this will be passed to the AI)",
            name="context",
            datatype="GPstring",
            parameterType="Optional",
            direction="Input",
            category="Context",
        )
        context.controlCLSID = '{E5456E51-0C41-4797-9EE4-5269820C6F0E}'

        params = [source, model, endpoint, deployment, layers, prompt, context]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        source = parameters[0].value
        current_model = parameters[1].value

        update_model_parameters(source, parameters, current_model)

        layers = parameters[4].values
        # combine map and layer data into one JSON
        # only do this if context is empty
        if not parameters[6].valueAsText or parameters[6].valueAsText.strip() == "":
            context_json = {
                "map": map_to_json(), 
                "layers": FeatureLayerUtils.get_layer_info(layers)
            }
            parameters[6].value = json.dumps(context_json, indent=2)
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        source = parameters[0].valueAsText
        model = parameters[1].valueAsText
        endpoint = parameters[2].valueAsText
        deployment = parameters[3].valueAsText
        layers = parameters[4].values
        prompt = parameters[5].value
        derived_context = parameters[6].value

        # Get the appropriate API key
        api_key_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Azure OpenAI": "AZURE_OPENAI_API_KEY",
            "Claude": "ANTHROPIC_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "Local LLM": None
        }
        api_key = get_env_var(api_key_map.get(source, "OPENAI_API_KEY"))

        # Generate Python code
        kwargs = {}
        if model:
            kwargs["model"] = model
        if endpoint:
            kwargs["endpoint"] = endpoint
        if deployment:
            kwargs["deployment_name"] = deployment

        # If derived_context is None, create a default context
        if derived_context is None:
            context_json = {
                "map": map_to_json(), 
                "layers": FeatureLayerUtils.get_layer_info(layers) if layers else []
            }
        else:
            context_json = json.loads(derived_context)

        try:
            code_snippet = generate_python(
                api_key,
                context_json,
                prompt.strip(),
                source,
                **kwargs
            )

            # if eval == True:
            #     try:
            #         if code_snippet:
            #             arcpy.AddMessage("Executing code... fingers crossed!")
            #             exec(code_snippet)
            #         else:
            #             raise Exception("No code generated. Please try again.")
            #     except AttributeError as e:
            #         arcpy.AddError(f"{e}\n\nMake sure a map view is active.")
            #     except Exception as e:
            #         arcpy.AddError(
            #             f"{e}\n\nThe code may be invalid. Please check the code and try again."
            #         )

        except Exception as e:
            if "429" in str(e):
                arcpy.AddError(
                    "Rate limit exceeded. Please try:\n"
                    "1. Wait a minute and try again\n"
                    "2. Use a different model (e.g. GPT-3.5 instead of GPT-4)\n"
                    "3. Use a different provider (e.g. Claude or DeepSeek)\n"
                    "4. Check your API key's rate limits and usage"
                )
            else:
                arcpy.AddError(str(e))
            return

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return
    

class ConvertTextToNumeric(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Convert Text to Numeric"
        self.description = "Clean up numbers stored in inconsistent text formats into a numeric field."
        self.params = arcpy.GetParameterInfo()
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define the tool parameters."""
        source = arcpy.Parameter(
            displayName="Source",
            name="source",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        source.filter.type = "ValueList"
        source.filter.list = ["OpenAI", "Azure OpenAI", "Claude", "DeepSeek", "Local LLM"]
        source.value = "OpenAI"

        model = arcpy.Parameter(
            displayName="Model",
            name="model",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        model.value = ""
        model.enabled = True

        endpoint = arcpy.Parameter(
            displayName="Endpoint",
            name="endpoint",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        endpoint.value = ""
        endpoint.enabled = False

        deployment = arcpy.Parameter(
            displayName="Deployment Name",
            name="deployment",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        deployment.value = ""
        deployment.enabled = False

        in_layer = arcpy.Parameter(
            displayName="Input Layer",
            name="in_layer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        field = arcpy.Parameter(
            displayName="Field",
            name="field",
            datatype="Field",
            parameterType="Required",
            direction="Input",
        )

        params = [source, model, endpoint, deployment, in_layer, field]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        source = parameters[0].value
        current_model = parameters[1].value

        update_model_parameters(source, parameters, current_model)
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        source = parameters[0].valueAsText
        model = parameters[1].valueAsText
        endpoint = parameters[2].valueAsText
        deployment = parameters[3].valueAsText
        in_layer = parameters[4].valueAsText
        field = parameters[5].valueAsText

        # Get the appropriate API key
        api_key_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Azure OpenAI": "AZURE_OPENAI_API_KEY",
            "Claude": "ANTHROPIC_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "Local LLM": None
        }
        api_key = get_env_var(api_key_map.get(source, "OPENAI_API_KEY"))

        # Get the field values
        field_values = []
        with arcpy.da.SearchCursor(in_layer, [field]) as cursor:
            for row in cursor:
                field_values.append(row[0])        # Convert the entire series using the selected AI provider
        kwargs = {}
        if model:
            kwargs["model"] = model
        if endpoint:
            kwargs["endpoint"] = endpoint
        if deployment:
            kwargs["deployment_name"] = deployment

        converted_values = get_client(source, api_key, **kwargs).convert_series_to_numeric(field_values)

        # Add a new field to store the converted numeric values
        field_name_new = f"{field}_numeric"
        arcpy.AddField_management(in_layer, field_name_new, "DOUBLE")

        # Update the new field with the converted values
        with arcpy.da.UpdateCursor(in_layer, [field, field_name_new]) as cursor:
            for i, row in enumerate(cursor):
                row[1] = converted_values[i]
                cursor.updateRow(row)

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

class GenerateTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Generate Tool"
        self.description = "Transforms a Python code sample or natural language prompt into a fully functional, documented, and parameterized Python toolbox (.pyt)"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define the tool parameters."""
        source = arcpy.Parameter(
            displayName="Source",
            name="source",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        source.filter.type = "ValueList"
        source.filter.list = ["OpenAI", "Azure OpenAI", "Claude", "DeepSeek", "Local LLM"]
        source.value = "OpenAI"

        model = arcpy.Parameter(
            displayName="Model",
            name="model",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        model.value = ""
        model.enabled = True

        endpoint = arcpy.Parameter(
            displayName="Endpoint",
            name="endpoint",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        endpoint.value = ""
        endpoint.enabled = False

        deployment = arcpy.Parameter(
            displayName="Deployment Name",
            name="deployment",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        deployment.value = ""
        deployment.enabled = False

        input_type = arcpy.Parameter(
            displayName="Input Type",
            name="input_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        input_type.filter.type = "ValueList"
        input_type.filter.list = ["Natural Language Prompt", "Python Code"]
        input_type.value = "Natural Language Prompt"

        input_code = arcpy.Parameter(
            displayName="Python Code",
            name="input_code",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        input_code.enabled = False
        input_code.controlCLSID = '{E5456E51-0C41-4797-9EE4-5269820C6F0E}'

        input_prompt = arcpy.Parameter(
            displayName="Natural Language Prompt",
            name="input_prompt",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        input_prompt.enabled = True

        toolbox_name = arcpy.Parameter(
            displayName="Toolbox Name",
            name="toolbox_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        toolbox_name.value = "MyToolbox"

        tool_name = arcpy.Parameter(
            displayName="Tool Name",
            name="tool_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        tool_name.value = "MyTool"

        output_path = arcpy.Parameter(
            displayName="Output Path",
            name="output_path",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )
        # Try to set current project home folder as default
        try:
            aprx = arcpy.mp.ArcGISProject("CURRENT")
            if aprx and aprx.homeFolder:
                output_path.value = aprx.homeFolder
        except:
            # If we can't access the current project, leave it blank
            pass

        advanced_mode = arcpy.Parameter(
            displayName="Advanced Mode",
            name="advanced_mode",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Parameter Details",
        )
        advanced_mode.value = False

        # Parameter Details section (only shown in advanced mode)
        parameter_definition = arcpy.Parameter(
            displayName="Parameter Definition (JSON)",
            name="parameter_definition",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            category="Parameter Details",
        )
        parameter_definition.enabled = False
        parameter_definition.controlCLSID = '{E5456E51-0C41-4797-9EE4-5269820C6F0E}'
        parameter_definition.value = '''{
  "parameters": [
    {
      "name": "example_param",
      "displayName": "Example Parameter",
      "datatype": "GPString",
      "parameterType": "Required",
      "direction": "Input"
    }
  ]
}'''

        output_toolbox = arcpy.Parameter(
            displayName="Output Toolbox",
            name="output_toolbox",
            datatype="DEFile",
            parameterType="Derived",
            direction="Output",
        )
        
        params = [source, model, endpoint, deployment, 
                  input_type, input_code, input_prompt, 
                  toolbox_name, tool_name, output_path, 
                  advanced_mode, parameter_definition, output_toolbox]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        source = parameters[0].value
        current_model = parameters[1].value
        
        # Update model parameters based on selected source
        update_model_parameters(source, parameters, current_model)
        
        # Handle input type change
        if parameters[4].altered:
            if parameters[4].value == "Python Code":
                parameters[5].enabled = True
                parameters[6].enabled = False
            else:  # Natural Language Prompt
                parameters[5].enabled = False
                parameters[6].enabled = True
        
        # Handle advanced mode toggle
        if parameters[10].altered:
            parameters[11].enabled = parameters[10].value
            
        # Set output file path
        if parameters[9].value and parameters[7].value:
            output_file = os.path.join(parameters[9].valueAsText, f"{parameters[7].valueAsText}.pyt")
            parameters[12].value = output_file
            
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        # Validate input based on input type
        if parameters[4].value == "Python Code" and not parameters[5].value:
            parameters[5].setErrorMessage("Python code is required when Input Type is set to 'Python Code'")
        elif parameters[4].value == "Natural Language Prompt" and not parameters[6].value:
            parameters[6].setErrorMessage("Natural language prompt is required when Input Type is set to 'Natural Language Prompt'")
            
        # Validate toolbox name
        if parameters[7].value:
            if not parameters[7].valueAsText.isalnum():
                parameters[7].setWarningMessage("Toolbox name should be alphanumeric for best compatibility")
                
        # Validate tool name
        if parameters[8].value:
            if not parameters[8].valueAsText.isalnum():
                parameters[8].setWarningMessage("Tool name should be alphanumeric for best compatibility")
                
        # Validate parameter definition JSON if advanced mode is enabled
        if parameters[10].value and parameters[11].value:
            try:
                json.loads(parameters[11].valueAsText)
            except json.JSONDecodeError:
                parameters[11].setErrorMessage("Parameter definition must be valid JSON")
                
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        source = parameters[0].valueAsText
        model = parameters[1].valueAsText
        endpoint = parameters[2].valueAsText
        deployment = parameters[3].valueAsText
        input_type = parameters[4].valueAsText
        input_code = parameters[5].valueAsText
        input_prompt = parameters[6].valueAsText
        toolbox_name = parameters[7].valueAsText
        tool_name = parameters[8].valueAsText
        output_path = parameters[9].valueAsText
        advanced_mode = parameters[10].value
        parameter_definition = parameters[11].valueAsText
        output_toolbox = parameters[12].valueAsText
        
        # Get the appropriate API key
        api_key_map = {
            "OpenAI": "OPENAI_API_KEY",
            "Azure OpenAI": "AZURE_OPENAI_API_KEY",
            "Claude": "ANTHROPIC_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "Local LLM": None
        }
        api_key = get_env_var(api_key_map.get(source, "OPENAI_API_KEY"))
        
        # Set up parameters for API calls
        kwargs = {}
        if model:
            kwargs["model"] = model
        if endpoint:
            kwargs["endpoint"] = endpoint
        if deployment:
            kwargs["deployment_name"] = deployment
            
        # Step 1: Get the Python code to convert to a tool
        python_code = ""
        if input_type == "Python Code":
            python_code = input_code
            arcpy.AddMessage("Using provided Python code as input")
        else:  # Natural Language Prompt
            arcpy.AddMessage("Generating Python code from natural language prompt...")
            # Create a minimal context - no map info needed for this tool
            context_json = {"layers": []}
            try:
                # Use the existing generate_python function to convert prompt to code
                python_code = generate_python(
                    api_key,
                    context_json,
                    input_prompt.strip(),
                    source,
                    **kwargs
                )
                
                if not python_code:
                    arcpy.AddError("Failed to generate Python code from prompt. Please try again.")
                    return
                    
                arcpy.AddMessage("Successfully generated Python code from prompt")
            except Exception as e:
                arcpy.AddError(f"Error generating Python code: {str(e)}")
                return
                
        # Step 2: Generate the toolbox structure
        arcpy.AddMessage("Generating toolbox structure...")
        
        # Parse parameters if in advanced mode
        param_structure = None
        if advanced_mode and parameter_definition:
            try:
                param_structure = json.loads(parameter_definition)
                arcpy.AddMessage("Using user-defined parameter structure")
            except json.JSONDecodeError:
                arcpy.AddError("Failed to parse parameter definition JSON. Using auto-inference instead.")
                param_structure = None
        
        # Create prompt for generating the PYT file
        prompt_text = f"""Convert the following Python code into a fully functional ArcGIS Python Toolbox (.pyt) file.
        
Toolbox Name: {toolbox_name}
Tool Name: {tool_name}

Python Code to Convert:
```python
{python_code}
```

Requirements:
1. Create a valid .pyt file structure with Toolbox class and a Tool class named {tool_name}
2. Automatically infer appropriate parameters from the code
3. Include proper documentation and docstrings
4. Implement all required methods: __init__, getParameterInfo, isLicensed, updateParameters, updateMessages, execute, and postExecute
5. Follow ArcGIS Pro Python Toolbox best practices

"""        
        if param_structure:
            prompt_text += f"Use the following parameter structure: {json.dumps(param_structure, indent=2)}"
            
        arcpy.AddMessage("Generating toolbox code...")
        try:
            # Generate the toolbox code using the AI model
            client = get_client(source, api_key, **kwargs)
            
            # Format the prompt as messages for the AI client
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates ArcGIS Python Toolbox (.pyt) files."},
                {"role": "user", "content": prompt_text}
            ]
            
            response = client.get_completion(messages)
            
            if not response:
                arcpy.AddError("Failed to generate toolbox code. Please try again.")
                return
                
            # Extract the Python code from the response
            toolbox_code = response
            if "```python" in response:
                toolbox_code = response.split("```python")[1].split("```")[0].strip()
            
            # Write the toolbox code to the output file
            with open(output_toolbox, "w") as f:
                f.write(toolbox_code)
                
            arcpy.AddMessage(f"Successfully created toolbox at: {output_toolbox}")
            
            # Validate the generated toolbox
            arcpy.AddMessage("Validating toolbox...")
            try:
                # Import the toolbox to check for syntax errors
                import importlib.util
                spec = importlib.util.spec_from_file_location("generated_toolbox", output_toolbox)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                arcpy.AddMessage("Toolbox validation successful!")
            except Exception as e:
                arcpy.AddWarning(f"Toolbox validation warning: {str(e)}")
                arcpy.AddWarning("The toolbox may require manual adjustments.")
                
        except Exception as e:
            arcpy.AddError(f"Error generating toolbox: {str(e)}")
            return
            
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return