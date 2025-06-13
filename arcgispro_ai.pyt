# -*- coding: utf-8 -*-
"""
arcgispro_ai.pyt - Monolithic Python Toolbox
This file is auto-generated. Do not edit directly.
"""

import arcpy
import os
import json
# ...add any other standard library imports as needed...

# --- BEGIN INLINED UTILITY CODE ---
import time
from datetime import datetime
import arcpy
import json
import os
import tempfile
import re
from typing import Dict, List, Union, Optional, Any

class OpenAIClient:
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.endpoint = kwargs.get("endpoint", "https://api.openai.com/v1")
        self.model = kwargs.get("model", "gpt-4")
        self.deployment_name = kwargs.get("deployment_name", None)

    def get_completion(self, messages: List[Dict[str, str]], response_format: str = "text") -> Union[str, Dict[str, Any]]:
        """Get completion from OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if self.deployment_name:
            payload["deployment"] = self.deployment_name

        response = requests.post(f"{self.endpoint}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if response_format == "json_object":
            return data
        return data["choices"][0]["message"]["content"]

    def get_available_models(self) -> List[str]:
        """Fetch available models from OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.get(f"{self.endpoint}/models", headers=headers)
        response.raise_for_status()
        data = response.json()
        return [model["id"] for model in data["data"]]

class GeoJSONUtils:
    @staticmethod
    def infer_geometry_type(geojson_data: Dict[str, Any]) -> str:
        """Infer geometry type from GeoJSON data."""
        if "features" in geojson_data and geojson_data["features"]:
            geometry = geojson_data["features"][0].get("geometry", {})
            return geometry.get("type", "Unknown")
        return "Unknown"

def parse_numeric_value(value: str) -> Optional[float]:
    """Parse a numeric value from a string."""
    try:
        return float(value)
    except ValueError:
        return None

def get_env_var(var_name: str) -> str:
    """Get environment variable."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Environment variable {var_name} is not set.")
    return value

class MapUtils:
    @staticmethod
    def metadata_to_dict(metadata: Any) -> Dict[str, Any]:
        """Convert metadata object to dictionary."""
        if metadata is None:
            return "No metadata"

        extent_dict = {}
        for attr in ['XMax', 'XMin', 'YMax', 'YMin']:
            if hasattr(metadata, attr):
                extent_dict[attr.lower()] = getattr(metadata, attr)

        meta_dict = {
            "title": getattr(metadata, "title", "No title"),
            "tags": getattr(metadata, "tags", "No tags"),
            "summary": getattr(metadata, "summary", "No summary"),
            "description": getattr(metadata, "description", "No description"),
            "credits": getattr(metadata, "credits", "No credits"),
            "access_constraints": getattr(metadata, "accessConstraints", "No access constraints"),
            "extent": extent_dict
        }
        return meta_dict

    @staticmethod
    def expand_extent(extent: arcpy.Extent, factor: float = 1.1) -> arcpy.Extent:
        """Expand the given extent by a factor."""
        width = extent.XMax - extent.XMin
        height = extent.YMax - extent.YMin
        expansion = {
            'x': width * (factor - 1) / 2,
            'y': height * (factor - 1) / 2
        }
        return arcpy.Extent(
            extent.XMin - expansion['x'],
            extent.YMin - expansion['y'],
            extent.XMax + expansion['x'],
            extent.YMax + expansion['y']
        )

class FeatureLayerUtils:
    @staticmethod
    def get_top_n_records(feature_class: str, fields: List[str], n: int) -> List[Dict[str, Any]]:
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
                            layer,
                            [f.name for f in dataset.fields],
                            5
                        )
                    }
        return layers_info

def map_to_json(in_map: Optional[str] = None, output_json_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate a JSON object containing information about a map."""
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    if not in_map:
        active_map = aprx.activeMap
        if not active_map:
            raise ValueError("No active map found in the current project.")
    else:
        active_map = aprx.listMaps(in_map)[0]

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
            "metadata": MapUtils.metadata_to_dict(active_map.metadata) if hasattr(active_map, "metadata") else "No metadata",
        },
    }

    for layer in active_map.listLayers():
        layer_info = {
            "name": layer.name,
            "feature_layer": layer.isFeatureLayer,
            "raster_layer": layer.isRasterLayer,
            "web_layer": layer.isWebLayer,
            "visible": layer.visible,
            "metadata": MapUtils.metadata_to_dict(layer.metadata) if hasattr(layer, "metadata") else "No metadata",
        }

        if layer.isFeatureLayer:
            dataset = arcpy.Describe(layer.dataSource)
            layer_info.update({
                "spatial_reference": getattr(dataset.spatialReference, "name", "Unknown"),
                "extent": {
                    "xmin": dataset.extent.XMin,
                    "ymin": dataset.extent.YMin,
                    "xmax": dataset.extent.XMax,
                    "ymax": dataset.extent.YMax,
                } if hasattr(dataset, "extent") else "Unknown",
                "fields": [
                    {
                        "name": field.name,
                        "type": field.type,
                        "length": field.length,
                    }
                    for field in dataset.fields
                ] if hasattr(dataset, "fields") else [],
                "record_count": int(arcpy.management.GetCount(layer.dataSource)[0]) if dataset.dataType in ["FeatureClass", "Table"] else 0,
                "source_type": getattr(dataset, "dataType", "Unknown"),
                "geometry_type": getattr(dataset, "shapeType", "Unknown"),
                "renderer": layer.symbology.renderer.type if hasattr(layer, "symbology") and hasattr(layer.symbology, "renderer") else "Unknown",
                "labeling": getattr(layer, "showLabels", "Unknown"),
            })

        map_info["layers"].append(layer_info)

    if output_json_path:
        with open(output_json_path, "w") as json_file:
            json.dump(map_info, json_file, indent=4)
        print(f"Map information has been written to {output_json_path}")

    return map_info

def create_feature_layer_from_geojson(geojson_data: Dict[str, Any], output_layer_name: str) -> None:
    """Create a feature layer in ArcGIS Pro from GeoJSON data."""
    geometry_type = GeoJSONUtils.infer_geometry_type(geojson_data)
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    geojson_file = os.path.join(temp_dir, f"{output_layer_name}.geojson")
    
    if os.path.exists(geojson_file):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        geojson_file = os.path.join(temp_dir, f"{output_layer_name}_{timestamp}.geojson")
    
    with open(geojson_file, 'w') as f:
        json.dump(geojson_data, f)
        arcpy.AddMessage(f"GeoJSON file saved to: {geojson_file}")
    
    time.sleep(1)
    arcpy.AddMessage(f"Converting GeoJSON to feature layer: {output_layer_name}")
    arcpy.conversion.JSONToFeatures(geojson_file, output_layer_name, geometry_type=geometry_type)
    
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
                
                if hasattr(active_view, 'camera'):
                    active_view.camera.setExtent(expanded_extent)
                    arcpy.AddMessage(f"Layer '{output_layer_name}' added and extent set successfully.")
                else:
                    arcpy.AddWarning("The active view is not a map view, unable to set the extent.")
            else:
                arcpy.AddWarning(f"Unable to get extent for layer '{output_layer_name}'.")
        except Exception as e:
            arcpy.AddError(f"Error processing layer: {str(e)}")
    else:
        arcpy.AddWarning("No active map found in the current project.")

def fetch_geojson(api_key: str, query: str, output_layer_name: str, source: str = "OpenAI", **kwargs) -> Optional[Dict[str, Any]]:
    """Fetch GeoJSON data using AI response and create a feature layer."""
    client = get_client(source, api_key, **kwargs)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that always only returns valid GeoJSON in response to user queries. "
                      "Don't use too many vertices. Include somewhat detailed geometry and any attributes you think might be relevant. "
                      "Include factual information. If you want to communicate text to the user, you may use a message property "
                      "in the attributes of geometry objects. For compatibility with ArcGIS Pro, avoid multiple geometry types "
                      "in the GeoJSON output. For example, don't mix points and polygons."
        },
        {"role": "user", "content": query}
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

def generate_python(api_key: str, map_info: Dict[str, Any], prompt: str, source: str = "OpenAI", explain: bool = False, **kwargs) -> Optional[str]:
    """Generate Python code using AI response."""
    if not prompt:
        return None

    client = get_client(source, api_key, **kwargs)
    
    # Load prompts from config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'config', 'prompts.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    messages = prompts["python"] + [
        {"role": "system", "content": json.dumps(map_info, indent=4)},
        {"role": "user", "content": prompt},
    ]

    try:
        code_snippet = client.get_completion(messages)
        
        def trim_code_block(code_block: str) -> str:
            """Remove language identifier and triple backticks from code block."""
            code_block = re.sub(r'^```[a-zA-Z]*\n', '', code_block)
            code_block = re.sub(r'\n```$', '', code_block)
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
    **kwargs
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
        sql_query: Optional[str] = None
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
                    {"role": "user", "content": prompt}
                ]
                responses_dict[oid] = client.get_completion(messages)

        # Update feature class with responses
        with arcpy.da.UpdateCursor(feature_class, [oid_field_name, field_name]) as cursor:
            for row in cursor:
                oid = row[0]
                if oid in responses_dict:
                    row[1] = responses_dict[oid]
                    cursor.updateRow(row)

    generate_ai_responses_for_feature_class(source, layer_to_use, field_name, prompt_template, sql_query)


# --- END INLINED UTILITY CODE ---

# --- BEGIN TOOLBOX AND TOOL CLASSES ---
import arcpy
import json
import os
from arcgispro_ai.arcgispro_ai_utils import (
    MapUtils,
    FeatureLayerUtils,
    fetch_geojson,
    generate_python,
    add_ai_response_to_feature_layer,
    map_to_json
)
from arcgispro_ai.core.api_clients import (
    get_client,
    get_env_var,
    OpenAIClient
)

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
            if config["models"]:  # If we got models from the API
                config["default"] = "gpt-4" if "gpt-4" in config["models"] else config["models"][0]
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
                      ConvertTextToNumeric]

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
        if parameters[6].valueAsText == "":
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
                field_values.append(row[0])

        # Convert the entire series using the selected AI provider
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
# --- END TOOLBOX AND TOOL CLASSES ---
