import time
from datetime import datetime
import arcpy
import json
import os
import tempfile
import re
from typing import Dict, List, Union, Optional, Any

from .core.api_clients import (
    get_client,
    GeoJSONUtils,
    parse_numeric_value,
    get_env_var
)

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


