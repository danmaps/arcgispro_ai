import base64
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

    @staticmethod
    def extent_polygon(extent: arcpy.Extent, spatial_reference: Optional[arcpy.SpatialReference]) -> Optional[arcpy.Polygon]:
        """Create a polygon from an extent."""
        if not extent:
            return None

        array = arcpy.Array(
            [
                arcpy.Point(extent.XMin, extent.YMin),
                arcpy.Point(extent.XMin, extent.YMax),
                arcpy.Point(extent.XMax, extent.YMax),
                arcpy.Point(extent.XMax, extent.YMin),
                arcpy.Point(extent.XMin, extent.YMin),
            ]
        )
        return arcpy.Polygon(array, spatial_reference)


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

    @staticmethod
    def _get_attribute_fields(dataset, max_fields: int = 5) -> List[str]:
        """Return a small set of representative attribute fields."""
        skip_names = {"shape", "shape_area", "shape_length", "geometry"}
        allowed_types = {"OID", "Integer", "SmallInteger", "Double", "Single", "String"}
        fields: List[str] = []

        if not hasattr(dataset, "fields"):
            return fields

        for field in dataset.fields:
            if field.name.lower() in skip_names:
                continue
            if field.type not in allowed_types:
                continue
            fields.append(field.name)
            if len(fields) >= max_fields:
                break
        return fields

    @staticmethod
    def _simplify_geometry(geometry: arcpy.Geometry, tolerance: float) -> arcpy.Geometry:
        """Simplify geometry to reduce payload size."""
        if not geometry or tolerance <= 0:
            return geometry
        try:
            simplified = geometry.generalize(tolerance, True)
            return simplified if simplified else geometry
        except Exception:
            # If simplification fails, return the original geometry
            return geometry

    @staticmethod
    def _get_layer_features(
        layer: arcpy.mapping.Layer,
        extent_polygon: Optional[arcpy.Polygon],
        max_features: int = 50,
        simplify_ratio: float = 0.01,
    ) -> Dict[str, Any]:
        """Collect simplified GeoJSON-like data for the layer within the view extent."""
        dataset = arcpy.Describe(layer.dataSource)
        spatial_reference = getattr(dataset, "spatialReference", None)
        attribute_fields = FeatureLayerUtils._get_attribute_fields(dataset)
        layer_features: List[Dict[str, Any]] = []

        tolerance = 0
        if extent_polygon and extent_polygon.extent:
            tolerance = max(extent_polygon.extent.width, extent_polygon.extent.height) * simplify_ratio

        temp_layer_name = f"interpret_{re.sub('[^0-9A-Za-z]+', '_', layer.name)}_{int(time.time())}"
        temp_layer = None
        try:
            temp_layer = arcpy.management.MakeFeatureLayer(layer, temp_layer_name).getOutput(0)
            if extent_polygon:
                arcpy.management.SelectLayerByLocation(temp_layer, "INTERSECT", extent_polygon)

            cursor_fields = ["SHAPE@"] + attribute_fields
            with arcpy.da.SearchCursor(temp_layer, cursor_fields, spatial_reference=spatial_reference) as cursor:
                for i, row in enumerate(cursor):
                    if i >= max_features:
                        break
                    geometry = FeatureLayerUtils._simplify_geometry(row[0], tolerance)
                    attributes = {
                        attribute_fields[idx]: row[idx + 1]
                        for idx in range(len(attribute_fields))
                    }
                    layer_features.append(
                        {
                            "geometry": json.loads(geometry.JSON) if geometry else None,
                            "attributes": attributes,
                        }
                    )
        except Exception as exc:
            arcpy.AddWarning(f"Unable to sample features for {layer.name}: {exc}")
        finally:
            if temp_layer:
                try:
                    arcpy.management.Delete(temp_layer)
                except Exception:
                    pass

        return {
            "name": layer.name,
            "geometry_type": getattr(dataset, "shapeType", "Unknown"),
            "renderer": (
                layer.symbology.renderer.type
                if hasattr(layer, "symbology") and hasattr(layer.symbology, "renderer")
                else "Unknown"
            ),
            "spatial_reference": getattr(spatial_reference, "name", "Unknown"),
            "feature_cap": max_features,
            "field_sample": attribute_fields,
            "feature_sample_count": len(layer_features),
            "features": layer_features,
        }

    @staticmethod
    def capture_visible_layer_context(
        active_map: arcpy.mp.Map,
        view_extent: Optional[arcpy.Extent],
        max_features_per_layer: int = 50,
    ) -> List[Dict[str, Any]]:
        """Gather simplified GeoJSON for visible layers within the active view."""
        if not active_map:
            return []

        extent_polygon = None
        if view_extent:
            extent_polygon = MapUtils.extent_polygon(view_extent, active_map.spatialReference)

        layers_data: List[Dict[str, Any]] = []
        for layer in active_map.listLayers():
            if not getattr(layer, "visible", False) or not getattr(layer, "isFeatureLayer", False):
                continue
            layers_data.append(
                FeatureLayerUtils._get_layer_features(
                    layer,
                    extent_polygon,
                    max_features=max_features_per_layer,
                )
            )
        return layers_data


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
    api_key: str, query: str, output_layer_name: str, source: str = "OpenRouter", **kwargs
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
    source: str = "OpenRouter",
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


def capture_map_view_screenshot(
    view: Any, width: int = 1280, height: int = 720, resolution: int = 150
) -> Optional[Dict[str, Any]]:
    """Capture the active map view as a PNG and return metadata plus base64 content."""
    if not view or not hasattr(view, "exportToPNG"):
        arcpy.AddWarning("The active view does not support image export; skipping screenshot.")
        return None

    temp_dir = tempfile.mkdtemp()
    screenshot_path = os.path.join(temp_dir, "map_view.png")
    try:
        view.exportToPNG(screenshot_path, width=width, height=height, resolution=resolution)
        with open(screenshot_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return {
            "path": screenshot_path,
            "width": width,
            "height": height,
            "resolution": resolution,
            "base64": encoded,
        }
    except Exception as exc:
        arcpy.AddWarning(f"Unable to capture map view screenshot: {exc}")
        return None


def capture_interpretation_context(max_features_per_layer: int = 50) -> Dict[str, Any]:
    """Collect view, map, and layer context for interpretation."""
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    active_map = aprx.activeMap
    active_view = aprx.activeView

    if not active_map:
        return {
            "map": {"status": "No active map"},
            "view": {},
            "layers": [],
            "screenshot": None,
        }

    camera = getattr(active_view, "camera", None)
    extent = camera.getExtent() if camera and hasattr(camera, "getExtent") else None
    view_details = {
        "scale": getattr(camera, "scale", None) if camera else None,
        "spatial_reference": getattr(active_map.spatialReference, "name", "Unknown"),
    }
    if extent:
        view_details.update(
            {
                "extent": {
                    "xmin": extent.XMin,
                    "ymin": extent.YMin,
                    "xmax": extent.XMax,
                    "ymax": extent.YMax,
                }
            }
        )

    screenshot = capture_map_view_screenshot(active_view)
    visible_layers = FeatureLayerUtils.capture_visible_layer_context(
        active_map, extent, max_features_per_layer=max_features_per_layer
    )

    map_metadata = {
        "name": active_map.name,
        "spatial_reference": getattr(active_map.spatialReference, "name", "Unknown"),
        "visible_layer_count": len([lyr for lyr in active_map.listLayers() if getattr(lyr, "visible", False)]),
    }

    return {
        "map": map_metadata,
        "view": view_details,
        "layers": visible_layers,
        "screenshot": screenshot,
    }


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
