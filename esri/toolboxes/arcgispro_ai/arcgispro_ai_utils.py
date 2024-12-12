import requests
import time
import arcpy
import json
import os
import xml.etree.ElementTree as ET
import re

def map_to_json(in_map=None, output_json_path=None):
    """
    Generates a JSON object containing information about a map.

    Args:
        in_map (str, optional): The name of the map to get information from. If not provided, the active map in the current project will be used. Defaults to None.
        output_json_path (str, optional): The path to the JSON file where the map information will be saved. If not provided, the map information will not be saved. Defaults to None.

    Returns:
        dict: A dictionary containing information about the map, including the map name, title, description, spatial reference, layers, and properties.

    Raises:
        ValueError: If no active map is found in the current project.

    Notes:
        - The function uses the `arcpy` module to interact with the ArcGIS Pro application.
        - The function collects information about the active map, including the map name, title, description, spatial reference, and layers.
        - For each layer, the function collects information such as the layer name, feature layer status, raster layer status, web layer status, visibility, metadata, spatial reference, extent, fields, record count, source type, geometry type, renderer, and labeling.
        - If `output_json_path` is provided, the map information will be saved to a JSON file at the specified path.

    Example:
        >>> map_info = map_to_json()
        >>> print(map_info)
        {
            "map_name": "MyMap",
            "title": "My Map",
            "description": "This is my map.",
            "spatial_reference": "WGS84",
            "layers": [
                {
                    "name": "MyLayer",
                    "feature_layer": True,
                    "raster_layer": False,
                    "web_layer": False,
                    "visible": True,
                    "metadata": {
                        "title": "My Layer",
                        "tags": ["Tag1", "Tag2"],
                        "summary": "This is my layer.",
                        "description": "This is a description of my layer.",
                        "credits": "Credits for the layer.",
                        "access_constraints": "Access constraints for the layer.",
                        "extent": {
                            "xmin": -180,
                            "ymin": -90,
                            "xmax": 180,
                            "ymax": 90
                        }
                    },
                    "spatial_reference": "WGS84",
                    "extent": {
                        "xmin": -180,
                        "ymin": -90,
                        "xmax": 180,
                        "ymax": 90
                    },
                    "fields": [
                        {
                            "name": "ID",
                            "type": "Integer",
                            "length": 10
                        },
                        {
                            "name": "Name",
                            "type": "String",
                            "length": 50
                        }
                    ],
                    "record_count": 100,
                    "source_type": "FeatureClass",
                    "geometry_type": "Point",
                    "renderer": "SimpleRenderer",
                    "labeling": True
                }
            ],
            "properties": {
                "rotation": 0,
                "units": "DecimalDegrees",
                "time_enabled": False,
                "metadata": {
                    "title": "My Map",
                    "tags": ["Tag1", "Tag2"],
                    "summary": "This is my map.",
                    "description": "This is a description of my map.",
                    "credits": "Credits for the map.",
                    "access_constraints": "Access constraints for the map.",
                    "extent": {
                        "xmin": -180,
                        "ymin": -90,
                        "xmax": 180,
                        "ymax": 90
                    }
                }
            }
        }
    """

    # Function to convert metadata to a dictionary
    def metadata_to_dict(metadata):
        """
        Convert the given metadata object to a dictionary.

        Parameters:
            metadata (object): The metadata object to be converted.

        Returns:
            dict: The dictionary representation of the metadata object.

        Raises:
            None.

        Example:
            >>> metadata = Metadata(title="My Map", tags=["Tag1", "Tag2"], summary="This is my map.", description="This is a description of my map.", credits="Credits for the map.", accessConstraints="Access constraints for the map.", XMax=180, XMin=-180, YMax=90, YMin=-90)
            >>> metadata_to_dict(metadata)
            {'title': 'My Map', 'tags': ['Tag1', 'Tag2'], 'summary': 'This is my map.', 'description': 'This is a description of my map.', 'credits': 'Credits for the map.', 'access_constraints': 'Access constraints for the map.', 'extent': {'xmax': 180, 'xmin': -180, 'ymax': 90, 'ymin': -90}}
        """
        if metadata is None:
            return "No metadata"

        extent_dict = {}
        if hasattr(metadata, "XMax"):
            extent_dict["xmax"] = metadata.XMax
        if hasattr(metadata, "XMin"):
            extent_dict["xmin"] = metadata.XMin
        if hasattr(metadata, "YMax"):
            extent_dict["ymax"] = metadata.YMax
        if hasattr(metadata, "YMin"):
            extent_dict["ymin"] = metadata.YMin

        meta_dict = {
            "title": metadata.title if hasattr(metadata, "title") else "No title",
            "tags": metadata.tags if hasattr(metadata, "tags") else "No tags",
            "summary": metadata.summary
            if hasattr(metadata, "summary")
            else "No summary",
            "description": metadata.description
            if hasattr(metadata, "description")
            else "No description",
            "credits": metadata.credits
            if hasattr(metadata, "credits")
            else "No credits",
            "access_constraints": metadata.accessConstraints
            if hasattr(metadata, "accessConstraints")
            else "No access constraints",
            "extent": extent_dict,
        }
        return meta_dict
    
    # only perform this if we're executing in arcgis pro context

    aprx = arcpy.mp.ArcGISProject("CURRENT")
    if not in_map:
        active_map = aprx.activeMap
        if not active_map:
            raise ValueError("No active map found in the current project.")
    else:
        active_map = aprx.listMaps(in_map)[0]

    # Collect map information
    map_info = {
        "map_name": active_map.name,
        "title": active_map.title if hasattr(active_map, "title") else "No title",
        "description": active_map.description
        if hasattr(active_map, "description")
        else "No description",
        "spatial_reference": active_map.spatialReference.name,
        "layers": [],
        "properties": {
            "rotation": active_map.rotation
            if hasattr(active_map, "rotation")
            else "No rotation",
            "units": active_map.units if hasattr(active_map, "units") else "No units",
            "time_enabled": active_map.isTimeEnabled
            if hasattr(active_map, "isTimeEnabled")
            else "No time enabled",
            "metadata": metadata_to_dict(active_map.metadata)
            if hasattr(active_map, "metadata")
            else "No metadata",
        },
    }

    # Iterate through layers and collect information
    for layer in active_map.listLayers():
        layer_info = {
            "name": layer.name,
            "feature_layer": layer.isFeatureLayer,
            "raster_layer": layer.isRasterLayer,
            "web_layer": layer.isWebLayer,
            "visible": layer.visible,
            "metadata": metadata_to_dict(layer.metadata)
            if hasattr(layer, "metadata")
            else "No metadata",
        }

        if layer.isFeatureLayer:
            dataset = arcpy.Describe(layer.dataSource)
            layer_info.update(
                {
                    "spatial_reference": dataset.spatialReference.name
                    if hasattr(dataset, "spatialReference")
                    else "Unknown",
                    "extent": {
                        "xmin": dataset.extent.XMin,
                        "ymin": dataset.extent.YMin,
                        "xmax": dataset.extent.XMax,
                        "ymax": dataset.extent.YMax,
                    }
                    if hasattr(dataset, "extent")
                    else "Unknown",
                    "fields": [],
                    "record_count": 0,
                    "source_type": dataset.dataType
                    if hasattr(dataset, "dataType")
                    else "Unknown",
                    "geometry_type": dataset.shapeType
                    if hasattr(dataset, "shapeType")
                    else "Unknown",
                    "renderer": layer.symbology.renderer.type
                    if hasattr(layer, "symbology")
                    and hasattr(layer.symbology, "renderer")
                    else "Unknown",
                    "labeling": layer.showLabels
                    if hasattr(layer, "showLabels")
                    else "Unknown",
                }
            )

            # Get fields information
            if hasattr(dataset, "fields"):
                for field in dataset.fields:
                    field_info = {
                        "name": field.name,
                        "type": field.type,
                        "length": field.length,
                    }
                    layer_info["fields"].append(field_info)

            # Get record count if the layer has records
            if dataset.dataType in ["FeatureClass", "Table"]:
                layer_info["record_count"] = int(
                    arcpy.management.GetCount(layer.dataSource)[0]
                )

        map_info["layers"].append(layer_info)

    if output_json_path:
        # Write the map information to a JSON file
        with open(output_json_path, "w") as json_file:
            json.dump(map_info, json_file, indent=4)

        print(f"Map information has been written to {output_json_path}")

    return map_info


def fetch_geojson(api_key, query, output_layer_name):
    """
    Fetches GeoJSON data using an AI response and creates a feature layer in ArcGIS Pro.

    Parameters:
    api_key (str): API key for OpenAI.
    query (str): User query for the AI to generate GeoJSON data.
    output_layer_name (str): Name of the output layer to be created in ArcGIS Pro.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that always only returns valid GeoJSON in response to user queries. Don't use too many vertices. Include somewhat detailed geometry and any attributes you think might be relevant. Include factual information. If you want to communicate text to the user, you may use a message property in the attributes of geometry objects. For compatibility with ArcGIS Pro, avoid multiple geometry types in the GeoJSON output. For example, don't mix points and polygons."},
        {"role": "user", "content": query}
    ]

    try:
        geojson_data = get_openai_response(api_key, messages)

        # Add debugging to inspect the raw GeoJSON response
        arcpy.AddMessage(f"Raw GeoJSON data: {geojson_data}")

        geojson_data = json.loads(geojson_data)  # Assuming single response for simplicity
        geometry_type = infer_geometry_type(geojson_data)
    except Exception as e:
        arcpy.AddError(str(e))
        return

    geojson_file = os.path.join("geojson_output", f"{output_layer_name}.geojson")
    with open(geojson_file, 'w') as f:
        json.dump(geojson_data, f)

    arcpy.conversion.JSONToFeatures(geojson_file, output_layer_name, geometry_type=geometry_type)
    
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    if aprx.activeMap:
        active_map = aprx.activeMap
        output_layer_path = os.path.join(aprx.defaultGeodatabase, output_layer_name)
        arcpy.AddMessage(f"Adding layer from: {output_layer_path}")
        
        try:
            active_map.addDataFromPath(output_layer_path)
            layer = active_map.listLayers(output_layer_name)[0]

            # Get the data source and its extent
            desc = arcpy.Describe(layer.dataSource)
            extent = desc.extent

            if extent:
                expanded_extent = expand_extent(extent)
                active_view = aprx.activeView

                # Check if the active view is a map view
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


def expand_extent(extent, factor=1.1):
    """
    Expands the given extent by the specified factor.

    Parameters:
    extent (arcpy.Extent): The extent to be expanded.
    factor (float): The factor by which to expand the extent.

    Returns:
    arcpy.Extent: The expanded extent.
    """
    width = extent.XMax - extent.XMin
    height = extent.YMax - extent.YMin
    expanded_extent = arcpy.Extent(
        extent.XMin - width * (factor - 1) / 2,
        extent.YMin - height * (factor - 1) / 2,
        extent.XMax + width * (factor - 1) / 2,
        extent.YMax + height * (factor - 1) / 2
    )
    return expanded_extent

def infer_geometry_type(geojson_data):
    """
    Infers the geometry type from GeoJSON data.

    Parameters:
    geojson_data (dict): GeoJSON data as a dictionary.

    Returns:
    str: Geometry type compatible with ArcGIS Pro.

    Raises:
    ValueError: If multiple geometry types are found in the GeoJSON data.
    """
    geometry_type_map = {
        "Point": "Point",
        "MultiPoint": "Multipoint",
        "LineString": "Polyline",
        "MultiLineString": "Polyline",
        "Polygon": "Polygon",
        "MultiPolygon": "Polygon"
    }

    geometry_types = set()
    for feature in geojson_data["features"]:
        geometry_type = feature["geometry"]["type"]
        arcpy.AddMessage(f"found {geometry_type}")
        geometry_types.add(geometry_type_map.get(geometry_type))

    if len(geometry_types) == 1:
        return geometry_types.pop()
    else:
        raise ValueError("Multiple geometry types found in GeoJSON")


def convert_series_to_numeric(api_key, field_values):
    """Convert a series of text representations of numeric data to actual numeric values using OpenAI API."""
    # Create a prompt for the OpenAI API to clean and convert the data
    prompt = (
        "Convert the following text representations of numeric data into consistent numeric values:\n\n"
    )
    for value in field_values:
        prompt += f"{value}\n"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini", # gpt-4o-2024-08-06
        "messages": prompt,
        "temperature": 0.3,  # be more predictable, less creative
        "max_tokens": 1500,
        "n": 1,
        "stop": None,
    }
    # Retry up to 3 times if the request fails
    for _ in range(3):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                verify=False,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            arcpy.AddWarning(f"Retrying openai response generation due to: {e}")
            time.sleep(1)

    converted_values = response.choices[0].text.strip().split("\n")
    numeric_values = [parse_numeric_value(val) for val in converted_values]
    return numeric_values

def parse_numeric_value(text_value):
    """
    Parses a text value representing a numeric value and returns the corresponding numeric value.

    Args:
        text_value (str): The text value to be parsed.

    Returns:
        float or int: The parsed numeric value. If the text value contains a comma, it is replaced with an empty
        string and the resulting value is converted to a float. If the resulting value contains a dot, it is returned
        as a float. Otherwise, it is converted to an integer.
    """
    if "," in text_value:
        no_commas = float(text_value.replace(",", ""))
    if "." in no_commas:
        return no_commas
    else:
        return int(no_commas)


def get_openai_response(api_key, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.5,  # be more predictable, less creative
        "max_tokens": 500,
    }

    # Retry up to 3 times if the request fails
    for _ in range(3):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                verify=False,
            )
            response.raise_for_status()
            arcpy.AddMessage(f"Returning response from {data['model']}")
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            arcpy.AddWarning(f"Retrying openai response generation due to: {e}")
            time.sleep(1)
    raise Exception("Failed to get openai response after retries")

def get_symphony_response(endpoint, api_key, messages):
    arcpy.AddMessage(f"getting symphony respose with {api_key}")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.5,  # be more predictable, less creative
        "max_tokens": 500,
    }

    # Retry up to 3 times if the request fails
    for _ in range(3):
        try:
            response = requests.post(
                f"https://symphony.com/api/{endpoint}",
                headers=headers,
                json=data,
                verify=False,
            )
            response.raise_for_status()
            arcpy.AddMessage(f"Returning response from {data['model']}")
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            arcpy.AddWarning(f"Retrying symphony response generation due to: {e}")
            time.sleep(1)
    raise Exception("Failed to get symphony response after retries")


def get_wolframalpha_response(api_key, query):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"appid": api_key, "input": query}
    # data = {"appid": "AX338J-7Y83GVP84G", "input": "what is Arizona's state bird?"}

    # Retry up to 3 times if the request fails
    for _ in range(3):
        try:
            response = requests.post(
                "https://api.wolframalpha.com/v2/query",
                headers=headers,
                data=data,
                verify=False,
            )
            response.raise_for_status()

            # arcpy.AddMessage("Response: " + response.text) # debug
            # Parsing the XML response
            root = ET.fromstring(response.content)
            if root.attrib.get('success') == 'true':
                for pod in root.findall(".//pod[@title='Result']"):
                    for subpod in pod.findall('subpod'):
                        plaintext = subpod.find('plaintext')
                        if plaintext is not None and plaintext.text:
                            return plaintext.text.strip()
                arcpy.AddWarning("Result pod not found in the response")
            else:
                arcpy.AddWarning("Query was not successful")
            
        except requests.exceptions.RequestException as e:
            arcpy.AddWarning(f"Retrying wolframalpha response generation due to: {e}")
            time.sleep(1)
    raise Exception("Failed to get wolframalpha response after retries")


def generate_python(api_key, map_info, prompt, explain=False):
    """
    Generate python using AI response for a given API key, JSON data, and a question.

    Parameters:
    api_key (str): API key for OpenAI.
    json_data (dict): JSON data containing map information.
    prompt (str): A prompt for the AI referring to a certain selection in the map.

    Returns:
    str: AI-generated code.
    """
    if prompt:
        messages = [
            {"role": "system", "content": json.dumps(map_info, indent=4)},
            {"role": "user", "content": f"{prompt}"},
        ]

    # prepend messages with python prompts from prompts.json
    # C:\Users\mcveydb\dev\geoprocess-ai-node-proxy\config
    # C:\Users\mcveydb\dev\geoprocess-ai-node-proxy\client\python_tool_openai\symphony_tools.py
    # arcpy.AddMessage(f"Current working directory: {os.getcwd()}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'config', 'prompts.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    # prompts = json.loads(json_string)
    messages = prompts["python"] + messages
    # arcpy.AddMessage(f"messages: {messages}")


    try:
        code_snippet = get_openai_response(api_key, messages)
        # code_snippet = get_symphony_response("python",api_key,messages)

        def trim_code_block(code_block):
            """
            Remove the language identifier from the beginning and the triple backticks from the end of a code block.
            
            Args:
                code_block (str): The code block string to be trimmed.
            
            Returns:
                str: The trimmed code block.
            """
            # Use regular expressions to remove the language identifier and triple backticks
            code_block = re.sub(r'^```[a-zA-Z]*\n', '', code_block)  # Remove ```language\n at the start
            code_block = re.sub(r'\n```$', '', code_block)  # Remove \n``` at the end
            return code_block.strip()

        # trim response and show to user through message
        code_snippet = trim_code_block(code_snippet)
        line = f"<html><hr></html>"
        # arcpy.AddMessage("AI generated code:")
        arcpy.AddMessage(line)
        arcpy.AddMessage(code_snippet)
        arcpy.AddMessage(line)

    except Exception as e:
        arcpy.AddError(str(e))
        return

    return code_snippet


def get_top_n_records(feature_class, fields, n):
    """
    Retrieves the top 5 records from a feature class.
    
    Parameters:
    feature_class (str): Path to the feature class.
    fields (list): List of fields to retrieve.

    Returns:
    list: A list of dictionaries containing the top 5 records.
    """
    records = []

    try:
        with arcpy.da.SearchCursor(feature_class, fields) as cursor:
            for i, row in enumerate(cursor):
                if i >= n:
                    break
                record = {field: value for field, value in zip(fields, row)}
                records.append(record)
                
    except Exception as e:
        arcpy.AddError(f"Error retrieving records: {e}")
    
    return records

def get_layer_info(input_layers):
    """
    Gathers layer information, including data records, for the selected layer.
    This is for context for the AI response. Keeping this separate from the map
    information might make users feel more comfortable sharing data with the AI.
    They can easily control, and see/edit, what data is passed to the AI.

    Returns:
    dict: JSON object representing the layer.
    """
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    active_map = aprx.activeMap
    layers_info = {}
        
    if input_layers:
        for l in input_layers:
            layer = active_map.listLayers(l)[0]
            if layer.isFeatureLayer:
                layers_info[layer.name] = {"name": layer.name, "path": layer.dataSource}
                dataset = arcpy.Describe(layer.dataSource)
                layers_info[layer.name]["data"] = get_top_n_records(layer, [f.name for f in dataset.fields], 5)
    
    return layers_info


def get_env_var(var_name="OPENAI_API_KEY"):
    # arcpy.AddMessage(f"Fetching API key from {var_name} environment variable.")
    return os.environ.get(var_name, "")

config_path = os.path.join(os.getenv("APPDATA"), "SymphonyGIS", "config.json")


def get_SymphonyGIS_api_key():
    if not os.path.exists(config_path):
        raise Exception(f"API key not found in {config_path}. Run setup.py to create a new config file with your SymphonyGIS API key.")
    else:
        with open(config_path) as config_file:
            api_key = json.load(config_file).get("api_key")
    return api_key


def add_ai_response_to_feature_layer(api_key, source, in_layer, out_layer, field_name, prompt_template, sql_query=None):
    """
    Enriches an existing feature layer by adding a new field with AI-generated responses.

    Parameters:
    api_key (str): API key for OpenAI.
    in_layer (str): Path to the input feature layer.
    out_layer (str): Path to the output feature layer. If None, in_layer will be updated.
    field_name (str): Name of the field to add the AI responses.
    prompt_template (str): Template for the prompt to be used by AI.
    sql_query (str, optional): Optional SQL query to filter the features.
    """
    if out_layer:
        arcpy.CopyFeatures_management(in_layer, out_layer)
        layer_to_use = out_layer
    else:
        layer_to_use = in_layer

    # debug
    # arcpy.AddMessage(layer_to_use)

    # Add new field for AI responses
    if field_name not in [f.name for f in arcpy.ListFields(layer_to_use)]:
        arcpy.management.AddField(layer_to_use, field_name, "TEXT")
    else:
        arcpy.management.AddField(layer_to_use, field_name + "_AI", "TEXT")
        field_name += "_AI"

    def generate_ai_responses_for_feature_class(source, feature_class, field_name, prompt_template):
        """
        Generates AI responses for each feature in the feature class and updates the new field with these responses.

        Parameters:
        source (str): Source to use for the intelligence.
        feature_class (str): Path to the feature class.
        field_name (str): Name of the field to add the AI responses.
        prompt_template (str): Template for the prompt to be used by AI.
        """
        # Get the OID field name
        desc = arcpy.Describe(feature_class)
        oid_field_name = desc.OIDFieldName

        # Define the fields to be included in the cursor
        fields = [field.name for field in arcpy.ListFields(feature_class)]
        
        # Ensure the new field exists
        if field_name not in fields:
            arcpy.AddField_management(feature_class, field_name, "TEXT")
        
        fields.append(field_name)  # Add the new field to the fields list

        # Store prompts and their corresponding OIDs in a dictionary
        prompts_dict = {}

        # Use a SearchCursor to iterate over the rows in the feature class and generate prompts
        with arcpy.da.SearchCursor(feature_class, fields[:-1], sql_query) as cursor:  # Exclude the new field
            for row in cursor:
                row_dict = {field: value for field, value in zip(fields[:-1], row)}
                formatted_prompt = prompt_template.format(**row_dict)
                oid = row_dict[oid_field_name]
                prompts_dict[oid] = formatted_prompt

        # Debug: Check a sample record from prompts_dict
        if prompts_dict:
            # Get the first key-value pair from the dictionary
            sample_oid, sample_prompt = next(iter(prompts_dict.items()))
            arcpy.AddMessage(f"{oid_field_name} {sample_oid}: {sample_prompt}")
        else:
            arcpy.AddMessage("prompts_dict is empty.")

        # Get AI responses for each prompt
        if source == "OpenAI":
            role = "Respond without any other information, not even a complete sentence. No need for any other decoration or verbage."
            responses_dict = {
                # oid: get_openai_response(api_key, [{"role": "system", "content": role}, {"role": "user", "content": prompt}])
                oid: get_openai_response(api_key, [{"role": "system", "content": role}, {"role": "user", "content": prompt}])
                for oid, prompt in prompts_dict.items()
            }
        elif source == "Wolfram Alpha":
            responses_dict = {
                oid: get_wolframalpha_response(api_key, prompt)
                for oid, prompt in prompts_dict.items()
            }
            

        # Use an UpdateCursor to write the AI responses back to the feature class
        with arcpy.da.UpdateCursor(feature_class, [oid_field_name, field_name]) as cursor:
            for row in cursor:
                oid = row[0]
                if oid in responses_dict:
                    row[1] = responses_dict[oid]
                    cursor.updateRow(row)
    
    generate_ai_responses_for_feature_class(source, layer_to_use, field_name, prompt_template)


