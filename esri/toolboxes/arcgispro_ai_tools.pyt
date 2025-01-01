import arcpy
import json
import os
from arcgispro_ai.arcgispro_ai_utils import (
    get_client,
    fetch_geojson,
    generate_python,
    add_ai_response_to_feature_layer,
    get_env_var
)

class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "SymphonyGIS"
        self.alias = "SymphonyGIS"

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
        if source == "Azure OpenAI":
            parameters[1].enabled = True
            parameters[2].enabled = True
            parameters[3].enabled = True
            parameters[1].value = "gpt-4"
        elif source == "OpenAI":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "gpt-4"
        elif source == "Claude":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "claude-3-opus-20240229"
        elif source == "DeepSeek":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "deepseek-chat"
        elif source == "Local LLM":
            parameters[1].enabled = False
            parameters[2].enabled = True
            parameters[3].enabled = False
            parameters[2].value = "http://localhost:8000"

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
        if source == "Azure OpenAI":
            parameters[1].enabled = True
            parameters[2].enabled = True
            parameters[3].enabled = True
            parameters[1].value = "gpt-4"
        elif source == "OpenAI":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "gpt-4"
        elif source == "Claude":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "claude-3-opus-20240229"
        elif source == "DeepSeek":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "deepseek-chat"
        elif source == "Local LLM":
            parameters[1].enabled = False
            parameters[2].enabled = True
            parameters[3].enabled = False
            parameters[2].value = "http://localhost:8000"
        elif source == "Wolfram Alpha":
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
        map_info = MapUtils.map_to_json(in_map)
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

        eval = arcpy.Parameter(
            displayName="Execute Generated Code",
            name="eval",
            datatype="Boolean",
            parameterType="Required",
            direction="Input",
        )
        eval.value = False

        context = arcpy.Parameter(
            displayName="Context (this will be passed to the AI)",
            name="context",
            datatype="GPstring",
            parameterType="Optional",
            direction="Input",
            category="Context",
        )
        context.controlCLSID = '{E5456E51-0C41-4797-9EE4-5269820C6F0E}'

        params = [source, model, endpoint, deployment, layers, prompt, eval, context]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        source = parameters[0].value
        if source == "Azure OpenAI":
            parameters[1].enabled = True
            parameters[2].enabled = True
            parameters[3].enabled = True
            parameters[1].value = "gpt-4"
        elif source == "OpenAI":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "gpt-4"
        elif source == "Claude":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "claude-3-opus-20240229"
        elif source == "DeepSeek":
            parameters[1].enabled = True
            parameters[2].enabled = False
            parameters[3].enabled = False
            parameters[1].value = "deepseek-chat"
        elif source == "Local LLM":
            parameters[1].enabled = False
            parameters[2].enabled = True
            parameters[3].enabled = False
            parameters[2].value = "http://localhost:8000"

        layers = parameters[4].values
        # combine map and layer data into one JSON
        # only do this if context is empty
        if parameters[7].valueAsText == "":
            context_json = {
                "map": MapUtils.map_to_json(), 
                "layers": FeatureLayerUtils.get_layer_info(layers)
            }
            parameters[7].value = json.dumps(context_json, indent=2)
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
        eval = parameters[6].value
        derived_context = parameters[7].value

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

        code_snippet = generate_python(
            api_key,
            json.loads(derived_context),
            prompt.strip(),
            source,
            **kwargs
        )

        if eval == True:
            try:
                if code_snippet:
                    arcpy.AddMessage("Executing code... fingers crossed!")
                    exec(code_snippet)
                else:
                    raise Exception("No code generated. Please try again.")
            except AttributeError as e:
                arcpy.AddError(f"{e}\n\nMake sure a map view is active.")
            except Exception as e:
                arcpy.AddError(
                    f"{e}\n\nThe code may be invalid. Please check the code and try again."
                )

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

        # field type must be text
        # field must be a field in the input layer

        params = [in_layer,field]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        # Get the API key from the environment variable
        api_key = get_env_var()  # default is OpenAI API key
        openai_client = OpenAIClient(api_key)

        in_layer = parameters[0].valueAsText  # feature layer
        field = parameters[1].valueAsText  # field

        # Get the field values
        field_values = []
        with arcpy.da.SearchCursor(in_layer, [field]) as cursor:
            for row in cursor:
                field_values.append(row[0])

        # Convert the entire series using OpenAI API
        converted_values = openai_client.convert_series_to_numeric(field_values)

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