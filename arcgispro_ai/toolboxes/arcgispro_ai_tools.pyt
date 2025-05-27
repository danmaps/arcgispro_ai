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

        converted_values = get_client(api_key, source, **kwargs).convert_series_to_numeric(field_values)

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
        self.label = "Generate Reusable Tool"
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
        input_type.filter.list = ["Python Code", "Natural Language Prompt"]
        input_type.value = "Python Code"

        input_code = arcpy.Parameter(
            displayName="Python Code",
            name="input_code",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        input_code.enabled = True
        input_code.controlCLSID = '{E5456E51-0C41-4797-9EE4-5269820C6F0E}'

        input_prompt = arcpy.Parameter(
            displayName="Natural Language Prompt",
            name="input_prompt",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        input_prompt.enabled = False

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
            # Create a default context for code generation
            context_json = {"map": map_to_json(), "layers": []}
            
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
            client = get_client(api_key, source, **kwargs)
            response = client.generate_text(prompt_text)
            
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