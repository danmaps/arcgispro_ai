import arcpy
import json
import os
from arcgispro_ai.arcgispro_ai_utils import (
    OpenAIClient,
    WolframAlphaClient,
    MapUtils,
    GeoJSONUtils,
    FeatureLayerUtils,
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

        params = [prompt, output_layer]
        return params

    def isLicensed(self):   
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        
        import re  # Import the regular expression module

        # Use regex to replace unwanted characters with underscores
        parameters[1].value = re.sub(r'[^\w]', '_', parameters[0].valueAsText)
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        api_key = get_env_var()
        prompt = parameters[0].valueAsText
        output_layer_name = parameters[1].valueAsText

        # Initialize OpenAI client
        openai_client = OpenAIClient(api_key)

        # Fetch GeoJSON and create feature layer using the utility function
        try:
            geojson_data = openai_client.get_geojson(prompt, output_layer_name)
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
        self.description = "Adds a new attribute field to feature layers with AI-generated text. It uses the OpenAI API to create responses based on user-defined prompts that can reference existing attributes. Users provide the input layer, output layer, field name, prompt template, and an optional SQL query. The tool enriches datasets but may produce inconsistent or unexpected AI responses, reflecting the nature of AI text generation."
        self.getParameterInfo()

    def getParameterInfo(self):
        """Define the tool parameters."""
        source = arcpy.Parameter(
            displayName="Source",
            name="source",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            # multiValue=True
        )
        # source.controlCLSID = '{172840BF-D385-4F83-80E8-2AC3B79EB0E0}'
        source.filter.type = "ValueList"
        source.filter.list = ["OpenAI", "Wolfram Alpha"]
        source.value = "OpenAI"

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

        params = [source, in_layer, out_layer, field_name, prompt, sql]
        # params = None
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        source = parameters[0].value

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # associate the source with the api key variable name
        api_key_name = {"OpenAI": "OPENAI_API_KEY", "Wolfram Alpha": "WOLFRAM_ALPHA_API_KEY"}[parameters[0].valueAsText]
        # Get the API key from the environment variable
        api_key = get_env_var(api_key_name)

        # Initialize appropriate client based on source
        if parameters[0].valueAsText == "OpenAI":
            client = OpenAIClient(api_key)
        else:
            client = WolframAlphaClient(api_key)

        # Add AI response to feature layer
        client.add_ai_response_to_feature_layer(
            parameters[1].valueAsText,  # input layer
            parameters[2].valueAsText,  # output layer
            parameters[3].valueAsText,  # field name
            parameters[4].valueAsText,  # prompt
            parameters[5].valueAsText   # sql query
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
            displayName="Run the code (not a good idea!)",
            name="eval",
            datatype="Boolean",
            parameterType="Required",
            direction="Input",
        )
        
        eval.value = False # default value False

        context = arcpy.Parameter(
            displayName="Context (this will be passed to the AI)",
            name="context",
            datatype="GPstring",
            parameterType="Optional",
            direction="Input",
            category="Context",
        )
        context.controlCLSID = '{E5456E51-0C41-4797-9EE4-5269820C6F0E}'

        params = [layers,prompt,eval,context]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        layers = parameters[0].values
        # combine map and layer data into one JSON
        # only do this if context_json is empty
        if parameters[3].valueAsText == "":
            context_json = {
                "map": MapUtils.map_to_json(), 
                "layers": FeatureLayerUtils.get_layer_info(layers)
            }
            parameters[3].value = json.dumps(context_json, indent=2)
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):

        # Get the API key from the environment variable
        api_key = get_env_var()
        # api_key = arcgispro_ai_utils.get_SymphonyGIS_api_key()
        layers = parameters[0].values # feature layer (multiple)
        prompt = parameters[1].value  # string
        eval = parameters[2].value  # boolean
        derived_context = parameters[3].value  # string with multiple lines
        
        #debug
        # arcpy.AddMessage("api_key: {}".format(api_key))
        # arcpy.AddMessage("feature_layer: {}".format(layers))
        # arcpy.AddMessage("prompt: {}".format(prompt))
        # arcpy.AddMessage("eval: {}".format(eval))

        code_snippet = openai_client.generate_python(
            derived_context,
            prompt.strip(),
        )

        if eval == True:
            try:
                if code_snippet:
                    # execute the code
                    arcpy.AddMessage("Executing code... fingers crossed!")
                    exec(code_snippet)
                else:
                    raise Exception("No code generated. Please try again.")

            # catch AttributeError: 'NoneType' object has no attribute 'camera'
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