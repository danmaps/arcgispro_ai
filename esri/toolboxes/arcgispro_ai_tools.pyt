import arcpy
import json
import os
from arcgispro_ai import arcgispro_ai_utils

class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "SymphonyGIS"
        self.alias = "SymphonyGIS"

        # List of tool classes associated with this toolbox
        self.tools = [FeatureLayer,
                      Field,
                      MapInsights,
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

        # set output layer name to prompt with no spaces or special characters
        parameters[1].value = parameters[0].valueAsText.replace(" ", "_").replace("-", "_").replace(".", "_")
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        api_key = arcgispro_ai_utils.get_env_var()
        # request geojson from AI
        prompt = parameters[0].valueAsText
        geojson_data = arcgispro_ai_utils.fetch_geojson(api_key, prompt)

        # create geojson file
        output_layer = parameters[1].valueAsText
        geojson_file = f"{output_layer}.geojson"
        with open(geojson_file, "w") as f:
            json.dump(geojson_data, f, indent=4)

        # create feature layer from geojson file
        arcpy.conversion.JSONToFeatures(geojson_file, output_layer, geometry_type=arcgispro_ai_utils.infer_geometry_type(geojson_data))

        # delete geojson file
        os.remove(geojson_file)
        
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
        api_key = arcgispro_ai_utils.get_env_var(api_key_name)
        # arcpy.AddMessage(f"api_key: {api_key}")
        arcgispro_ai_utils.add_ai_response_to_feature_layer(api_key,
                                         parameters[0].valueAsText,
                                         parameters[1].valueAsText,
                                         parameters[2].valueAsText,
                                         parameters[3].valueAsText,
                                         parameters[4].valueAsText,
                                         parameters[5].valueAsText)
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return
    
class MapInsights(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Map Insights"
        self.description = "Map Insights"

    def getParameterInfo(self):
        """Define the tool parameters."""
        params = None
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
        """The source code of the tool."""
        
        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
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
            context_json = {"map": arcgispro_ai_utils.map_to_json(), "layers": arcgispro_ai_utils.get_layer_info(layers)}
            parameters[3].value = json.dumps(context_json, indent=2)
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):

        # Get the API key from the environment variable
        api_key = arcgispro_ai_utils.get_env_var()
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

        code_snippet = arcgispro_ai_utils.generate_python(
            api_key,
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
        api_key = arcgispro_ai_utils.get_env_var() # default is openai api key

        in_layer = parameters[0].values # feature layer
        field = parameters[1].value  # field

        # infer numeric type
            # shortint — Short integers (16-bit)
            # longint — Long integers (32-bit)
            # bigint — Big integers (64-bit)
            # float — Single-precision (32-bit) floating point numbers
            # double — Double-precision (64-bit) floating point numbers
        
        # store normalized values in new field like f"{field_name}_{type}"

        '''
        Population data (2020)
        Wolfram             GPT-4             GPT-4_longint
        7.178 million       7,151,502         7,151,502
        39.5 million        39.24 million     39,240,000
        5.784 million       5,758,736         5,758,736
        1.848 million       1,839,106         1,839,106
        1.961 million       1,961,504         1,961,504
        3.114 million       3,138,259         3,138,259
        2.118 million       2,117,522         2,117,522
        4.242 million       4.2 million       4,200,000
        29.22 million       29,145,505        29,145,505
        3.282 million       3,205,958         3,205,958
        '''

        # Get the API key from the environment variable
        api_key = arcgispro_ai_utils.get_env_var()  # default is OpenAI API key

        in_layer = parameters[0].valueAsText  # feature layer
        field = parameters[1].valueAsText  # field

        # Get the field values
        field_values = []
        with arcpy.da.SearchCursor(in_layer, [field]) as cursor:
            for row in cursor:
                field_values.append(row[0])

        # Convert the entire series using OpenAI API
        converted_values = self.convert_series_to_numeric(field_values, api_key)

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