import arcpy
import os
import json
import importlib.util
import re
from typing import Dict, Any, List, Optional

# Import utility functions
from arcgispro_ai.arcgispro_ai_utils import (
    generate_python,
    map_to_json
)
from arcgispro_ai.core.api_clients import (
    get_client,
    get_env_var,
    OpenAIClient,
    OpenRouterClient
)

DEFAULT_OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/o3-mini",
    "google/gemini-2.0-flash-exp:free",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat"
]

def update_model_parameters(source: str, parameters: list, current_model: str = None) -> None:
    """Update model parameters based on selected source."""
    model_configs = {
        "Azure OpenAI": {
            "models": ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
            "default": "gpt-4o-mini",
            "endpoint": True,
            "deployment": True
        },
        "OpenAI": {
            "models": [],
            "default": "gpt-4o-mini",
            "endpoint": False,
            "deployment": False
        },
        "OpenRouter": {
            "models": [],
            "default": "openai/gpt-4o-mini",
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

    # Fetch dynamic model lists for providers that support it
    if source == "OpenAI":
        try:
            api_key = get_env_var("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY")
            client = OpenAIClient(api_key)
            model_configs["OpenAI"]["models"] = client.get_available_models()
        except Exception:
            model_configs["OpenAI"]["models"] = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    elif source == "OpenRouter":
        try:
            api_key = get_env_var("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENROUTER_API_KEY")
            client = OpenRouterClient(api_key)
            models = client.get_available_models()
            model_configs["OpenRouter"]["models"] = models or DEFAULT_OPENROUTER_MODELS
        except Exception:
            model_configs["OpenRouter"]["models"] = DEFAULT_OPENROUTER_MODELS

    # Model parameter
    parameters[1].enabled = bool(config["models"])
    if config["models"]:
        parameters[1].filter.list = config["models"]
        if current_model not in config["models"]:
            parameters[1].value = config["models"][0]

    # Endpoint parameter
    parameters[2].enabled = config["endpoint"]
    if config.get("endpoint_value"):
        parameters[2].value = config["endpoint_value"]

    # Deployment parameter
    parameters[3].enabled = config["deployment"]

class FixedGenerateTool(object):
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
        source.filter.list = ["OpenRouter", "OpenAI", "Azure OpenAI", "Claude", "DeepSeek", "Local LLM"]
        source.value = "OpenRouter"

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
            "OpenRouter": "OPENROUTER_API_KEY",
            "OpenAI": "OPENAI_API_KEY",
            "Azure OpenAI": "AZURE_OPENAI_API_KEY",
            "Claude": "ANTHROPIC_API_KEY",
            "DeepSeek": "DEEPSEEK_API_KEY",
            "Local LLM": None
        }
        api_key = get_env_var(api_key_map.get(source, "OPENROUTER_API_KEY"))
        
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

# Replace the existing GenerateTool class
GenerateTool = FixedGenerateTool
