# Import non-arcpy components by default
from .core import *

# Import arcpy-dependent components only when explicitly requested
import arcpy
import os
import sys

# Get the path to the toolbox using a relative path from this file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up one level to the toolboxes directory
arcgispro_ai_tools_path = os.path.join(parent_dir, "arcgispro_ai_tools.pyt")

# Check if the toolbox file exists
if not os.path.exists(arcgispro_ai_tools_path):
    raise FileNotFoundError(f"Toolbox file not found at: {arcgispro_ai_tools_path}")

# Import the toolbox
arcpy.ImportToolbox(arcgispro_ai_tools_path, "arcgispro_ai_tools")
