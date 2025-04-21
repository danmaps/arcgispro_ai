# Import non-arcpy components by default
from .core import *

# Import arcpy-dependent components only when explicitly requested
import arcpy
# import the pyt from: %localappdata%\ESRI\conda\envs\<your_env>\Lib\site-packages\arcgispro_ai\toolboxes
# this should resolve to something like "C:\Users\danny\AppData\Local\ESRI\conda\envs\arcgispro-py3-clone\Lib\site-packages\arcgispro_ai\toolboxes\arcgispro_ai_tools.pyt"

env_path = arcpy.env.activeEnvironment.path

# get the path to the arcgispro_ai_tools.pyt file
arcgispro_ai_tools_path = os.path.join(env_path, "Lib", "site-packages", "arcgispro_ai", "toolboxes", "arcgispro_ai_tools.pyt")

# import the toolbox
arcpy.importToolbox(arcgispro_ai_tools_path, "arcgispro_ai_tools.pyt")