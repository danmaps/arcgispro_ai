# Import non-arcpy components by default
from .core import *

# Import arcpy-dependent components only when explicitly requested
def import_arcpy_utils():
    from . import arcgispro_ai_utils
    return arcgispro_ai_utils