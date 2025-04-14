import sys
import os
import unittest
from unittest.mock import Mock

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Mock arcpy before any other imports
mock_arcpy = Mock()
mock_arcpy.AddMessage = Mock()
mock_arcpy.AddWarning = Mock()
mock_arcpy.AddError = Mock()
mock_arcpy.Extent = Mock()
sys.modules['arcpy'] = mock_arcpy

# Now we can import and run the tests
from test_arcgispro_ai_utils import *

if __name__ == '__main__':
    unittest.main() 