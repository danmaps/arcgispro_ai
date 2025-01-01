import unittest
import json
import os
from arcgispro_ai_utils import (
    get_openai_response,
    get_wolframalpha_response,
    parse_numeric_value,
    trim_code_block,
    expand_extent,
    infer_geometry_type
)

class TestArcGISProAIUtils(unittest.TestCase):
    def setUp(self):
        # You can set your API keys here for testing
        self.openai_api_key = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
        self.wolframalpha_api_key = os.getenv('WOLFRAMALPHA_API_KEY', 'your-wolframalpha-api-key-here')

    def test_parse_numeric_value(self):
        test_cases = [
            ("1,000", 1000),
            ("1,000.5", 1000.5),
            ("1.5", 1.5),
            ("1000", 1000),
        ]
        for input_value, expected_output in test_cases:
            with self.subTest(input_value=input_value):
                result = parse_numeric_value(input_value)
                self.assertEqual(result, expected_output)

    def test_infer_geometry_type(self):
        # Test Point geometry
        point_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [0, 0]
            },
            "properties": {}
        }
        self.assertEqual(infer_geometry_type(point_geojson), "Point")

        # Test MultiPoint geometry
        multipoint_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPoint",
                "coordinates": [[0, 0], [1, 1]]
            },
            "properties": {}
        }
        self.assertEqual(infer_geometry_type(multipoint_geojson), "Multipoint")

        # Test LineString geometry
        linestring_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[0, 0], [1, 1]]
            },
            "properties": {}
        }
        self.assertEqual(infer_geometry_type(linestring_geojson), "Polyline")

        # Test multiple features with same geometry type
        features_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [1, 1]
                    },
                    "properties": {}
                }
            ]
        }
        self.assertEqual(infer_geometry_type(features_geojson), "Point")

        # Test mixed geometry types should raise ValueError
        mixed_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0, 0], [1, 1]]
                    },
                    "properties": {}
                }
            ]
        }
        with self.assertRaises(ValueError):
            infer_geometry_type(mixed_geojson)

    def test_get_openai_response(self):
        if self.openai_api_key != 'your-openai-api-key-here':
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ]
            response = get_openai_response(self.openai_api_key, messages, "test")
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)

    def test_get_wolframalpha_response(self):
        if self.wolframalpha_api_key != 'your-wolframalpha-api-key-here':
            query = "What is 2+2?"
            response = get_wolframalpha_response(self.wolframalpha_api_key, query)
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)

    def test_expand_extent(self):
        class MockExtent:
            def __init__(self, xmin, ymin, xmax, ymax):
                self.XMin = xmin
                self.YMin = ymin
                self.XMax = xmax
                self.YMax = ymax

        # Test with factor 1.1 (10% expansion)
        extent = MockExtent(0, 0, 10, 10)
        expanded = expand_extent(extent, 1.1)
        
        self.assertAlmostEqual(expanded.XMin, -0.5)
        self.assertAlmostEqual(expanded.YMin, -0.5)
        self.assertAlmostEqual(expanded.XMax, 10.5)
        self.assertAlmostEqual(expanded.YMax, 10.5)

if __name__ == '__main__':
    unittest.main() 