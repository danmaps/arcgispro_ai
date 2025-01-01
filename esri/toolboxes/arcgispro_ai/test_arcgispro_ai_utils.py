import unittest
import json
import os
from unittest.mock import Mock, patch
from arcgispro_ai_utils import (
    APIClient,
    OpenAIClient,
    WolframAlphaClient,
    MapUtils,
    GeoJSONUtils,
    parse_numeric_value
)

class TestAPIClient(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-api-key"
        self.base_url = "https://api.test.com"
        self.client = APIClient(self.api_key, self.base_url)

    @patch('requests.post')
    def test_make_request(self, mock_post):
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        # Test successful request
        result = self.client.make_request("test", {"data": "test"})
        self.assertEqual(result, {"result": "success"})

        # Verify the request was made correctly
        mock_post.assert_called_with(
            f"{self.base_url}/test",
            headers=self.client.headers,
            json={"data": "test"},
            verify=False
        )

class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv('OPENAI_API_KEY', 'test-api-key')
        self.client = OpenAIClient(self.api_key)

    @patch.object(APIClient, 'make_request')
    def test_get_completion(self, mock_make_request):
        # Setup mock response
        mock_make_request.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"}
        ]

        # Test without response format
        response = self.client.get_completion(messages)
        self.assertEqual(response, "Hello!")

        # Test with json_object response format
        response = self.client.get_completion(messages, response_format="json_object")
        self.assertEqual(response, "Hello!")

class TestWolframAlphaClient(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv('WOLFRAMALPHA_API_KEY', 'test-api-key')
        self.client = WolframAlphaClient(self.api_key)

    @patch.object(APIClient, 'make_request')
    def test_get_result(self, mock_make_request):
        # Setup mock response with XML content - ensure no leading whitespace
        mock_response = Mock()
        mock_response.content = '<?xml version="1.0" encoding="UTF-8"?><queryresult success="true"><pod title="Result"><subpod><plaintext>4</plaintext></subpod></pod></queryresult>'.encode()
        mock_make_request.return_value = mock_response

        result = self.client.get_result("2+2")
        self.assertEqual(result, "4")

        # Test unsuccessful query
        # mock_response.content = '<?xml version="1.0" encoding="UTF-8"?><queryresult success="false"></queryresult>'.encode()
        # with self.assertRaises(Exception):
        #     self.client.get_result("invalid query")

        # Test missing result pod
        # mock_response.content = '<?xml version="1.0" encoding="UTF-8"?><queryresult success="true"><pod title="Other"><subpod><plaintext>other data</plaintext></subpod></pod></queryresult>'.encode()
        # with self.assertRaises(Exception):
        #     self.client.get_result("query without result")

class TestMapUtils(unittest.TestCase):
    def test_metadata_to_dict(self):
        # Create a mock metadata object
        class MockMetadata:
            def __init__(self):
                self.title = "Test Title"
                self.tags = ["tag1", "tag2"]
                self.summary = "Test Summary"
                self.description = "Test Description"
                self.credits = "Test Credits"
                self.accessConstraints = "Test Constraints"
                self.XMax = 10
                self.XMin = 0
                self.YMax = 10
                self.YMin = 0

        metadata = MockMetadata()
        result = MapUtils.metadata_to_dict(metadata)

        self.assertEqual(result["title"], "Test Title")
        self.assertEqual(result["tags"], ["tag1", "tag2"])
        self.assertEqual(result["extent"]["xmax"], 10)
        self.assertEqual(result["extent"]["xmin"], 0)

    def test_expand_extent(self):
        class MockExtent:
            def __init__(self, xmin, ymin, xmax, ymax):
                self.XMin = xmin
                self.YMin = ymin
                self.XMax = xmax
                self.YMax = ymax

        extent = MockExtent(0, 0, 10, 10)
        expanded = MapUtils.expand_extent(extent, 1.1)
        
        self.assertAlmostEqual(expanded.XMin, -0.5)
        self.assertAlmostEqual(expanded.YMin, -0.5)
        self.assertAlmostEqual(expanded.XMax, 10.5)
        self.assertAlmostEqual(expanded.YMax, 10.5)

class TestGeoJSONUtils(unittest.TestCase):
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
        self.assertEqual(GeoJSONUtils.infer_geometry_type(point_geojson), "Point")

        # Test MultiPoint geometry
        multipoint_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPoint",
                "coordinates": [[0, 0], [1, 1]]
            },
            "properties": {}
        }
        self.assertEqual(GeoJSONUtils.infer_geometry_type(multipoint_geojson), "Multipoint")

        # Test LineString geometry
        linestring_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[0, 0], [1, 1]]
            },
            "properties": {}
        }
        self.assertEqual(GeoJSONUtils.infer_geometry_type(linestring_geojson), "Polyline")

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
            GeoJSONUtils.infer_geometry_type(mixed_geojson)

class TestUtilityFunctions(unittest.TestCase):
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

        # Test invalid input
        with self.assertRaises(ValueError):
            parse_numeric_value("invalid")

if __name__ == '__main__':
    unittest.main() 