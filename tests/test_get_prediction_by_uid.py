
import unittest

from fastapi.params import Depends
from fastapi.testclient import TestClient
from PIL import Image
import io

from pytest import Session
from app import app,DB_PATH
import base64
from unittest.mock import patch, Mock

from db import get_db
import repository
def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


class FakeSession:
    def __init__(self, uid,timestamp,original_image,predicted_image,detection_objects):
        self.uid = uid
        self.timestamp = timestamp
        self.original_image = original_image
        self.predicted_image = predicted_image
        self.detection_objects = detection_objects


class TestGetPredictionByUid(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        db: Session = Depends(get_db)

        
        # Register test user
        self.username = "tameer"
        self.password = "1234"
        auth_header = get_basic_auth_header(self.username, self.password)
        self.client.post("/register", headers=auth_header)

        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_uid_not_found(self):
        headers = get_basic_auth_header(self.username, self.password)
        response2 = self.client.get("/prediction/-1",headers=headers)
        self.assertEqual(response2.status_code, 404)
        self.assertEqual(response2.json()["detail"], "Prediction not found")
    

    @patch("repository.query_get_prediction_by_uid")
    def test_get_uid(self, mock_query):
        # First call returns single PredictionSession mock
        prediction_mock = Mock(
            uid="123",
            timestamp="2023-01-01T12:00:00",
            original_image="input.png",
            predicted_image="output.png"
        )
        # Second call returns list of DetectionObjects mocks (empty here)
        detection_objects_mock = []

        # Side effect list for sequential return values:
        mock_query.side_effect = [prediction_mock, detection_objects_mock]

        headers = get_basic_auth_header(self.username, self.password)
        response = self.client.get("/prediction/123", headers=headers)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], "123")
        self.assertEqual(data["detection_objects"], [])
        self.assertEqual(mock_query.call_count, 2)