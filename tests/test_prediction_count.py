
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
import io


from app import app, verify_user
from db import get_db
from models import DetectionObjects, PredictionSession

def mock_verify_user(credentials=None, db=None):
    # Ignore credentials and just return a fixed username
    return "mockuser"

#  Patch model.predict to return a mocked result
class MockModel:
    def predict(self, source, save=False, save_txt=False):
        class Result:
            def __init__(self):
                self.path = source if isinstance(source, str) else "mock_path.jpg"
                self.names = {0: "label"}
                self.boxes = type("boxes", (), {"cls": [0], "conf": [0.9], "xyxy": [[10, 10, 50, 50]]})()
        return [Result()]

class TestProcessingCount(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[verify_user] = mock_verify_user
        app.model = MockModel()
        self.client = TestClient(app)

        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

        # manually get DB session from generator
        db_gen = get_db()   # get generator object
        db = next(db_gen)   # get Session instance from generator

        # do cleanup
        db.query(PredictionSession).delete()
        db.query(DetectionObjects).delete()
        db.commit()

        # close the generator to clean up resources
        try:
            next(db_gen)
        except StopIteration:
            pass

    def test_prediction_count_empty(self):
        response = self.client.get("/prediction/count")
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("count"), 0)
        
    @patch("repository.save_prediction_session")
    @patch("repository.query_prediction_count", return_value=1)
    def test_prediction_count_after_prediction(self, mock_query_count, mock_insert):
        # First, simulate a prediction
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        mock_insert.assert_called_once()

        # Then, get the count
        response2 = self.client.get("/prediction/count")
        self.assertEqual(response2.status_code, 200)
        data = response2.json()
        self.assertEqual(data.get("count"), 1)
        mock_query_count.assert_called_with(unittest.mock.ANY, "mockuser")
    

        