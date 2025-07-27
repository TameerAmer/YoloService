import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app import app, verify_user

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

class TestGetImage(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides[verify_user] = mock_verify_user
        app.model = MockModel()
        self.client = TestClient(app)

        # Create and upload an image
        image = Image.new("RGB", (100, 100), "blue")
        self.image_bytes = io.BytesIO()
        image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        self.uid = response.json()["prediction_uid"]
        self.filename = f"{self.uid}.jpg"
    
    @classmethod
    def tearDownClass(cls):
        # Clear overrides after all tests
        app.dependency_overrides = {}

    def test_get_original_image(self):
        response = self.client.get(f"/image/original/{self.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_get_predicted_image(self):
        response = self.client.get(f"/image/predicted/{self.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_invalid_type(self):
        response = self.client.get(f"/image/invalidtype/{self.filename}")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image type")

    def test_image_not_found(self):
        response = self.client.get(f"/image/original/nonexistent.jpg")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Image not found")
