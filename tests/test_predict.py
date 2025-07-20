import os
import unittest
from fastapi.testclient import TestClient
from PIL import Image,ImageDraw
import io
import base64

from app import app, DB_PATH, init_db

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestPredict(unittest.TestCase):
    def setUp(self):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        self.client = TestClient(app)

        init_db()

        # Register test user
        self.username = "tameer"
        self.password = "1234"
        auth_header = get_basic_auth_header(self.username, self.password)
        self.client.post("/register", headers=auth_header)

        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_predict_with_invalid_credentials(self):
        # Corrupt the password or use wrong auth
        wrong_auth = get_basic_auth_header("wronguser", "wrongpass")
        
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")},
            headers=wrong_auth
        )

        self.assertEqual(response.status_code, 200)  # prediction still works
        data = response.json()
        self.assertIn("prediction_uid", data)
        self.assertIsInstance(data["prediction_uid"], str)

    def test_predict(self):
        image = Image.new("RGB", (200, 200), "white")
        draw = ImageDraw.Draw(image)

        # Draw the seat (rectangle)
        draw.rectangle([(50, 100), (150, 130)], fill="brown")  # Seat

        # Draw the backrest (rectangle)
        draw.rectangle([(50, 60), (150, 100)], fill="brown")  # Backrest

        # Save to BytesIO for testing upload
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200) 
        data = response.json()
        self.assertGreater(data.get("detection_count"), 0)
    
    def test_predict_without_auth(self):
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_uid", response.json())
    
    def test_predict_with_valid_auth(self):
        headers = get_basic_auth_header(self.username, self.password)
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")},
            headers=headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_uid", response.json())

