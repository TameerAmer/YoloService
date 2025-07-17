import os
import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import base64

from app import app, DB_PATH, init_db

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestProcessingCount(unittest.TestCase):
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

    def test_prediction_count_empty(self):
        headers = get_basic_auth_header(self.username, self.password)
        response = self.client.get("/prediction/count", headers=headers)
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("count"), 0)
        
    def test_prediction_count_after_prediction(self):
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        headers = get_basic_auth_header(self.username, self.password)
        response2= self.client.get("/prediction/count", headers=headers)
        # Check response
        self.assertEqual(response2.status_code, 200)
        data = response2.json()
        self.assertEqual(data.get("count"), 1)
    

        