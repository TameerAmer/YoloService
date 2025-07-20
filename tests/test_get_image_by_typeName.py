import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from app import app, init_db, DB_PATH
import base64

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestGetImage(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()

        self.username = "testuser"
        self.password = "1234"
        auth_header = get_basic_auth_header(self.username, self.password)
        self.client.post("/register", headers=auth_header)

        self.headers = auth_header

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

    def test_get_original_image(self):
        response = self.client.get(f"/image/original/{self.filename}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_get_predicted_image(self):
        response = self.client.get(f"/image/predicted/{self.filename}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_invalid_type(self):
        response = self.client.get(f"/image/invalidtype/{self.filename}", headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image type")

    def test_image_not_found(self):
        response = self.client.get(f"/image/original/nonexistent.jpg", headers=self.headers)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Image not found")
