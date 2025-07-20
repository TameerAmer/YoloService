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

class TestAuth(unittest.TestCase):
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

    def test_no_auth(self):
        response = self.client.get("/prediction/count")
        # Check response
        self.assertEqual(response.status_code, 401)

    def test_wrong_password(self):
        headers = get_basic_auth_header(self.username, "1")
        response = self.client.get("/prediction/count")
        # Check response
        self.assertEqual(response.status_code, 401)
        data=response.json()
        self.assertEqual(data.get("detail"),"Not authenticated")
    
    def test_sucess_auth(self):
        headers = get_basic_auth_header(self.username, self.password)
        response = self.client.get("/prediction/count",headers=headers)
        # Check response
        self.assertEqual(response.status_code, 200)
        data=response.json()
        self.assertEqual(data.get("count"), 0)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        # Check response
        self.assertEqual(response.status_code, 200)
        data=response.json()
        self.assertEqual(data.get("status"), "ok")

    def test_exception_handle_verify_user(self):
        headers = get_basic_auth_header("", "")
        response = self.client.get("/prediction/count",headers=headers)
        self.assertEqual(response.status_code, 401)
        data=response.json()
        self.assertEqual(data.get("detail"),"Invalid username or password")

     

        