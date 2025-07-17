import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from app import app, init_db,DB_PATH
import base64

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class Test_Delete(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        
        # Register test user
        self.username = "tameer"
        self.password = "1234"
        auth_header = get_basic_auth_header(self.username, self.password)
        self.client.post("/register", headers=auth_header)

        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_detete_not_found(self):
        headers = get_basic_auth_header(self.username, self.password)
        response2 = self.client.delete("/prediction/-1",headers=headers)
        self.assertEqual(response2.status_code, 404)
        self.assertEqual(response2.json()["detail"], "Prediction not found")

    
    def test_delete_prediction(self):
        
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        headers = get_basic_auth_header(self.username, self.password)
        uid = response.json()["prediction_uid"]

        if response.json()["detection_count"] == 0:
            self.assertEqual(self.client.delete(f"/prediction/{uid}", headers=headers).status_code, 404)
        else:
            response2 = self.client.delete(f"/prediction/{uid}", headers=headers)
            self.assertEqual(response2.status_code, 200)
            self.assertEqual(response2.json()["message"], "Successfully Deleted")
