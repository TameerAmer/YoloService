import datetime
import sqlite3
import unittest
import uuid
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

    
    def test_delete_prediction_not_found(self):
        
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        headers = get_basic_auth_header(self.username, self.password)
        uid = response.json()["prediction_uid"]

        if response.json()["detection_count"] == 0:
            self.assertEqual(self.client.delete(f"/prediction/{uid}", headers=headers).status_code, 404)

    
    def test_delete_prediction_success(self):
        # Make a prediction first
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        headers = get_basic_auth_header(self.username, self.password)
        uid = response.json()["prediction_uid"]

        # Force an artificial detection so delete won't 404
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid, "dog"))
            conn.commit()

        # Now delete should succeed
        response2 = self.client.delete(f"/prediction/{uid}", headers=headers)
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response2.json()["message"], "Successfully Deleted")

    def test_delete_prediction_rowcount_zero(self):
        headers = get_basic_auth_header(self.username, self.password)
        fake_uid = str(uuid.uuid4())

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (fake_uid, "ghost"))
            conn.commit()

        response = self.client.delete(f"/prediction/{fake_uid}", headers=headers)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")

    def test_delete_prediction_session_missing(self):
        headers = get_basic_auth_header(self.username, self.password)
        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        uid = "existing_but_file_missing"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)", (uid, now))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid, "cat"))
            conn.commit()
        response = self.client.delete(f"/prediction/{uid}",headers=headers)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction file not found")





