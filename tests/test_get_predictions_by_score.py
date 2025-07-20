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

class TestGetPredictionByScore(unittest.TestCase):

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

    
    def test_get_uid(self):
        uid1 = str(uuid.uuid4())

        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        score=0.20
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)", (uid1, now))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label, score) VALUES (?, ? ,?)", (uid1, "cat",score))
            conn.commit()
        headers = get_basic_auth_header(self.username, self.password)
        response = self.client.get(f"/predictions/score/{score}",headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data[0].get("uid"), uid1)