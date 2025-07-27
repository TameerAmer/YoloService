import datetime
import sqlite3
import unittest
import uuid
from fastapi.testclient import TestClient
from PIL import Image
import io
from app import app,DB_PATH, verify_user


def mock_verify_user(credentials=None, db=None):
    # Ignore credentials and just return a fixed username
    return "mockuser"

class TestGetPredictionByScore(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides[verify_user] = mock_verify_user
        self.client = TestClient(app)
    
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    
    def test_get_uid(self):
        uid1 = str(uuid.uuid4())

        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        score=0.20
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp,username) VALUES (?, ?, ?)", (uid1, now, "mockuser"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label, score) VALUES (?, ? ,?)", (uid1, "cat",score))
            conn.commit()
        response = self.client.get(f"/predictions/score/{score}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        uids = [item.get("uid") for item in data]
        self.assertIn(uid1, uids)