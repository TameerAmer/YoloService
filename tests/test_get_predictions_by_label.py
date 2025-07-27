import datetime
import sqlite3
import unittest
import uuid
from fastapi.testclient import TestClient

from app import DB_PATH, app, verify_user

def mock_verify_user(credentials=None, db=None):
    # Ignore credentials and just return a fixed username
    return "mockuser"

class TestGetPredictionByLabel(unittest.TestCase):

    def setUp(self):
        app.dependency_overrides[verify_user] = mock_verify_user
        self.client = TestClient(app)

    def test_label_not_found(self):
        response2 = self.client.get("/predictions/label/asd")
        self.assertEqual(response2.status_code, 200)

    
    def test_get_uid(self):
        uid1 = str(uuid.uuid4())

        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp,username) VALUES (?, ?, ?)", (uid1, now, "mockuser"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid1, "cat"))
            conn.commit()
        response = self.client.get("/predictions/label/cat")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        uids = [item.get("uid") for item in data]
        self.assertIn(uid1, uids)
