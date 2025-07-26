import datetime
import sqlite3
import unittest
import uuid
from fastapi.testclient import TestClient

from app import app, DB_PATH, verify_user

def mock_verify_user(credentials=None, db=None):
    # Ignore credentials and just return a fixed username
    return "mockuser"

class TestUniqueLabels(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[verify_user] = mock_verify_user
        self.client = TestClient(app)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects")
            conn.execute("DELETE FROM prediction_sessions")
            conn.commit()

    def test_unique_labels_zero(self):

        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data.get("labels")), 0)

    def test_unique(self):
        uid1 = str(uuid.uuid4())
        uid2 = str(uuid.uuid4())

        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp, username) VALUES (?, ?, ?)", (uid1, now, "mockuser"))
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp, username) VALUES (?, ?, ?)", (uid2, now, "mockuser"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid1, "cat"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid2, "dog"))
            conn.commit()
        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("labels"), ['cat', 'dog'])
    
    def test_not_unique(self):
        uid1 = str(uuid.uuid4())
        uid2 = str(uuid.uuid4())

        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp, username) VALUES (?, ?, ?)", (uid1, now, "mockuser"))
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp, username) VALUES (?, ?, ?)", (uid2, now, "mockuser"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid1, "cat"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid2, "cat"))
            conn.commit()


        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("labels"), ['cat'])
