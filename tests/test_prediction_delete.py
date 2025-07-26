import datetime
import sqlite3
import unittest
import uuid
from fastapi.testclient import TestClient
from PIL import Image
import io
from app import app,DB_PATH,get_db
import base64

from db import SessionLocal
from models import PredictionSession,DetectionObjects

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class Test_Delete(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.db = SessionLocal()

        # Override FastAPI dependency to use test session
        def override_get_db():
            try:
                yield cls.db
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(app)

        cls.username = "mockuser"
        cls.password = "1234"

        cls.test_image = Image.new('RGB', (100, 100), color='red')
        cls.image_bytes = io.BytesIO()
        cls.test_image.save(cls.image_bytes, format='JPEG')
        cls.image_bytes.seek(0)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        app.dependency_overrides.clear()

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
        self.image_bytes.seek(0)
        # Make a prediction first
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        headers = get_basic_auth_header(self.username, self.password)
        uid = response.json()["prediction_uid"]

        # Force insert username and detection manually
        session = self.db.query(PredictionSession).filter_by(uid=uid).first()
        self.assertIsNotNone(session)
        session.username = self.username
        self.db.add(DetectionObjects(prediction_uid=uid, label="dog"))
        self.db.commit()

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
        now = datetime.datetime.now(datetime.timezone.utc)

        uid = "existing_but_file_missing"
        with SessionLocal() as db:
            # Clear if exists
            self.db.query(DetectionObjects).filter_by(prediction_uid=uid).delete()
            self.db.query(PredictionSession).filter_by(uid=uid).delete()
            self.db.commit()

            # Insert via ORM
            ps = PredictionSession(uid=uid, timestamp=now, username="mockuser")
            self.db.add(ps)
            self.db.add(DetectionObjects(prediction_uid=uid, label="cat"))
            self.db.commit()
        response = self.client.delete(f"/prediction/{uid}",headers=headers)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction file not found")




