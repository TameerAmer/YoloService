import base64
import unittest
import os
import sqlite3
from fastapi.testclient import TestClient
from PIL import Image
from app import app, DB_PATH, verify_user  


def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestGetPredictionImage(unittest.TestCase):
    def setUp(self):
        self.username = "admin"
        self.test_uid = "test123"
        self.test_image_path = f"uploads/predicted/{self.test_uid}.jpg"
        os.makedirs("uploads/predicted", exist_ok=True)

        #  OVERRIDE verify_user to mock authentication
        app.dependency_overrides[verify_user] = lambda: self.username

        self.client = TestClient(app)

        # Create dummy image
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(self.test_image_path, format="JPEG")

        # Insert DB row with username
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.test_uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, predicted_image, username) VALUES (?, ?, ?)",
                (self.test_uid, self.test_image_path, self.username)
            )
            conn.commit()


    def tearDown(self):
        # Clean up DB and files
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.test_uid,))
            conn.commit()
    
        # Clear overrides to not affect other tests
        app.dependency_overrides = {}

    def test_get_prediction_image_jpeg(self):
        response = self.client.get(
            f"/prediction/{self.test_uid}/image",
            headers={"accept": "image/jpeg"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertGreater(len(response.content), 0)

    def test_get_prediction_image_unsupported_accept(self):
        response = self.client.get(
            f"/prediction/{self.test_uid}/image",
            headers={"accept": "application/json"}
        )
        self.assertEqual(response.status_code, 406)

    def test_get_prediction_image_not_found(self):
        response = self.client.get(
            f"/prediction/doesnotexist/image",
            headers={"accept": "image/jpeg"}
        )
        self.assertEqual(response.status_code, 404)

    def test_get_prediction_image_png(self):
        # Prepare dummy PNG image and DB row
        image_path = "uploads/predicted/test_image.png"
        os.makedirs("uploads/predicted", exist_ok=True)
        image = Image.new("RGB", (100, 100), color="green")
        image.save(image_path, format="PNG")

        uid = "test-uid-png"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, predicted_image, username) VALUES (?, ?, ?)",
                (uid, image_path, self.username)
            )
            conn.commit()

        response = self.client.get(
            f"/prediction/{uid}/image",
            headers={
                "Accept": "image/png"}
        )

        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
            conn.commit()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_get_prediction_image_file_not_found_on_disk(self):
        uid = "missing-image-uid"
        fake_path = f"predicted/{uid}.jpg"

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, predicted_image, username) VALUES (?, ?, ?)",
                (uid, fake_path, self.username)
            )
            conn.commit()

        response = self.client.get(
            f"/prediction/{uid}/image",
            headers={"Accept": "image/jpeg"}
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("Predicted image file not found", response.text)
