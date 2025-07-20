import base64
import unittest
import os
import sqlite3
from fastapi.testclient import TestClient
from PIL import Image
import io
from app import app, DB_PATH

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestGetPredictionImage(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        self.username = "admin"
        self.password = "admin"
        auth_header = get_basic_auth_header(self.username, self.password)
        self.client.post("/register", headers=auth_header)

        # Create test image
        self.test_uid = "test123"
        self.test_image_path = f"uploads/predicted/{self.test_uid}.jpg"
        os.makedirs("uploads/predicted", exist_ok=True)

        image = Image.new("RGB", (100, 100), color="blue")
        image.save(self.test_image_path, format="JPEG")

        # Insert test DB row
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.test_uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, predicted_image) VALUES (?, ?)",
                (self.test_uid, self.test_image_path)
            )
            conn.commit()

        self.auth = (self.username, self.password)  # Change if needed

    def tearDown(self):
        # Cleanup image and DB entry
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (self.test_uid,))
            conn.commit()

    def test_get_prediction_image_jpeg(self):
        response = self.client.get(
            f"/prediction/{self.test_uid}/image",
            headers={"accept": "image/jpeg"},
            auth=self.auth
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertGreater(len(response.content), 0)

    def test_get_prediction_image_unsupported_accept(self):
        response = self.client.get(
            f"/prediction/{self.test_uid}/image",
            headers={"accept": "application/json"},
            auth=self.auth
        )
        self.assertEqual(response.status_code, 406)

    def test_get_prediction_image_not_found(self):
        response = self.client.get(
            f"/prediction/doesnotexist/image",
            headers={"accept": "image/jpeg"},
            auth=self.auth
        )
        self.assertEqual(response.status_code, 404)

    def test_get_prediction_image_png(self):
        # Prepare: Create dummy image file
        image_path = "uploads/predicted/test_image.png"
        image = Image.new("RGB", (100, 100), color="green")
        image.save(image_path, format="PNG")


        # Insert dummy prediction session into DB
        uid = "test-uid-png"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))  # <- ADD this line
            conn.execute(
                "INSERT INTO prediction_sessions (uid, predicted_image, username) VALUES (?, ?, ?)",
                (uid, image_path, "admin")
            )
            conn.commit()

        # Act: Make GET request with Accept: image/png
        response = self.client.get(
            f"/prediction/{uid}/image",
            headers={
                "Accept": "image/png",
                "Authorization": "Basic " + base64.b64encode(b"admin:admin").decode("utf-8")
            }
        )

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_get_prediction_image_file_not_found_on_disk(self):
        # Use a new uid and predicted_image path
        uid = "missing-image-uid"
        fake_path = f"predicted/{uid}.jpg"

        # Insert the DB record, but do NOT create the image file
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
            conn.execute(
                "INSERT INTO prediction_sessions (uid, predicted_image, username) VALUES (?, ?, ?)",
                (uid, fake_path, self.username)
            )
            conn.commit()

        # Perform request
        response = self.client.get(
            f"/prediction/{uid}/image",
            headers={"Accept": "image/jpeg"},
            auth=self.auth
        )

        # Assertions
        self.assertEqual(response.status_code, 404)
        self.assertIn("Predicted image file not found", response.text)


