import base64
import unittest
import uuid
from fastapi.testclient import TestClient

from app import app

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestRegister(unittest.TestCase):
    def setUp(self):
        # Register test user
        self.client = TestClient(app)
        self.username = "tameer"
        self.password = "1234"
        self.auth_header = get_basic_auth_header(self.username, self.password)
        self.client.post("/register", headers=self.auth_header)

    def test_existing_user(self):
        response = self.client.post("/register",headers=self.auth_header)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Username already exists")

    def test_register_success(self):
        new_username = f"user_{uuid.uuid4()}"
        new_password = "newpass"
        new_auth_header = get_basic_auth_header(new_username, new_password)
        response = self.client.post("/register", headers=new_auth_header)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "User registered successfully"})


    
