import unittest
from fastapi.testclient import TestClient
import base64

from app import app

def get_basic_auth_header(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Register test user
        self.username = "tameer"
        self.password = "1234"
        auth_header = get_basic_auth_header(self.username, self.password)
        self.client.post("/register", headers=auth_header)


    def test_no_auth(self):
        response = self.client.get("/prediction/count")
        # Check response
        self.assertEqual(response.status_code, 401)

    def test_wrong_password(self):
        headers = get_basic_auth_header(self.username, "1")
        response = self.client.get("/prediction/count")
        # Check response
        self.assertEqual(response.status_code, 401)
        data=response.json()
        self.assertEqual(data.get("detail"),"Not authenticated")
    
    def test_sucess_auth(self):
        headers = get_basic_auth_header(self.username, self.password)
        response = self.client.get("/prediction/count",headers=headers)
        # Check response
        self.assertEqual(response.status_code, 200)
        data=response.json()
        self.assertEqual(data.get("count"), 0)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        # Check response
        self.assertEqual(response.status_code, 200)
        data=response.json()
        self.assertEqual(data.get("status"), "ok")

    def test_exception_handle_verify_user(self):
        headers = get_basic_auth_header("", "")
        response = self.client.get("/prediction/count",headers=headers)
        self.assertEqual(response.status_code, 401)
        data=response.json()
        self.assertEqual(data.get("detail"),"Invalid username or password")

     

        