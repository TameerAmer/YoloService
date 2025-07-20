import base64
import datetime
import os
import sqlite3
import unittest
import uuid
from fastapi.testclient import TestClient
from PIL import Image
import io

from app import app, DB_PATH, init_db

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
        response = self.client.post("register",headers=self.auth_header)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Username already exists")


    
