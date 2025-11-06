import sys
import os
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.main import app

client = TestClient(app)

# This file can be used for application-level tests.

# To run the tests, install pytest and pytest-mock:
# pip install pytest pytest-mock
#
# Then, from the root directory of the project, run:
# pytest api/tests