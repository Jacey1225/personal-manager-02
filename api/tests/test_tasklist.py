import sys
import os
import logging
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.main import app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = TestClient(app)

@patch('api.routes.tasklist_router.commander.list_events', new_callable=AsyncMock)
def test_list_events(mock_list_events):
    logger.info("Starting test_list_events")
    mock_list_events.return_value = [{"event_name": "Test Event"}]
    
    request_data = {
        "calendar_event": {
            "event_name": "Test Event"
        },
        "user_id": "test_user"
    }
    
    response = client.post("/task_list/list_events", json=request_data)
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == [{"event_name": "Test Event"}]
    logger.info("Finished test_list_events")