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

def test_read_root():
    """
    Test the root endpoint, which is defined in the event_router.
    """
    logger.info("Starting test_read_root")
    response = client.get("/")
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == {"Hello": "Jacey"}
    logger.info("Finished test_read_root")

@patch('api.routes.event_router.commander.generate_events', new_callable=AsyncMock)
@patch('api.routes.event_router.commander.process_input', new_callable=AsyncMock)
def test_process_input(mock_process_input, mock_generate_events):
    logger.info("Starting test_process_input")
    mock_generate_events.return_value = [{"event": "some_event"}]
    mock_process_input.return_value = [{"status": "completed"}]
    
    request_data = {"input_text": "some text", "user_id": "test_user"}
    response = client.post("/scheduler/process_input", json=request_data)
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == [{"status": "completed"}]
    logger.info("Finished test_process_input")

@patch('api.routes.event_router.commander.delete_event', new_callable=AsyncMock)
def test_delete_event(mock_delete_event):
    logger.info("Starting test_delete_event")
    mock_delete_event.return_value = {"status": "success"}
    
    response = client.post("/scheduler/delete_event/some_event_id", json={})
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    logger.info("Finished test_delete_event")

@patch('api.routes.event_router.commander.update_event', new_callable=AsyncMock)
def test_update_event(mock_update_event):
    logger.info("Starting test_update_event")
    mock_update_event.return_value = {"status": "success"}
    
    response = client.post("/scheduler/update_event/some_event_id", json={})
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    logger.info("Finished test_update_event")