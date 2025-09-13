from src.coordinate_datetimes.coordinate_datetimes import CoordinateDateTimes
from fastapi import APIRouter, HTTPException
from datetime import datetime
import json
import os

router = APIRouter()

@router.get("/coordinate/fetch-users")
def fetch_users(usernames: list[str]):
    """Fetch user data from the local JSON files.

    Args:
        usernames (list[str]): A list of usernames to fetch data for.

    Returns:
        dict: A dictionary containing the fetched user data.
    """
    users = []
    for username in usernames:
        if not os.path.exists(f'data/users/{username}.json'):
            continue
        with open(f'data/users/{username}.json', 'r') as f:
            user_data = json.load(f)
            users.append(user_data)

    return {"users": users}

@router.post("/coordinate/get-availability")
def get_availability(users: list[dict], request_start: str, request_end: str):
    """Get the availability of users for a specific time range.

    Args:
        users (list[dict]): A list of user dictionaries containing their information.
        request_start (str): The start time of the availability request (ISO format).
        request_end (str): The end time of the availability request (ISO format).

    Raises:
        HTTPException: If no users are available for the requested time range.

    Returns:
        dict: A dictionary containing the status and available users.
    """
    user_availability = []
    for user in users:
        request_start_obj = datetime.fromisoformat(request_start)
        request_end_obj = datetime.fromisoformat(request_end)
        coordinator = CoordinateDateTimes(user['user_id'], request_start_obj, request_end_obj)
        user_availability.append((user["username"], coordinator.coordinate()))

    if len(user_availability) == 0:
        raise HTTPException(status_code=404, detail="No users are available for the requested time range.")

    return {"status": "success", "users": user_availability}