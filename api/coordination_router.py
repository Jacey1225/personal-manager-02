from src.track_projects.coordinate_datetimes import CoordinateDateTimes
from fastapi import APIRouter, HTTPException
from datetime import datetime
import json
import os

router = APIRouter()

@router.post("/coordinate/fetch_users")
def fetch_users(request_body: dict):
    """Fetch user data from the local JSON files.

    Args:
        members (list[dict]): A list of dictionaries containing email and username to fetch data for.

    Returns:
        dict: A dictionary containing the fetched user data.
    """
    users = []
    with open("data/user_log.json", "r") as f:
        user_log = json.load(f)

    members = request_body.get("members", [])
    for member in members:
        email = member.get("email")
        username = member.get("username")
        if username in user_log:
            user_id = user_log[username]
            with open(f"data/users/{user_id}.json", "r") as f:
                user_data = json.load(f)
                users.append(user_data)

    print(f"Users fetched: {users}")
    return {"users": users}

@router.post("/coordinate/get_availability")
def get_availability(request_body: dict):
    """Get the availability of users for a specific time range.

    Args:
        request_body (dict): The request body containing users, request_start, and request_end.

    Raises:
        HTTPException: If no users are available for the requested time range.

    Returns:
        dict: A dictionary containing the status and available users.
    """
    users = request_body.get("users", [])
    request_start = request_body.get("request_start", "")
    request_end = request_body.get("request_end", "")
    
    print(f"Checking availability for users: {users} from {request_start} to {request_end}")
    
    if not users:
        raise HTTPException(status_code=400, detail="No users provided")
    if not request_start or not request_end:
        raise HTTPException(status_code=400, detail="Missing request_start or request_end")
    
    user_availability = []
    for user in users:
        try:
            request_start_obj = datetime.fromisoformat(request_start)
            request_end_obj = datetime.fromisoformat(request_end)
            coordinator = CoordinateDateTimes(user['user_id'], request_start_obj, request_end_obj)
            user_availability.append((user["username"], coordinator.coordinate()))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid datetime format: {e}")
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing required user field: {e}")

    if len(user_availability) == 0:
        raise HTTPException(status_code=404, detail="No users are available for the requested time range.")

    percent_available = (len([username for username, available in user_availability if available]) / len(users)) * 100
    print(f"User availability: {user_availability}, Percent available: {percent_available}%")
    return {"status": "success", "users": user_availability, "percent_available": percent_available}