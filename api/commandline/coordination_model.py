from src.track_projects.coordinate_datetimes import CoordinateDateTimes
from fastapi import APIRouter, HTTPException
from datetime import datetime
import json
import os
from src.fetchMongo import MongoHandler

mongo_client = MongoHandler()

class CoordinationModel:
    @staticmethod
    async def fetch_users(request_body: dict):
        """Fetch user data from the local JSON files.

        Args:
            members (list[dict]): A list of dictionaries containing email and username to fetch data for.

        Returns:
            dict: A dictionary containing the fetched user data.
        """
        users = []
        members = request_body.get("members", [])
        for member in members:
            query_user = {"email": member.get("email").lower()}
            user_data = mongo_client.get_single_doc(query_user)
            all_users = mongo_client.get_all()
            print(f"All users in database: {all_users}")
            if user_data:
                users.append(user_data)

        print(f"Users fetched: {users}")
        return {"users": users}
    
    @staticmethod
    async def get_availability(request_body: dict):
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