import typing
from fastapi import FastAPI, HTTPException, APIRouter
from src.google_calendar.enable_google_api import ConfigureGoogleAPI
from pydantic import BaseModel
import os
import uuid
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Optional
from src.fetchMongo import MongoHandler

mongo_client = MongoHandler("userCredentials")
auth_router = APIRouter()

class OAuthCompleteRequest(BaseModel):
    user_id: str
    authorization_code: str

class RemoveUserRequest(BaseModel):
    user_id: str

@auth_router.get("/auth/signup")
def signup(username: str, email: str, password: str):
    """Sign up a new user.

    Args:
        username (str): Username for the new user.
        email (str): Email address for the new user.
        password (str): Password for the new user.

    Returns:
        dict: A dictionary containing the status and user ID.
    """

    # Check if user already exists
    user_id = str(uuid.uuid4())
    while mongo_client.get_single_doc({"user_id": user_id}):
        user_id = str(uuid.uuid4())

    query_email = {"email": email}
    query_username = {"username": username}

    if mongo_client.get_single_doc(query_email) or mongo_client.get_single_doc(query_username):
        return {"status": "failed", "message": "Email or username already exists"}

    user_data = {
        "username": username,
        "email": email,
        "password": password,
        "user_id": user_id,
    }

    result = mongo_client.post_insert(user_data)
    print(f"User created with ID: {result.inserted_id}")

    return {"status": "success", "user_id": user_data.get("user_id")}

@auth_router.get("/auth/login")
def login(username: str, password: str) -> Optional[dict]:
    """Log in an existing user.

    Args:
        username (str): Username of the user.
        password (str): Password of the user.

    Returns:
        dict: A dictionary containing the login status and user ID if successful.
    """
    query_login = {
        "$expr": {
            "$and": [
                {"$eq": ["$username", username]},
                {"$eq": ["$password", password]}
            ]
        }
    }

    try:
        if mongo_client.get_single_doc(query_login):
            user_data = mongo_client.get_single_doc(query_login)
            user_id = user_data.get("user_id") #type: ignore
            print(f"User {username} logged in successfully with user_id: {user_id}")
            print(f"User data: {user_data}")
            return {"status": "success", "user_id": user_id}
        else:
            return {"status": "failed", "message": "Invalid username or password"}
    except Exception as e:
        print(f"Error in login: {str(e)}")
        return {"status": "failed", "message": "Invalid username or password"}

@auth_router.post("/auth/remove_user")
def remove_user(request: RemoveUserRequest) -> dict:
    """Remove a user.

    Args:
        request (RemoveUserRequest): The request containing the user ID to remove.

    Returns:
        dict: A dictionary containing the removal status.
    """
    try:
        user_id = request.user_id
        mongo_client.post_delete({"user_id": user_id})
        
        # Try to remove the token file, but don't fail if it doesn't exist
        token_file = f"data/tokens/token_{user_id}.json"
        if os.path.exists(token_file):
            os.remove(token_file)
        
        return {"status": "success", "message": "User removed successfully"}
    except Exception as e:
        print(f"Error in remove_user: {str(e)}")
        return {"status": "failed", "message": "Error removing user"}

@auth_router.get("/auth/google")
def google_auth(user_id: str) -> dict:
    """Get Google OAuth authorization URL for the user"""
    try:
        print(f"Google auth request for user_id: {user_id}")
        
        if not os.path.exists('data/credentials.json'):
            raise HTTPException(
                status_code=500, 
                detail="Google credentials file not found. Please ensure credentials.json is in the data directory."
            )
        
        google_api = ConfigureGoogleAPI(user_id)
        result = google_api.enable_google_calendar_api()
        
        print(f"Result type: {type(result)}, Result: {result}")
        
        if isinstance(result, str):  
            return {"status": "auth_required", "auth_url": result}
        elif result is not None and len(result) == 2:  
            return {"status": "already_authenticated", "message": "User already has valid Google credentials"}
        else:
            raise HTTPException(status_code=500, detail="Unexpected result from Google API setup")
            
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Error in google_auth: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing Google auth: {str(e)}")

@auth_router.post("/auth/google/complete")
def complete_google_auth(request: OAuthCompleteRequest):
    """Complete Google OAuth flow with authorization code"""
    try:
        print(f"Completing Google auth for user_id: {request.user_id}")
        print(f"Authorization code length: {len(request.authorization_code)}")
        
        google_api = ConfigureGoogleAPI(request.user_id)
        result = google_api.complete_auth_flow(request.authorization_code)
        
        if result is not None and len(result) == 2:
            return {"status": "success", "message": "Google authentication completed successfully"}
        else:
            return {"status": "failed", "message": "Failed to complete Google authentication"}
            
    except Exception as e:
        print(f"Error completing Google auth: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error completing Google auth: {str(e)}")

