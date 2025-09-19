from fastapi import FastAPI, HTTPException, APIRouter
from src.google_calendar.enable_google_api import ConfigureGoogleAPI
from pydantic import BaseModel
import os
import uuid
import json

auth_router = APIRouter()

class OAuthCompleteRequest(BaseModel):
    user_id: str
    authorization_code: str

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
    if not os.path.exists('data/users'):
        os.makedirs('data/users')
    user_id = str(uuid.uuid4())

    # Check if user already exists
    while os.path.exists(f'data/users/{user_id}.json'):
        user_id = str(uuid.uuid4())
    files = os.listdir('data/users')
    for filename in files:
        with open(f'data/users/{filename}', 'r') as f:
            existing_user_data = json.load(f)
            if existing_user_data.get("email") == email or existing_user_data.get("username") == username:
                return {"status": "failed", "message": "Email or username already exists"}

    user_data = {
        "username": username,
        "email": email,
        "password": password,
        "user_id": user_id,
    }

    with open(f'data/users/{user_id}.json', 'w') as f:
        f.write(json.dumps(user_data))
    
    with open(f'data/user_log.json', 'w') as f:
        f.write(json.dumps({username: user_id}))

    return {"status": "success", "user_id": user_data.get("user_id")}

@auth_router.get("/auth/login")
def login(username: str, password: str) -> dict:
    """Log in an existing user.

    Args:
        username (str): Username of the user.
        password (str): Password of the user.

    Returns:
        dict: A dictionary containing the login status and user ID if successful.
    """
    with open('data/user_log.json', 'r') as f:
        user_log = json.load(f)
    user_file = None
    if username in user_log:
        user_id = user_log[username]
        user_file = f'data/users/{user_id}.json' 
    if not user_file:
        return {"status": "failed", "message": "User not found"}

    with open(user_file, 'r') as f:
        user_data = json.load(f)
        if user_data.get("password") == password:
            return {"status": "success", "user_id": user_data.get("user_id")}
        else:
            return {"status": "failed", "message": "Invalid password"}

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

