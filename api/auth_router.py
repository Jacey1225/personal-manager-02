from fastapi import FastAPI, HTTPException, APIRouter
from src.google_calendar.enable_google_api import ConfigureGoogleAPI
from pydantic import BaseModel
import os
import uuid
import json

router = APIRouter()

class OAuthCompleteRequest(BaseModel):
    user_id: str
    authorization_code: str

@router.get("/auth/signup")
def signup(username: str, email: str, password: str):
    if not os.path.exists('data/users'):
        os.makedirs('data/users')

    # Check if user already exists
    if os.path.exists(f'data/users/{username}.json'):
        return {"status": "failed", "message": "Username already exists"}

    user_data = {
        "username": username,
        "email": email,
        "password": password,
        "user_id": str(uuid.uuid4()),
    }

    with open(f'data/users/{username}.json', 'w') as f:
        f.write(json.dumps(user_data))

    return {"status": "success", "user_id": user_data.get("user_id")}

@router.get("/auth/login")
def login(username: str, password: str):
    user_file = f'data/users/{username}.json'
    if not os.path.exists(user_file):
        return {"status": "failed", "message": "User not found"}

    with open(user_file, 'r') as f:
        user_data = json.load(f)
        if user_data.get("password") == password:
            return {"status": "success", "user_id": user_data.get("user_id")}
        else:
            return {"status": "failed", "message": "Invalid password"}

@router.get("/auth/google")
def google_auth(user_id: str):
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

@router.post("/auth/google/complete")
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

