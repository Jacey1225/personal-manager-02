from fastapi import FastAPI, HTTPException, APIRouter
from api.config.extensions.enable_google_api import ConfigureGoogleAPI
from api.schemas.auth import OAuthCompleteRequest, RemoveUserRequest
from api.config.fetchMongo import MongoHandler
import os
import uuid
import json
from dotenv import load_dotenv
from typing import Optional

user_handler = None
project_handler = None
auth_router = APIRouter()

def set_handlers(_user_handler, _project_handler):
    global user_handler, project_handler
    user_handler = _user_handler
    project_handler = _project_handler


user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
project_config = MongoHandler(None, "userProjectsDatabase", "projects")

@auth_router.on_event("startup")
async def startup_event():
    await user_config.get_client()
    await project_config.get_client()
    set_handlers(user_config, project_config)

@auth_router.get("/auth/signup")
async def signup(
    username: str, 
    email: str, 
    password: str, 
    project_id: Optional[str]=None, 
    org_id: Optional[str]=None) -> dict:
    """Sign up a new user.

    Args:
        username (str): Username for the new user.
        email (str): Email address for the new user.
        password (str): Password for the new user.
        project_id (Optional[str], optional): Project ID to associate with the user. Defaults to None.
        org_id (Optional[str], optional): Organization ID to associate with the user. Defaults to None.

    Returns:
        dict: A dictionary containing the status and user ID.
    """
    # Check if user already exists
    user_id = str(uuid.uuid4())
    user_data = await user_handler.get_single_doc({"user_id": user_id})
    while user_data:
        user_id = str(uuid.uuid4())
        user_data = await user_handler.get_single_doc({"user_id": user_id})

    query_email = {"email": email}
    query_username = {"username": username}

    if await user_handler.get_single_doc(query_email) or \
       await user_handler.get_single_doc(query_username):
        return {"status": "failed", "message": "Email or username already exists"}

    user_data = {
        "username": username,
        "email": email,
        "password": password,
        "user_id": user_id,
        "projects": {},
        "organizations": []
    }

    if project_id:
        project = await project_handler.get_single_doc({"project_id": project_id})
        if project:
            user_data["projects"][project_id] = (project.get("project_name"), "view")

    if org_id:
        user_data["organizations"].append(org_id)

    result = await user_handler.post_insert(user_data)
    print(f"User created with ID: {result.inserted_id}")

    return {"status": "success", "user_id": user_data.get("user_id")}

@auth_router.get("/auth/login")
async def login(
    username: str, 
    password: str, 
    project_id: Optional[str]=None, 
    org_id: Optional[str]=None) -> Optional[dict]:
    """Log in an existing user.

    Args:
        username (str): Username of the user.
        password (str): Password of the user.
        project_id (Optional[str], optional): Project ID to associate with the user upon login. Defaults to None.
        org_id (Optional[str], optional): Organization ID to associate with the user upon login. Defaults to None.

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
        user_data = await user_handler.get_single_doc(query_login)
        if user_data:
            if project_id and project_id not in user_data.get("projects", {}):
                project = await project_handler.get_single_doc({"project_id": project_id})
                user_data["projects"][project_id] = (project.get("project_name"), "view")

            if org_id and org_id not in user_data.get("organizations", []):
                user_data["organizations"].append(org_id)

            user_id = user_data.get("user_id") #type: ignore
            await user_handler.post_update({"user_id": user_id}, user_data)
            print(f"User {username} logged in successfully with user_id: {user_id}")
            print(f"User data: {user_data}")
            return {"status": "success", "user_id": user_id}
        else:
            return {"status": "failed", "message": "Invalid username or password"}
    except Exception as e:
        print(f"Error in login: {str(e)}")
        return {"status": "failed", "message": "Invalid username or password"}

@auth_router.post("/auth/remove_user")
async def remove_user(request: RemoveUserRequest) -> dict:
    """Remove a user.

    Args:
        request (RemoveUserRequest): The request containing the user ID to remove.

    Returns:
        dict: A dictionary containing the removal status.
    """
    try:
        user_id = request.user_id
        await user_handler.post_delete({"user_id": user_id})
        
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

