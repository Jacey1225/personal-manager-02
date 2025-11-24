from fastapi import HTTPException, APIRouter
from api.schemas.auth import RemoveUserRequest, ICloudUserRequest
from api.config.fetchMongo import MongoHandler
import os
import uuid
import logging
from typing import Optional
import keyring

logger = logging.getLogger(__name__)

auth_router = APIRouter()
service_name = "usr_auth"

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
    # Create new instances for each request to avoid event loop issues
    user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
    project_config = MongoHandler(None, "userProjectsDatabase", "projects")
    
    # Check if user already exists
    client_connected = await user_config.get_client()
    if not client_connected:
        await user_config.close_client()
        raise HTTPException(status_code=500, detail="Unable to connect to user database")
    
    user_id = str(uuid.uuid4())
    user_data = await user_config.get_single_doc({"user_id": user_id})
    while user_data:
        user_id = str(uuid.uuid4())
        user_data = await user_config.get_single_doc({"user_id": user_id})

    query_email = {"email": email}
    query_username = {"username": username}

    if await user_config.get_single_doc(query_email) or \
       await user_config.get_single_doc(query_username):
        return {"status": "failed", "message": "Email or username already exists"}

    user_data = {
        "username": username,
        "email": email,
        "apple_email": None,
        "user_id": user_id,
        "services": {
            'google': False,
            'apple': False
        },
        "projects": {},
        "organizations": []
    }

    await project_config.get_client()
    if not project_config.client:
        await user_config.close_client()
        await project_config.close_client()
        raise HTTPException(status_code=500, detail="Unable to connect to project database")
    
    if project_id:
        project = await project_config.get_single_doc({"project_id": project_id})
        if project:
            user_data["projects"][project_id] = (project.get("project_name"), "view")

    if org_id:
        user_data["organizations"].append(org_id)

    keyring.set_password(service_name, user_id, password)
    result = await user_config.post_insert(user_data)
    logger.info(f"User created with ID: {result.inserted_id}")

    await user_config.close_client()
    await project_config.close_client()
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
    # Create new instances for each request to avoid event loop issues
    user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
    project_config = MongoHandler(None, "userProjectsDatabase", "projects")
    
    found_password = keyring.get_password(service_name, username)
    user_connected = await user_config.get_client()
    project_connected = await project_config.get_client()
    if not user_connected or not project_connected:
        await project_config.close_client()
        await user_config.close_client()
        raise HTTPException(status_code=500, detail="Unable to connect to user database")
    
    if not found_password:
        await project_config.close_client()
        await user_config.close_client()
        return {"status": "failed", "message": "Invalid username or password"}
    else:
        if found_password == password:
            user_info = await user_config.get_single_doc({"username": username})
            if project_id not in user_info.get("projects", {}):
                project = await project_config.get_single_doc({"project_id": project_id})
                if project:
                    user_info["projects"][project_id] = (project.get("project_name"), "view")
            if org_id and org_id not in user_info.get("organizations", []):
                user_info["organizations"].append(org_id)
            await user_config.post_update({"username": username}, user_info)
            await project_config.close_client()
            await user_config.close_client()
            return {"status": "success", "user_id": user_info.get("user_id")}
        else:
            await project_config.close_client()
            await user_config.close_client()
            return {"status": "failed", "message": "Invalid username or password"}

@auth_router.post("/auth/set_icloud_user")
async def set_icloud_user(request: ICloudUserRequest):
    found_pass = True if keyring.get_password(request.service_name, request.apple_user) else False
    if not found_pass:
        keyring.set_password(request.service_name, request.apple_user, request.apple_pass)
        return {"status": "success", "message": "iCloud user set successfully"}
    else:
        return {"status": "failed", "message": "iCloud user already set"}

@auth_router.post("/auth/remove_user")
async def remove_user(request: RemoveUserRequest) -> dict:
    """Remove a user.

    Args:
        request (RemoveUserRequest): The request containing the user ID to remove.

    Returns:
        dict: A dictionary containing the removal status.
    """
    try:
        # Create new instance for each request to avoid event loop issues
        user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
        
        user_connected = await user_config.get_client()
        if not user_connected:
            raise HTTPException(status_code=500, detail="Unable to connect to user database")
        user_id = request.user_id
        await user_config.post_delete({"user_id": user_id})
        await user_config.close_client()

        # Try to remove the token file, but don't fail if it doesn't exist
        token_file = f"data/tokens/token_{user_id}.json"
        if os.path.exists(token_file):
            os.remove(token_file)
        
        return {"status": "success", "message": "User removed successfully"}
    except Exception as e:
        logger.info(f"Error in remove_user: {str(e)}")
        return {"status": "failed", "message": "Error removing user"}