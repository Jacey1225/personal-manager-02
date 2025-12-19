from fastapi import HTTPException, APIRouter
from api.routes.auth.public import OAuthUser
from api.schemas.auth import RemoveUserRequest, ICloudUserRequest, User, UserInDB
from api.config.fetchMongo import MongoHandler
import logging
from typing import Optional
import keyring
from passlib.context import CryptContext
logger = logging.getLogger(__name__)

auth_router = APIRouter()
service_name = "user_auth"


@auth_router.get("/auth/signup")
async def signup(
    username: str, 
    email: str, 
    password: str, 
    project: Optional[dict]=None, 
    organization: Optional[str]=None) -> dict:
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
    user_config = MongoHandler("userAuthDatabase", "userCredentials")
    query_email = {"email": email}
    query_username = {"username": username}

    if await user_config.get_single_doc(query_email) or \
        await user_config.get_single_doc(query_username):
        return {"status": "failed", "message": "Email or username already exists"}

    user = UserInDB(
        username=username,
        email=email,
        projects=project if project else {},
        organizations=[organization] if organization else []
    )

    oauth_handler = OAuthUser(username, password=password)
    user.hashed_password = oauth_handler.hash_pass()
    logger.info(f"Signing up user: {username} with email: {email}")
    result = await user_config.post_insert(user.model_dump())
    logger.info(f"User created with ID: {result.inserted_id}")

    return {"status": "success", "user_id": user.user_id}

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
    user_config = MongoHandler("userAuthDatabase", "userCredentials")
    project_config = MongoHandler("userProjectsDatabase", "projects")
    
    found_password = keyring.get_password(service_name, username)    
    if not found_password:
        return {"status": "failed", "message": "Invalid username or password"}
    if found_password == password:
        user_info = await user_config.get_single_doc({"username": username})
        user = User(**user_info)
        if project_id and project_id not in user.projects:
            project = await project_config.get_single_doc({"project_id": project_id})
            if project:
                user.projects[project_id] = (project.get("project_name", ""), "member")
        if org_id and org_id not in user.organizations:
            user.organizations.append(org_id)

        await user_config.post_update({"username": username}, user.model_dump())
        return {"status": "success", "user_id": user_info.get("user_id")}
    else:
        return {"status": "failed", "message": "Invalid username or password"}

@auth_router.post("/auth/set_icloud_user")
async def set_icloud_user(request: ICloudUserRequest):
    found_pass = True if keyring.get_password(request.service_name, request.apple_user) else False
    if not found_pass:
        user_config = MongoHandler("userAuthDatabase", "userCredentials")

        user_info = await user_config.get_single_doc({"username": request.username})
        user = User(**user_info)
        user.icloud = request.apple_user

        await user_config.post_update({"username": request.username}, user.model_dump())
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
        user_config = MongoHandler("userAuthDatabase", "userCredentials")        
        user_id = request.user_id
        await user_config.post_delete({"user_id": user_id})
        await user_config.close_client()

        return {"status": "success", "message": "User removed successfully"}
    except Exception as e:
        logger.info(f"Error in remove_user: {str(e)}")
        return {"status": "failed", "message": "Error removing user"}