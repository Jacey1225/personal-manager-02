from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Any
import uuid

class OAuthCompleteRequest(BaseModel):
    user_id: str
    authorization_code: str

class RemoveUserRequest(BaseModel):
    user_id: str

class ICloudUserRequest(BaseModel):
    username: str
    service_name: str = Field(default="userAuth")
    apple_user: str = Field(..., description="Apple user ID")
    apple_pass: str = Field(..., description="Apple user password")

class User(BaseModel):
    user_id: str = Field(default=str(uuid.uuid4()), description="Unique identifier for the user")
    username: str = Field(default="", description="Username of the user")
    email: str = Field(default="", description="Email address of the user")
    icloud: str = Field(default="", description="iCloud account of the user")
    projects: dict[str, tuple] = Field(default={}, description="Projects associated with the user, eg. {'project_id': (project_name, role)}")
    projects_liked: int = Field(default=0, description="Number of projects liked by the user")
    organizations: list[str] = Field(default=[], description="Organizations associated with the user")
    service: dict[str, str] = Field(default_factory=dict, description="Services associated with the user")
    developer: bool = Field(default=False, description="Indicates if the user is a developer")
    google_auth: Dict[str, Any] = Field(default={}, description="Google authentication details")

    def set_google_auth(self, 
                        token: str,
                        refresh_token: str,
                        token_uri: str,
                        client_id: str,
                        client_secret: str,
                        scopes: list[str],
                        universe_domain: str,
                        account: str,
                        expiry: str):
        self.google_auth = {
            "token": token,
            "refresh_token": refresh_token,
            "token_uri": token_uri,
            "client_id": client_id,
            "client_secret": client_secret,
            "scopes": scopes,
            "universe_domain": universe_domain,
            "account": account,
            "expiry": expiry
        }

class UserInDB(User):
    hashed_password: str

class Scopes(str, Enum):
    WIDGETS_READ = "widgets:read"
    WIDGETS_WRITE = "widgets:write"
    WIDGETS_DELETE = "widgets:delete"

    FILES_WRITE = "files:write"
    FILES_DELETE = "files:delete"
    FILES_READ = "files:read"

    PROJECTS_READ = "projects:read"
    PROJECTS_WRITE = "projects:write"
    PROJECTS_DELETE = "projects:delete"

    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"
