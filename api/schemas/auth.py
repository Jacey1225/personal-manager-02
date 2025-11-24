from pydantic import BaseModel, Field
from enum import Enum

class OAuthCompleteRequest(BaseModel):
    user_id: str
    authorization_code: str

class RemoveUserRequest(BaseModel):
    user_id: str

class ICloudUserRequest(BaseModel):
    service_name: str = Field(default="userAuth")
    apple_user: str = Field(..., description="Apple user ID")
    apple_pass: str = Field(..., description="Apple user password")

class User(BaseModel):
    user_id: str
    username: str
    email: str
    icloud: str
    password: str
    projects: dict[str, list[str]]
    projects_liked: int
    organizations: list[str]
    service: dict[str, str]
    developer: bool

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
