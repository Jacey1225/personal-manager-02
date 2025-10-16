from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class OAuthCompleteRequest(BaseModel):
    user_id: str
    authorization_code: str

class RemoveUserRequest(BaseModel):
    user_id: str

class ICloudUserRequest(BaseModel):
    service_name: str = Field(default="userAuth")
    apple_user: str = Field(..., description="Apple user ID")
    apple_pass: str = Field(..., description="Apple user password")