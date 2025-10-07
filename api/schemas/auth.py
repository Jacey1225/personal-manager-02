from pydantic import BaseModel Field
from typing import Optional, Any, Dict

class OAuthCompleteRequest(BaseModel):
    user_id: str
    authorization_code: str

class RemoveUserRequest(BaseModel):
    user_id: str