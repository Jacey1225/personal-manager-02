from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime

class ResourceDetails(BaseModel):
    resource_id: str = Field(default=str(uuid.uuid4()), description="Unique identifier for the resource")
    resource_name: str = Field(..., description="Name of the resource")
    resource_link: str = Field(..., description="Link to the resource (e.g., document, link, sheet)")
    resource_owner: str = Field(..., description="User ID of the resource owner")
    resource_timestamp: str = Field(default=datetime.now().isoformat(), description="Timestamp when the resource was created or modified")

class HandleResources:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def view_resource(self, resource: ResourceDetails):
        """View the details of a specific resource."""
        if resource.resource_owner == self.user_id:
            return resource
        else:
            raise PermissionError("You do not have permission to view this resource.")
