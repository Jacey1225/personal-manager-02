from typing import Optional
import uuid
from datetime import datetime
from api.schemas.projects import ResourceDetails

class HandleResources:
    def __init__(self, user_id: str):
        self.user_id = user_id

    def view_resource(self, resource: ResourceDetails):
        """View the details of a specific resource."""
        if resource.resource_owner == self.user_id:
            return resource
        else:
            raise PermissionError("You do not have permission to view this resource.")
