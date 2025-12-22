from api.config.fetchMongo import MongoHandler
from api.config.s3 import S3Handler
from api.routes.auth.public import OAuthUser
import logging
from fastapi import HTTPException
from api.schemas.widgets import WidgetConfig, WidgetSize, WidgetInteraction, WidgetComponents
from api.schemas.auth import Scopes
from typing import Optional, Annotated, Any, Callable
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
project_config = MongoHandler("userAuthDatabase", "openProjects")
widget_config = MongoHandler("userAuthDatabase", "openWidgets")

class WriteWidget:
    def __init__(self, 
                 username: str, 
                 token: str,
                 project_id: str):
        self.username = username
        self.token = token
        self.project_id = project_id
        self.current_widget = WidgetConfig()
        self.s3_client = S3Handler()
        self.oauth_client = OAuthUser(username, token=token)

    async def create(
            self,
            name: Optional[str]=None,
            size: Optional[str]=None,
    ):
        """Creating a widget with the required fields to start with. 

        Args:
            name (Optional[str], optional): Name of the widget to create. Defaults to None.
            size (Optional[str], optional): The size of the widget to create. Defaults to None.

        Raises:
            HTTPException: If there is an error during widget creation

        Returns:
            bool: True if the widget was created successfully, False otherwise
        """
        try:
            self.current_widget = WidgetConfig(
                name=name if name else "",
                size=WidgetSize(size) if size else WidgetSize.SMALL,
            )
            return True
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def interaction(self,
             endpoint: str, 
             headers: dict,
             refresh_interval: int,
             func: Callable):
        """Any code logic + context that the user develops for their widget interactions

        Args:
            endpoint (str): The endpoint name for the interaction
            headers (dict): The headers for the interaction
            refresh_interval (int): The refresh interval for the interaction
            func (Callable): The function containing the interaction logic

        Returns:
            WidgetInteraction: The created widget interaction
        """
        code_logic: str = inspect.getsource(func)
        interaction = WidgetInteraction(
            headers=headers,
            refresh_interval=refresh_interval,
            logic=code_logic
        )
        if endpoint not in self.current_widget.interactions:
            self.current_widget.interactions[endpoint] = interaction
        
        return self.current_widget.interactions[endpoint]
    
    async def component(
            self,
            endpoint: str,
            type: str,
            content: list[Any],
            props: dict[str, Any]
    ):
        """Any widget content that the user develops for their widgets

        Args:
            endpoint (str): The endpoint name for the component
            type (str): The type of the component
            content (list[Any]): The content of the component
            props (dict[str, Any]): The properties of the component

        Raises:
            HTTPException: If the interaction endpoint is not found

        Returns:
            WidgetComponents: The created widget component
        """
        component = WidgetComponents(
            type=type,
            content=content,
            properties=props
        )
        if endpoint not in self.current_widget.interactions:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        self.current_widget.interactions[endpoint].components.append(component)
        return self.current_widget.interactions[endpoint].components[-1]
    
    async def uploadable(
            self,
            object_name: str,
            expire: int,
            filename: Optional[str]
    ):
        """Uploads any media to the AWS S3 bucket 

        Args:
            object_name (str): Name of the media to be stored
            expire (int): Expiration of the media
            filename (Optional[str]): The local filename of the media to be uploaded

        Raises:
            HTTPException: If the S3 client is not initialized or upload fails

        Returns:
            str: The presigned URL for accessing the media
        """
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 client not initialized")
        
        media_object = self.s3_client.fetch_file(
            object_name
        )
        if not media_object and filename:
            await self.s3_client.upload_file(
                filename,
                object_name
            )
        presigned_url = self.s3_client.generate_presigned_url(
            object_name,
            expire  
        )
        return presigned_url

    async def save(self):
        """Saving the widget to the databases after authenticating the user and their scopes

        Raises:
            HTTPException: If the user does not have WIDGETS_WRITE scope or is not active
            Exception: If the project is not found
        """
        logger.info(f"Starting save process for widget by user: {self.username}")
        auth = await self.oauth_client.get_current_user()
        if not auth or Scopes.WIDGETS_WRITE not in auth["scopes"]:
            logger.info(f"User {self.username} does not have WIDGETS_WRITE scope or is not active")
            raise HTTPException(status_code=403, detail="Not enough permissions")
        logger.info(f"Auth successful for user: {self.username}")

        await widget_config.post_insert(self.current_widget.model_dump())
        logger.info(f"Widget {self.current_widget.widget_id} inserted into widget_config")
        project = await project_config.get_single_doc({"project_id": self.project_id})
        if not project:
            raise Exception("Project not found")
        logger.info(f"Project {self.project_id} found")

        if project["widgets"] and isinstance(project["widgets"], list):
            project["widgets"].append(self.current_widget.widget_id)
        else:
            project["widgets"] = [self.current_widget.widget_id]
        await project_config.post_update({"project_id": self.project_id}, project)

        logger.info(f"Widget {self.current_widget.widget_id} saved to project {self.project_id}")
        await widget_config.close_client()
        await project_config.close_client()
        
