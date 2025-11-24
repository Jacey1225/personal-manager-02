from api.config.fetchMongo import MongoHandler
from typing import Optional
import uuid
from datetime import datetime
from api.schemas.projects import PageSchema

class PageClient(PageSchema):
    def __init__(self, api_key):
        self.api_key = api_key

    def validate_api(self):
        pass

    def view_page(self, page_data: PageSchema) -> dict:
        return {}

    def create_page(self, page_data: PageSchema) -> dict:
        return {}