from typing import Callable
from src.fetchMongo import MongoHandler

user_handler = MongoHandler("userCredentials")
organization_handler = MongoHandler("organizations")
project_handler = MongoHandler("projects")

class ValidateOrganizations:
    @staticmethod
    def validate_organization_data(func: Callable):
        def wrapper(self, *args, **kwargs):
            organization_id = kwargs.get("organization_id")
            if not organization_id:
                try:
                    organization_id = args[0]
                except IndexError:
                    raise ValueError("Organization ID is required")
                
            organization = organization_handler.get_single_doc({"id": organization_id})
            if not organization:
                print(f"Organization not found for organization_id: {organization_id}")
                raise ValueError("Organization not found")

            print(f"Found Organization Data: {organization}")

            if not self.user_data:
                print(f"User data not found for user_id: {self.user_id}")
                raise ValueError("User data not found")


            result = func(self, *args, **kwargs)

            if result:
                print(f"Result from {func.__name__}: {result}")
                print(f"{func.__name__} called with args: {func.__annotations__}")
                return result
            else:
                print(f"Validation failed for organization_id: {organization_id}")
        return wrapper
    
    @staticmethod
    def validate_project_actions(func: Callable):
        def wrapper(self, *args, **kwargs):
            for arg in args:
                print(f"Arg for {func.__name__}: {arg}")
            for key, value in kwargs.items():
                print(f"Kwarg for {func.__name__}: {key} = {value}")

            return func(self, *args, **kwargs)
        return wrapper