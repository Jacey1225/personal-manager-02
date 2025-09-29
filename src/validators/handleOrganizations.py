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
                    raise ValueError(f"{func.__name__}, {func.__class__}: Organization ID is required")

            organization = organization_handler.get_single_doc({"id": organization_id})
            if not organization:
                print(f"Organization not found for organization_id: {organization_id}")
                raise ValueError(f"{func.__name__}, {func.__class__}: Organization not found for organization_id: {organization_id}")

            print(f"Found Organization Data: {organization}")

            if not self.user_data:
                print(f"User data not found for user_id: {self.user_id}")
                raise ValueError(f"{func.__name__}, {func.__class__}: User data not found for user_id: {self.user_id}")


            result = func(self, *args, **kwargs)

            if result:
                print(f"Result from {func.__name__}: {result}")
                return result
            else:
                print(f"Validation failed for organization_id: {organization_id}")
        return wrapper