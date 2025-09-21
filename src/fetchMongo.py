from pymongo import MongoClient
import os
from dotenv import load_dotenv
from typing import Optional, Any

load_dotenv()
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["userAuthDatabase"]
collection = db["userCredentials"]

class MongoHandler:
    @staticmethod
    def post_insert(insertion: dict) -> Any:
        """Insert a document into the MongoDB collection.

        Args:
            insertion (dict): The document to be inserted.
        """
        try:
            result = collection.insert_one(insertion)
            print(f"Document inserted with ID: {result.inserted_id}")
            return result
        except Exception as e:
            print(f"Error inserting document: {str(e)}")
    
    @staticmethod
    def get_single_doc(query: dict, column: Optional[str] = None) -> dict:
        """Fetch a single document from the MongoDB collection.

        Args:
            query (dict): The query to find the document.
            column (Optional[Any]): The specific column to retrieve. If None, retrieves the entire document.
        """
        try:
            if column:
                query_col = collection[column]
                document = query_col.find_one(query, {"_id": 0})
                return document if document else {}
            else:
                document = collection.find_one(query, {"_id": 0})
                return document if document else {}
        except Exception as e:
            print(f"Error fetching document: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_multi_doc(query: dict, column: Optional[str] = None) -> Optional[list[dict]] | dict:
        """Fetch multiple documents from the MongoDB collection.

        Args:
            query (dict): The query to find the documents.
            column (Optional[str]): The specific column to retrieve. If None, retrieves the entire document.
        """
        try:
            if column:
                query_col = collection[column]
                documents = query_col.find(query, {"_id": 0})
                return [doc for doc in documents] if documents else []
            else:
                return [doc for doc in collection.find(query, {"_id": 0})]
        except Exception as e:
            print(f"Error fetching documents: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def post_update(query: dict, update: dict) -> None:
        """Update a document in the MongoDB collection.

        Args:
            query (dict): The query to find the document to update.
            update (dict): The update operations to apply.
        """
        try:
            result = collection.update_one(query, {"$set": update})
            if result.modified_count > 0:
                print(f"Document updated successfully.")
            else:
                print(f"No documents matched the query.")
        except Exception as e:
            print(f"Error updating document: {str(e)}")
    
    @staticmethod
    def get_all():
        try:
            result = collection.find()
            return [doc for doc in result]
        except Exception as e:
            print(f"Error fetching documents: {str(e)}")
            return []
