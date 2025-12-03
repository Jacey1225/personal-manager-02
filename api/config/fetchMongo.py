import logging
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from typing import Optional, Any
import certifi

logger = logging.getLogger(__name__)

load_dotenv()


class MongoHandler:
    def __init__(self, 
                 database: str, 
                 collection: str):
            self.client: Optional[AsyncIOMotorClient] = None
            self.database = database
            self.collection_name = collection
            self.db = None
            self.collection = None

    async def get_client(self):
        """Request a MongoDB client connection.
        """
        if not self.client:
            logger.info(f"Setting up Mongo client...")
            try:
                self.client = AsyncIOMotorClient(os.getenv("MONGO_URI", 'mongodb://localhost:27017'), tlsCAFile=certifi.where())
                self.db = self.client[self.database]
                self.collection = self.db[self.collection_name]
                logger.info(f"Mongo client set up successfully.")
            except Exception as e:
                logger.error(f"Error setting up Mongo client: {e}")
                return False

        if self.client:
            logger.info(f"Testing Connection...")
            try:
                await self.client.admin.command('ping')
                logger.info(f"MongoDB connection successful.")
                return True
            except Exception as e:
                logger.error(f"Error testing MongoDB connection: {e}")
                return False
        return False
 
    async def close_client(self):
        """Close the MongoDB client and release resources.
        """
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            logger.info(f"Mongo client closed.")

    async def post_insert(self, insertion: dict) -> Any:
        """Insert a document into the MongoDB collection.

        Args:
            insertion (dict): The document to be inserted.
        """
        try:
            await self.get_client()
            if not insertion or self.collection is None:
                raise ValueError("Insertion document cannot be empty.")
            result = await self.collection.insert_one(insertion)
            logger.info(f"Document inserted with ID: {result.inserted_id}")
            await self.close_client()
            return result
        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return None

    async def get_single_doc(self, query: dict, column: Optional[str] = None) -> dict:
        """Fetch a single document from the MongoDB collection.

        Args:
            query (dict): The query to find the document.
            column (Optional[Any]): The specific column to retrieve. If None, retrieves the entire document.
        """
        try:
            await self.get_client()
            if self.collection is None:
                raise ValueError("Collection is not initialized.")
            if column:
                document = await self.collection.find_one(query, {"_id": 0, column: 1})
                await self.close_client()
                return document if document else {}
            else:
                document = await self.collection.find_one(query, {"_id": 0})
                await self.close_client()
                return document if document else {}
        except Exception as e:
            logger.error(f"Error fetching document: {str(e)}")
            return {"error": str(e)}

    async def get_multi_doc(self, query: dict, column: Optional[str] = None) -> Optional[list[dict]] | dict:
        """Fetch multiple documents from the MongoDB collection.

        Args:
            query (dict): The query to find the documents.
            column (Optional[str]): The specific column to retrieve. If None, retrieves the entire document.
        """
        try:
            await self.get_client()
            if self.collection is None:
                raise ValueError("Collection is not initialized.")
            if column:
                cursor = self.collection.find(query, {"_id": 0, column: 1})
            else:
                cursor = self.collection.find(query, {"_id": 0})
            documents = []
            async for doc in cursor:
                documents.append(doc)
            await self.close_client()
            return documents
        except Exception as e:
            await self.close_client()
            logger.error(f"Error fetching documents: {str(e)}")
            return {"error": str(e)}

    async def post_update(self, query: dict, update: dict):
        """Update a document in the MongoDB collection.

        Args:
            query (dict): The query to find the document to update.
            update (dict): The update operations to apply.
            
        Returns:
            UpdateResult: The result of the update operation, or None if error.
        """
        try:
            await self.get_client()
            if self.collection is None:
                raise ValueError("Collection is not initialized.")
            result = await self.collection.update_one(query, {"$set": update})
            await self.close_client()
            if result.modified_count > 0:
                logger.info(f"Document updated successfully.")
            else:
                logger.warning(f"No documents matched the query.")
            return result
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return None

    async def post_delete(self, query: dict) -> None:
        """Delete a document from the MongoDB collection.

        Args:
            query (dict): The query to find the document to delete.
        """
        try:
            await self.get_client()
            if self.collection is None:
                raise ValueError("Collection is not initialized.")
            result = await self.collection.delete_one(query)
            await self.close_client()
            if result.deleted_count > 0:
                logger.info(f"Document deleted successfully.")
            else:
                logger.warning(f"No documents matched the query.")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")

    async def get_all(self):
        try:
            await self.get_client()
            if self.collection is None:
                raise ValueError("Collection is not initialized.")
            cursor = self.collection.find()
            documents = []
            async for doc in cursor:
                documents.append(doc)
            await self.close_client()
            return documents
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            return []

    async def delete_all(self):
        """Deletes all documents in the collection.
        """
        try:
            await self.get_client()
            if self.collection is None:
                raise ValueError("Collection is not initialized.")
            result = await self.collection.delete_many({})
            await self.close_client()
            logger.info(f"Documents deleted: {result.deleted_count}")
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")