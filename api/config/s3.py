import boto3
from botocore.exceptions import ClientError
from api.config.fetchMongo import MongoHandler
import os
from dotenv import load_dotenv
import logging
from typing import Optional, Any, Annotated

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)
BUCKET = "lazi-media-bucket"
widget_config = MongoHandler("userAuthDatabase", "openWidgets")

class S3Handler:
    def __init__(self, s3_client=s3, bucket_name=BUCKET):
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    async def fetch_file(self, object_name: str) -> Optional[bytes]:
        if not self.s3_client:
            raise IOError("S3 client not initialized")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_name)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Error fetching file: {e}")
        return None

    async def upload_file(self,
                    file_name: str,
                    object_name: Annotated[str, "Can be 'text', 'image', etc."]
                    ) -> bool:
        """Uploads file object to S3 bucket 

        Args:
            file_name (str): Name of the file being uploaded
            object_name (Annotated[Optional[str], &quot;Can be &#39;text&#39;, &#39;image&#39;, etc.&quot;], optional): Name of the object in S3. Defaults to None.

        Returns:
            bool: True if the file was uploaded successfully, False otherwise.
        """
        try:
            await widget_config.post_insert({"widget_type": {object_name: file_name}})
            self.s3_client.upload_file(
                file_name,
                self.bucket_name,
                object_name
            )
            logger.info(f"File uploaded successfully: {object_name}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
        return False

    def generate_presigned_url(self, 
                               object_name: str, 
                               expiration: int = 3600
                               ) -> Optional[str]:
        """Generates a presigned URL via boto3 client for frontend media access

        Args:
            object_name (str): The name of the object stored in the S3 bucket 
            expiration (int, optional): The time in seconds for the presigned URL to remain valid. Defaults to 3600.

        Returns:
            Optional[str]: The presigned URL as a string, or None if an error occurred.
        """
        try:
            response = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_name},
                ExpiresIn=expiration
            )
            logger.info(f"Generated presigned URL: {response}")
            return response
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
        return None