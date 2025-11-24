from fastapi.security import (OAuth2PasswordRequestForm, 
                              OAuth2PasswordBearer)
from fastapi import Depends, APIRouter, HTTPException, status
from api.schemas.auth import UserInDB
from api.config.fetchMongo import MongoHandler
from typing import Annotated, Optional
import logging
import os
from passlib.context import CryptContext 
from datetime import datetime, timedelta
import jwt
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
oauth_router = APIRouter()

class OAuthUser:
    def __init__(self, token: Annotated[str, Depends(OAuth2PasswordBearer)]):
        self.token = token
        self.auth_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.secret_key = os.getenv("JWT_SECRET")
        self.algorithm = "HS256"

    def hash_pass(self, password: str) -> str:
        """Hash password given by user

        Args:
            password (str): Password user inputs

        Returns:
            str: Hashed password
        """
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)

    def verify_hash(self, password: str, hashed: str) -> bool:
        """Verifies the password that the user inputs and the hased password in the database

        Args:
            password (str): the password that the user inputs
            hashed (str): the hashed password stored in the database

        Returns:
            bool: True if the passwords match, False otherwise
        """
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(password, hashed)

    async def get_user(self, user_id: str) -> UserInDB:
        """Fetches the user from the database via user_id

        Args:
            user_id (str): The ID of the user to fetch

        Returns:
            UserInDB: The user object if found, None otherwise
        """
        user = await user_config.get_single_doc({"user_id": user_id})
        return UserInDB(**user)

    async def get_current_user(self) -> UserInDB:
        """Fethces the user upon approval from the access token they were given 

        Raises:
            credentials_exception: An HTTPException for invalid credentials
            Exception: General exception for other errors
            credentials_exception: An HTTPException for missing user ID
            credentials_exception: An HTTPException for user not found

        Returns:
            UserInDB: The user object if found, None otherwise
        """
        credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                               detail="Could not validate credentials",
                                               headers={"WWW-Authenticate": "Bearer"})
        try:
            if self.secret_key:
                payload = jwt.decode(self.token, self.secret_key, algorithms=[self.algorithm])
                user_id = payload.get("sub")
                if user_id is None:
                    raise credentials_exception
                self.token = user_id
            else:
                raise Exception("No secret key found")
        except jwt.PyJWTError:
            raise credentials_exception
        user = await self.get_user(self.token)
        if not user:
            raise credentials_exception
        return user

    async def get_active_user(self) -> UserInDB:
        """Validates whether or not the current user is active as a developer or not

        Raises:
            HTTPException: Raises if the user is not a developer

        Returns:
            UserInDB: The user object if found, None otherwise
        """
        user = await self.get_current_user()
        if not user.developer:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="User is not authorized as a developer")
        return user
    
    def create_token(self, 
                           data: dict, 
                           expiration: Optional[timedelta]=None, 
                           scopes: Optional[list[str]]=None):
        """Creates an access token for a given user to complete full OAuth2 process with scopes for access
        restrictions

        Args:
            data (dict): The data to include in the token
            expiration (Optional[timedelta], optional): The expiration time for the token. Defaults to None.
            scopes (Optional[list[str]], optional): The scopes to include in the token. Defaults to None.

        Returns:
            str: The encoded JWT token
        """
        raw_token = data.copy()
        if expiration:
            expires = datetime.now() + expiration
        else:
            expires = datetime.now() + timedelta(hours=24)

        raw_token.update({
            "exp": expires,
            "scopes": scopes or [],
            "iat": datetime.now(),
            "type": "access"
        })
        if self.secret_key:
            self.token = jwt.encode(raw_token, 
                                     self.secret_key,
                                     algorithm=self.algorithm)
        return self.token

oauth_router.post("/oauth/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """The login endpoint that handles developers accessing their token to create new widgets

    Args:
        form_data (Annotated[OAuth2PasswordRequestForm, Depends): Data about the user to perform the requested
        authentication

    Raises:
        HTTPException: Invalid credentials for the user not existing
        HTTPException: Invalid credentials for the user not being a developer

    Returns:
        _type_: _description_
    """
    oauth_handler = OAuthUser(form_data.username)
    user: UserInDB = await oauth_handler.get_active_user()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials")

    token_expires = timedelta(hours=24)
    access_token = oauth_handler.create_token(data={"sub": user.user_id}, expiration=token_expires)

    if not oauth_handler.verify_hash(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials")

    return {"access_token": access_token, "token_type": "bearer"}