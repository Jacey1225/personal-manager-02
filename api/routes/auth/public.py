from fastapi.security import (OAuth2PasswordRequestForm, 
                              OAuth2PasswordBearer)
from fastapi import Depends, APIRouter, HTTPException, status
from api.schemas.auth import User, UserInDB
from api.config.fetchMongo import MongoHandler
from typing import Annotated, Optional, Any, Union
import logging
import os
from passlib.context import CryptContext 
from datetime import datetime, timedelta
import jwt
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
user_config = MongoHandler("userAuthDatabase", "userCredentials")
oauth_router = APIRouter()

class OAuthUser:
    def __init__(self, username: str, token: Optional[str] = None):
        self.username = username
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
        logger.info(f"Hashing password for user: {password[:3]}")
        
        # Ensure password is within bcrypt's 72 byte limit
        if isinstance(password, str):
            password_bytes = password.encode('utf-8')
            if len(password_bytes) > 72:
                logger.warning(f"Password is {len(password_bytes)} bytes, truncating to 72 bytes")
                password = password_bytes[:72].decode('utf-8', errors='ignore')
        
        return pwd_context.hash(password)

    def verify_hash(self, password: str, hashed: str) -> bool:
        """Verifies the password that the user inputs and the hased 
        password in the database

        Args:
            password (str): the password that the user inputs
            hashed (str): the hashed password stored in the database

        Returns:
            bool: True if the passwords match, False otherwise
        """
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        logger.info(f"Verifying password for user: {password[:3]}")
        return pwd_context.verify(password, hashed)
    
    
    async def get_user(self, username: str) -> UserInDB:
        """Fetches the user from the database via username

        Args:
            username (str): The username of the user to fetch

        Returns:
            UserInDB: The user object if found, None otherwise
        """
        user = await user_config.get_single_doc({"username": username})
        await user_config.close_client()
        return UserInDB(**user)

    async def authenticate_user(self, 
        username: str, 
        password: str
    ) -> Union[UserInDB, bool]:
        logger.info(f"Authenticating user: {username}")
        user = await self.get_user(username)
        if not user:
            logger.info(f"Authentication failed: user {username} not found")
            return False
        if not self.verify_hash(password, user.hashed_password):
            logger.info(f"Authentication failed: incorrect password for user {username}")
            return False
        return user

    async def get_current_user(self) -> dict:
        """Fethces the user upon approval from the access token they were given 

        Raises:
            credentials_exception: An HTTPException for invalid credentials
            Exception: General exception for other errors
            credentials_exception: An HTTPException for missing user ID
            credentials_exception: An HTTPException for user not found

        Returns:
            UserInDB: The user object if found, None otherwise
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
        try:
            if self.secret_key and self.token:
                payload = jwt.decode(
                    self.token, 
                    self.secret_key, 
                    algorithms=[self.algorithm])
                logger.info(f"Payload decoded successfully for token: {payload}")
                user_id = payload.get("sub")
                expiration = datetime.fromtimestamp(payload.get("exp"))
                logger.info(f"Token expiration time: {expiration}")
                if user_id is None or expiration < datetime.now():
                    logger.info(f"Token expired or invalid for user: {user_id}")
                    raise credentials_exception
            else:
                raise Exception("No secret key found")
        except jwt.PyJWTError:
            raise credentials_exception
        return payload

    def create_token(self, 
        data: dict, 
        expiration: Optional[timedelta]=None, 
        scopes: Optional[list[str]]=None
    ):
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
        logger.info(f"Token created for user: {data.get('sub')}")
        return self.token

@oauth_router.post("/oauth/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """The login endpoint that handles developers accessing their token to create 
    new widgets

    Args:
        form_data (Annotated[OAuth2PasswordRequestForm, Depends): Data about 
        the user to perform the requested authentication

    Raises:
        HTTPException: Invalid credentials for the user not existing
        HTTPException: Invalid credentials for the user not being a developer

    Returns:
        dict: The access token and token type
    """
    oauth_handler = OAuthUser(form_data.username)
    user: Union[UserInDB, bool] = await oauth_handler.authenticate_user(
        form_data.username, 
        form_data.password)
    if not user or isinstance(user, bool):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials")

    token_expires = timedelta(hours=24)
    
    # Get scopes from form_data (already a list)
    scopes = form_data.scopes if form_data.scopes else []
    
    access_token = oauth_handler.create_token(
        data={"sub": user.user_id}, 
        expiration=token_expires,
        scopes=scopes)

    if not oauth_handler.verify_hash(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials")

    return {"access_token": access_token, "token_type": "bearer"}

@oauth_router.get("/oauth/token/user/{username}/{token}")
async def read_user(
    username: str,
    token: str,
) -> dict:
    """Reads the user data through OAuth2 process

    Args:
        username (str): username of the user
        token (str): OAuth2 token

    Raises:
        HTTPException: If the user is not found or the token is invalid

    Returns:
        dict: The user data and payload
    """
    oauth = OAuthUser(username, token)
    user = oauth.authenticate_user(username, token)
    payload = await oauth.get_current_user()
    if payload and user:
        return {"user": user, "payload": payload}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found")