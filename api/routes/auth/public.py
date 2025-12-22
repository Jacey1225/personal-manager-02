from fastapi.security import (OAuth2PasswordRequestForm, 
                              OAuth2PasswordBearer)
from fastapi import Depends, APIRouter, HTTPException, status
from api.schemas.auth import UserInDB
from api.config.fetchMongo import MongoHandler
from typing import Annotated, Optional, Union
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
    def __init__(
            self, 
            username: str, 
            password: Optional[str]=None,
            token: Optional[str] = None):
        self.username = username
        self.password = password
        self.token = token
        self.auth_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.secret_key = os.getenv("JWT_SECRET")
        self.algorithm = "HS256"

    def hash_pass(self) -> str:
        """Hash password given by user

        Args:
            password (str): Password user inputs

        Returns:
            str: Hashed password
        """
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        if self.password is None:
            raise ValueError("Password must be provided for hashing")
        logger.info(f"Hashing password for user: {self.password[:3]}...")
        
        if isinstance(self.password, str):
            password_bytes = self.password.encode('utf-8')
            if len(password_bytes) > 72:
                logger.warning(f"Password is {len(password_bytes)} bytes, truncating to 72 bytes")
                self.password = password_bytes[:72].decode('utf-8', errors='ignore')
        
        try:
            hashed = pwd_context.hash(self.password)
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise

        return hashed

    def verify_hash(self, hashed: str) -> bool:
        """Verifies the password that the user inputs and the hased 
        password in the database

        Args:
            password (str): the password that the user inputs
            hashed (str): the hashed password stored in the database

        Returns:
            bool: True if the passwords match, False otherwise
        """
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        if self.password is None:
            raise ValueError("Password must be provided for verification")
        logger.info(f"Verifying password for user: {self.password[:3]}")
        return pwd_context.verify(self.password, hashed)
    
    
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

    async def authenticate_user(self) -> Union[UserInDB, bool]:
        logger.info(f"Authenticating user: {self.username}")
        user = await self.get_user(self.username)
        if not user:
            logger.info(f"Authentication failed: user {self.username} not found")
            return False
        if user.hashed_password and not self.verify_hash(user.hashed_password):
            logger.info(f"Authentication failed: incorrect password for user {self.username}")
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
                logger.error("Secret key or token is missing")
                raise Exception("No secret key found")
        except jwt.PyJWTError:
            logger.error("Error decoding JWT token")
            raise credentials_exception
        return payload

    async def create_token(self, 
        name: str,
        data: dict, 
        expiration: Optional[timedelta]=None, 
        scopes: Optional[list[str]]=None
    ):
        """Creates an access token for a given user to complete full OAuth2 process with scopes for access
        restrictions and stores a hashed version in the database

        Args:
            name (str): The name of the token
            data (dict): The data to include in the token
            expiration (Optional[timedelta], optional): The expiration time for the token. Defaults to None.
            scopes (Optional[list[str]], optional): The scopes to include in the token. Defaults to None.

        Returns:
            str: The encoded JWT token
        """
        raw_token = data.copy()
        user_config = MongoHandler("userAuthDatabase", "userCredentials")
        await user_config.get_client()
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
            
            pwd_context = CryptContext(schemes=['bcrypt'], deprecated="auto")
            hashed_token = pwd_context.hash(self.token)
            logger.info(f"Hashed token: {hashed_token[:10]}...")
            user_info = await user_config.get_single_doc({"username": self.username})
            user_info["tokens"] = user_info.get("tokens", [])
            user_info["tokens"].append({
                "name": name,
                "token": hashed_token
            })
            await user_config.post_update({"username": self.username}, user_info)
            logger.info(f"Token stored for user: {self.username}")
            await user_config.close_client()

        logger.info(f"Token created for user: {data.get('sub')}")
        return self.token

@oauth_router.post("/oauth/token")
async def fetch_token(
    token_name: str,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """The token endpoint that handles developers accessing their token to create 
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
    oauth_handler = OAuthUser(form_data.username, password=form_data.password)
    user: Union[UserInDB, bool] = await oauth_handler.authenticate_user()
    if not user or isinstance(user, bool):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials")

    token_expires = timedelta(hours=24)
    
    scopes = form_data.scopes if form_data.scopes else []
    
    access_token = oauth_handler.create_token(
        name=token_name,
        data={"sub": user.user_id}, 
        expiration=token_expires,
        scopes=scopes)

    if user.hashed_password and not oauth_handler.verify_hash(user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials")

    return {"access_token": access_token, "token_type": "bearer"}

@oauth_router.get("/oauth/list_tokens")
async def list_tokens(
    username: str
):
    user_config = MongoHandler("userAuthDatabase", "userCredentials")
    await user_config.get_client()

    user_info = await user_config.get_single_doc({"username": username})
    tokens_info = user_info.get("tokens", [])
    await user_config.close_client()
    return {"status": "success", "tokens": tokens_info}


@oauth_router.delete("/oauth/revoke")
async def revoke_token(
    username: str,
    token: Annotated[str, Depends(OAuth2PasswordBearer(tokenUrl="token"))]
):
    """Endpoint to revoke an access token

    Args:
        token (Annotated[str, Depends(OAuth2PasswordBearer(tokenUrl="token"))]): The token to be revoked

    Raises:
        HTTPException: Invalid credentials for the user not existing

    Returns:
        dict: Confirmation of token revocation
    """
    user_config = MongoHandler("userAuthDatabase", "userCredentials")
    await user_config.get_client()

    user_info = await user_config.get_single_doc({"username": username})
    if user_info:
        tokens = user_info.get("tokens", [])
        pwd_context = CryptContext(schemes=['bcrypt'], deprecated="auto")
        for token_info in tokens:
            t = token_info["token"]
            name = token_info["name"]
            if pwd_context.verify(token, t):
                user_info["tokens"].remove(token_info)
                logger.info(f"Token {name} revoked for user: {username}")
                break
        await user_config.post_update({"username": username}, user_info)
        await user_config.close_client()
        return {"status": "success", "message": "Token revoked"}
    else:
        await user_config.close_client()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication credentials")
