from fastapi import HTTPException, APIRouter
import logging
import os
from api.config.plugins.enable_google_api import ConfigureGoogleAPI
from api.schemas.auth import OAuthCompleteRequest 

google_router = APIRouter()
logger = logging.getLogger(__name__)

@google_router.get("/auth/google")
def google_auth(user_id: str) -> dict:
    """Get Google OAuth authorization URL for the user"""
    try:
        logger.info(f"Google auth request for user_id: {user_id}")
        
        if not os.path.exists('data/credentials.json'):
            raise HTTPException(
                status_code=500, 
                detail="Google credentials file not found. Please ensure credentials.json is in the data directory."
            )
        
        google_api = ConfigureGoogleAPI(user_id)
        result = google_api.enable()
        
        logger.info(f"Result type: {type(result)}, Result: {result}")
        
        if isinstance(result, str):  
            return {"status": "auth_required", "auth_url": result}
        elif result is not None and len(result) == 2:  
            return {"status": "already_authenticated", "message": "User already has valid Google credentials"}
        else:
            raise HTTPException(status_code=500, detail="Unexpected result from Google API setup")
            
    except FileNotFoundError as e:
        logger.info(f"File not found error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.info(f"Error in google_auth: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing Google auth: {str(e)}")

@google_router.post("/auth/google/complete")
def complete_google_auth(request: OAuthCompleteRequest):
    """Complete Google OAuth flow with authorization code"""
    try:
        logger.info(f"Completing Google auth for user_id: {request.user_id}")
        logger.info(f"Authorization code length: {len(request.authorization_code)}")
        
        google_api = ConfigureGoogleAPI(request.user_id)
        result = google_api.complete_auth_flow(request.authorization_code)
        
        if result is not None and len(result) == 2:
            return {"status": "success", "message": "Google authentication completed successfully"}
        else:
            return {"status": "failed", "message": "Failed to complete Google authentication"}
            
    except Exception as e:
        logger.info(f"Error completing Google auth: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error completing Google auth: {str(e)}")

