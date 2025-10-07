import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build 
from googleapiclient.errors import HttpError

# Updated scopes - removed conflicting readonly scopes
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly", 
    "https://www.googleapis.com/auth/tasks.readonly", 
    "https://www.googleapis.com/auth/tasks", 
    "https://www.googleapis.com/auth/calendar.events"
]

class ConfigureGoogleAPI:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.flow = None
        self.creds = None
        self.auth_url = None
        if not os.path.exists('data/tokens'):
            os.makedirs('data/tokens')

    def write_token(self):
        if self.creds:
            with open(f'data/tokens/token_{self.user_id}.json', 'w') as token:
                token.write(self.creds.to_json())
            print(f"Token written to data/tokens/token_{self.user_id}.json")
        else:
            print("No credentials available to write.")

    def get_auth_url(self):
        """Generate and return the Google OAuth authorization URL"""
        if not os.path.exists('data/credentials.json'):
            raise FileNotFoundError("credentials.json file not found in the 'data' directory.")
        
        self.flow = InstalledAppFlow.from_client_secrets_file('data/credentials.json', SCOPES)
        self.flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        auth_url, _ = self.flow.authorization_url(prompt='consent', access_type='offline')
        self.auth_url = auth_url
        return self.auth_url

    def complete_auth_flow(self, authorization_code: str):
        """Complete OAuth flow with authorization code"""
        print(f"Starting complete_auth_flow for user {self.user_id}")
        print(f"Authorization code provided: {len(authorization_code)} characters")
        
        try:
            if not os.path.exists('data/credentials.json'):
                raise FileNotFoundError("credentials.json file not found in the 'data' directory.")
            
            print("Creating fresh OAuth flow for token exchange...")
            self.flow = InstalledAppFlow.from_client_secrets_file('data/credentials.json', SCOPES)
            self.flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

            print("Exchanging authorization code for token...")
            self.flow.fetch_token(code=authorization_code)
            self.creds = self.flow.credentials
            
            print(f"Credentials obtained successfully")
            print(f"Credentials valid: {self.creds.valid}")
            print(f"Has refresh token: {self.creds.refresh_token is not None}")
            
            print("Writing token to file...")
            token_written = self.write_token()
            
            if not token_written:
                print("ERROR: Failed to write token to file!")
                return None, None
            
            print("Testing credentials by building Google services...")
            event_service = build('calendar', 'v3', credentials=self.creds)
            task_service = build('tasks', 'v1', credentials=self.creds)
            
            print("Google services built successfully!")
            return event_service, task_service
            
        except Exception as e:
            print(f"ERROR in complete_auth_flow: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def enable_google_calendar_api(self):
        """Enable Google Calendar API - returns services if authenticated, auth URL if not"""
        token_path = f'data/tokens/token_{self.user_id}.json'
        
        if os.path.exists(token_path):
            try:
                self.creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                
                if self.creds and self.creds.valid:
                    event_service = build('calendar', 'v3', credentials=self.creds)
                    task_service = build('tasks', 'v1', credentials=self.creds)
                    return event_service, task_service
                
                elif self.creds and self.creds.expired and self.creds.refresh_token:
                    try:
                        self.creds.refresh(Request())
                        self.write_token()
                        event_service = build('calendar', 'v3', credentials=self.creds)
                        task_service = build('tasks', 'v1', credentials=self.creds)
                        return event_service, task_service
                    except Exception as e:
                        print(f"Token refresh failed: {e}")
                        os.remove(token_path)
                        return self.get_auth_url()
                else:
                    os.remove(token_path)
                    return self.get_auth_url()
                    
            except Exception as e:
                print(f"Error loading token: {e}")
                if os.path.exists(token_path):
                    os.remove(token_path)
                return self.get_auth_url()
        else:
            return self.get_auth_url()


