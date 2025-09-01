import datetime
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build 
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/calendar.readonly', 'https://www.googleapis.com/auth/tasks']

def enable_google_calendar_api():
    creds = None
    if not os.path.exists('data/token.json'):
        if not os.path.exists('data/credentials.json'):
            raise FileNotFoundError("credentials.json file not found in the 'data' directory.")
        flow = InstalledAppFlow.from_client_secrets_file('data/credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('data/token.json', 'w') as token:
            token.write(creds.to_json())
    else:
        creds = Credentials.from_authorized_user_file('data/token.json', SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open('data/token.json', 'w') as token:
                token.write(creds.to_json())
    
    with open('data/token.json', 'w') as token:
        token.write(creds.to_json())
    
    try:
        service = build('calendar', 'v3', credentials=creds)
    except HttpError as error:
        print(f'An error occurred: {error}')
        service = None
    return service


