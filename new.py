# auth_setup.py
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os, pickle

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
creds = flow.run_local_server(port=0)

# Save credentials
with open('token.json', 'w') as token_file:
    token_file.write(creds.to_json())

print("✅ New token.json generated.")
