import keyring
from pyicloud import PyiCloudService    
from datetime import datetime
from typing import Optional

class ConfigureAppleAPI:
    def __init__(self,
                 user_id: str,
                 icloud_user: str,
                 ):
        self.user_id = user_id
        self.icloud_user = icloud_user
    
    def fetch_user_auth(self, auth_code: Optional[str]) -> PyiCloudService | None:
        try:
            icloud_pass = keyring.get_password(
                "user_auth",
                self.icloud_user
            )
            if not icloud_pass:
                print("iCloud password not found")
                return None
            else:
                api = PyiCloudService(self.icloud_user, icloud_pass)
                if api.requires_2fa:
                    result = api.validate_2fa_code(
                        auth_code
                    )
                    if not result:
                        print("Failed to verify 2FA code")
                        return None
                return api
        except Exception as e:
            print(f"Error fetching user auth: {str(e)}")
            return None
