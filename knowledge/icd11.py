"""
ICD-11 API Client — Integration with WHO International Classification of Diseases.
Used for retrieving diagnostic codes and medical context.
"""

from __future__ import annotations
import time
import requests
from typing import Optional, Any
from loguru import logger
from config.settings import ICD11Config, get_settings


class ICD11Client:
    """
    Client for the official WHO ICD-11 API.
    Updated for the 2026-01 Release.
    """

    def __init__(self, config: Optional[ICD11Config] = None, release_id: str = "2026-01"):
        self.config = config or get_settings().icd11
        self.release_id = release_id  # Latest release as of Feb 2026
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0
        self.available = bool(self.config.client_id and self.config.client_secret)
        
        if not self.available:
            logger.warning("ICD-11 API credentials missing. Integration disabled.")

    def _get_token(self) -> Optional[str]:
        """Obtain or refresh OAuth 2.0 access token."""
        if self.access_token and time.time() < self.token_expiry:
            return self.access_token

        logger.info("Refreshing ICD-11 API access token...")
        try:
            payload = {
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
                'scope': 'icdapi_access',
                'grant_type': 'client_credentials'
            }
            response = requests.post(
                self.config.token_url, 
                data=payload,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            # Set expiry with 60s buffer
            self.token_expiry = time.time() + data.get('expires_in', 3600) - 60
            return self.access_token
            
        except Exception as e:
            logger.error(f"Failed to obtain ICD-11 token: {e}")
            return None

    def _get_headers(self) -> dict:
        """Get standard headers for API requests."""
        token = self._get_token()
        return {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
            'Accept-Language': 'en',
            'API-Version': 'v2'
        }

    def search(self, query: str, linearization: str = "mms") -> list[dict[str, Any]]:
        """Search for ICD-11 entities in the 2026-01 release."""
        if not self.available:
            return []

        logger.debug(f"Searching ICD-11 ({self.release_id}) for: {query}")
        try:
            # Updated to use the 2026-01 release version
            url = f"{self.config.api_url}/icd/release/11/{self.release_id}/{linearization}/search"
            params = {'q': query}
            
            response = requests.get(
                url, 
                headers=self._get_headers(), 
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('destinationEntities', [])
            
        except Exception as e:
            logger.error(f"ICD-11 search failed for '{query}': {e}")
            return []

    def get_entity(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full details for a specific ICD-11 entity from the foundation."""
        if not self.available:
            return None

        try:
            url = f"{self.config.api_url}/icd/entity/{entity_id}"
            response = requests.get(
                url, 
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"ICD-11 lookup failed for entity {entity_id}: {e}")
            return None

    def get_mms_code(self, entity_id: str) -> Optional[str]:
        """Retrieve the clinical code for the 2026-01 release.

        Skips invalid entity IDs (e.g. 'unspecified', 'other') that may
        appear as suffixes in search result URIs.
        """
        if not self.available:
            return None

        # Skip non-numeric / invalid entity IDs
        if not entity_id or not entity_id.replace('.', '').isdigit():
            return None

        try:
            # Updated to use the 2026-01 release version
            url = f"{self.config.api_url}/icd/release/11/{self.release_id}/mms/{entity_id}"
            response = requests.get(
                url, 
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get('code')
        except Exception as e:
            logger.error(f"ICD-11 MMS code lookup failed for {entity_id}: {e}")
            return None