"""Logging in to the OpenSky REST API with OAuth2 (client credentials).

OpenSky stopped accepting username/password on the API in March 2026. Now I
have an API client on my account page, which gives me a client id and a secret.
I send those two to the token URL and get back an access token that lasts about
30 minutes. Every API request then carries the token, never the secret.
"""
import json
import time
from pathlib import Path

import requests

from .config import CREDENTIALS_FILE, TOKEN_URL


class TokenManager:
    """Keeps one token and gets a new one when it is about to expire."""

    REFRESH_MARGIN_S = 30  # get a new token 30 s early so a request never uses an expired one

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None
        self._expires_at = 0.0

    @classmethod
    def from_json_file(cls, path: Path = CREDENTIALS_FILE) -> "TokenManager":
        """Read my client id and secret from credentials.json (it is gitignored)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No credentials at {path}. Create an API client on your OpenSky "
                "account page and save the downloaded file there.")
        data = json.loads(path.read_text())
        # the file from OpenSky uses clientId/clientSecret, I also accept snake_case
        client_id = data.get("clientId") or data.get("client_id")
        client_secret = data.get("clientSecret") or data.get("client_secret")
        if not client_id or not client_secret:
            raise ValueError(f"{path} must contain clientId and clientSecret")
        return cls(client_id, client_secret)

    def token(self) -> str:
        """Return a token that is still valid, asking OpenSky for a new one if needed."""
        if self._token is None or time.time() >= self._expires_at - self.REFRESH_MARGIN_S:
            # client credentials grant: trade my id + secret for an access token
            resp = requests.post(TOKEN_URL, data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            # expires_in is in seconds, so I save the exact time the token runs out
            self._expires_at = time.time() + payload.get("expires_in", 1800)
        return self._token

    def headers(self) -> dict:
        # the API expects the header "Authorization: Bearer <token>"
        return {"Authorization": f"Bearer {self.token()}"}
