"""
Static bearer-token checks, shared by the MCP endpoint and the upload route
"""

import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier
from pydantic import SecretStr


def token_matches(expected: SecretStr, given: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.get_secret_value().encode())


def bearer_token(authorization: str | None) -> str | None:
    """The token from an ``Authorization: Bearer <token>`` header value"""
    scheme, _, token = (authorization or "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class BearerTokenVerifier(TokenVerifier):
    """Accepts exactly one static bearer token"""

    def __init__(self, token: SecretStr):
        super().__init__()
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token_matches(self._token, token):
            return AccessToken(token=token, client_id="flux-mcp-client", scopes=[])
        return None
