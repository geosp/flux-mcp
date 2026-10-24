"""
Static bearer-token check for the MCP endpoint
"""

import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier
from pydantic import SecretStr


def token_matches(expected: SecretStr, given: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.get_secret_value().encode())


class BearerTokenVerifier(TokenVerifier):
    """Accepts exactly one static bearer token"""

    def __init__(self, token: SecretStr):
        super().__init__()
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token_matches(self._token, token):
            return AccessToken(token=token, client_id="flux-mcp-client", scopes=[])
        return None
