"""
flux-mcp: FLUX image generation over MCP

Run it with:
    flux-mcp --mode stdio
    flux-mcp --mode http --port 8300
"""

import logging

from fastmcp.server.auth import TokenVerifier
from mcp_service_kit import BaseMCPServer, BaseServerConfig, create_main_with_modes

from .auth import BearerTokenVerifier
from .config import AppConfig
from .service import FluxMCPService

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
This server generates and edits images with FLUX 3 Image Klein.

- generate_image: create images from a text description. A short request is fine: by default
  a prompt upsampler expands it into a full scene caption. Set enhance_prompt=false only when
  you pass a complete, detailed caption yourself.
- edit_image: change an existing image, or combine up to 10 reference images into a new one.
  Pass image ids returned by generate_image/edit_image (or http(s) URLs) in `images`, and refer
  to them by order in the prompt ("the person from the first image in the scene from the
  second image").
- To use the user's own image, upload it first and pass the returned id. With a shell:
  curl -H "Authorization: Bearer $FLUX_MCP_TOKEN" -F file=@photo.jpg <server>/upload
  (returns JSON with "id"). Without a shell, ask the user to open <server>/upload in a browser,
  upload the image there and give you the id. Images pasted into the chat cannot be passed on.
- Each result includes a small JPEG preview plus a URL to the full-resolution PNG. Give the
  user the URL; quote the seed if they may want to reproduce the image.
- Generation takes from several seconds to a minute; ask for count > 1 only when the user
  wants variations.
"""


class FluxMCPServer(BaseMCPServer):
    def __init__(self, config: BaseServerConfig, service: FluxMCPService):
        super().__init__(config, service)
        self.flux_config = service.config.flux

    @property
    def service_title(self) -> str:
        return "FLUX MCP Server"

    @property
    def service_description(self) -> str:
        return "Generates and edits images with FLUX 3 Image Klein through ComfyUI."

    @property
    def service_version(self) -> str:
        return "0.1.0"

    @property
    def instructions(self) -> str:
        return INSTRUCTIONS

    def create_auth_provider(self) -> TokenVerifier | None:
        token = self.flux_config.mcp_token
        return BearerTokenVerifier(token) if token is not None else None

    def validate_configuration(self) -> None:
        token = self.flux_config.mcp_token
        if (
            self.config.transport == "http"
            and self.config.auth_enabled
            and (token is None or not token.get_secret_value())
        ):
            raise ValueError("FLUX_MCP_TOKEN must be set for HTTP transport (or pass --no-auth)")


main = create_main_with_modes(
    FluxMCPServer,
    FluxMCPService,
    AppConfig,
    "flux-mcp",
    "flux-mcp - FLUX image generation and editing over MCP",
)


if __name__ == "__main__":
    main()
