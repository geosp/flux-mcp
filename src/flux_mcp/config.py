"""
Configuration for flux-mcp

Two sections, each read from its own environment variables:
- server: mcp-service-kit's BaseServerConfig (MCP_TRANSPORT, MCP_HOST, MCP_PORT, ...)
- flux: FluxConfig (FLUX_*)
"""

from pathlib import Path

from mcp_service_kit import BaseServerConfig
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class FluxConfig(BaseSettings):
    """
    ComfyUI backend, auth and request limits

    Environment variables: FLUX_COMFYUI_URL, FLUX_MCP_TOKEN, FLUX_PUBLIC_BASE_URL,
    FLUX_OUTPUT_DIR, FLUX_WORKFLOWS_DIR, FLUX_MAX_SIDE, ...
    """
    model_config = SettingsConfigDict(env_prefix="FLUX_", extra="ignore")

    comfyui_url: str = Field(
        default="http://localhost:8188",
        description="Base URL of the ComfyUI server"
    )
    mcp_token: SecretStr | None = Field(
        default=None,
        description="Bearer token clients must send over HTTP transport"
    )
    public_base_url: str = Field(
        default="http://localhost:8300",
        description="Base URL clients use to fetch full-resolution images"
    )
    output_dir: Path = Field(
        default=Path("data/images"),
        description="Where generated images are stored"
    )
    workflows_dir: Path = Field(
        default=Path("workflows"),
        description="Directory with API-format workflow JSON files and nodes.yaml"
    )
    min_side: int = Field(default=256, ge=64, description="Smallest width or height")
    max_side: int = Field(default=4096, ge=256, description="Largest width or height")
    max_count: int = Field(default=4, ge=1, le=16, description="Most images per call")
    max_prompt_chars: int = Field(default=4000, ge=100, description="Longest prompt accepted")
    preview_max_side: int = Field(
        default=1024,
        ge=128,
        description="Longest side of the JPEG preview returned inline"
    )
    max_upscale_side: int = Field(
        default=8192,
        ge=1024,
        description="Longest side upscale_image may produce"
    )
    max_input_bytes: int = Field(
        default=20 * 1024 * 1024,
        description="Largest input image accepted by edit_image"
    )
    timeout_s: float = Field(
        default=300.0,
        gt=0,
        description="How long one tool call waits for ComfyUI"
    )


class AppConfig(BaseModel):
    """
    Complete application configuration

    The server section goes to the server; the whole object goes to the service.
    """
    server: BaseServerConfig = Field(default_factory=BaseServerConfig)
    flux: FluxConfig = Field(default_factory=FluxConfig)
