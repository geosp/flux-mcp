# flux-mcp: MCP server only (no GPU). Talks to ComfyUI over HTTP.
FROM python:3.12-slim

# git: uv fetches mcp-service-kit from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY workflows ./workflows
RUN uv sync --frozen --no-dev

RUN useradd --uid 1000 --create-home flux && mkdir -p /data/images && chown flux /data/images
USER flux

ENV PATH=/app/.venv/bin:$PATH \
    MCP_PORT=8300 \
    FLUX_WORKFLOWS_DIR=/app/workflows \
    FLUX_OUTPUT_DIR=/data/images

EXPOSE 8300
# --mode is required: the CLI default (stdio) overrides MCP_TRANSPORT
CMD ["flux-mcp", "--mode", "http"]
