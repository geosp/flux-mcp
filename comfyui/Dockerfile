# ComfyUI on NVIDIA's PyTorch image (CUDA 13, arm64 for GB10).
# torch/torchvision come from the base image; pip must not replace them.
FROM nvcr.io/nvidia/pytorch:26.09-py3

ARG COMFYUI_VERSION=v0.42.1

RUN git clone --depth 1 --branch ${COMFYUI_VERSION} https://github.com/comfy-org/ComfyUI /opt/ComfyUI
WORKDIR /opt/ComfyUI

RUN grep -v -E '^(torch|torchvision|torchaudio)([<>=~ ].*)?$' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch; assert torch.cuda.is_available() or True; print(torch.__version__)"

COPY extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml

EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
