#!/bin/bash
# referred from: https://github.com/volcengine/verl/blob/v0.4.0/docker/Dockerfile.vllm.sglang.megatron

# Install torch-2.6.0+cu124 + vllm-0.8.5.post1 + sglang-0.4.6.post5
pip install "sglang[all]==0.4.6.post5" --find-links ./packages && pip install torch-memory-saver
pip install "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata

# Install basic packages
pip install "opentelemetry-sdk==1.26.0" "opentelemetry-api==1.26.0" "opentelemetry-exporter-prometheus==0.47b0"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile \
    pytest py-spy pyext pre-commit ruff

# Install flash-attn-2.7.4.post1 (cxx11abi=False)
# wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install ./packages/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Fix packages
pip uninstall -y pynvml nvidia-ml-py && \
    pip install --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# Install cudnn (for megatron)
# pip install nvidia-cudnn-cu12==9.8.0.87

pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"
