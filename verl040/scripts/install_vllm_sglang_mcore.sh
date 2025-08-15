#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-0}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
pip --timeout=1000 --resume-retries=5 install "torch==2.6.0"
pip --timeout=1000 --resume-retries=5 install "torchvision==0.21.0"
pip --timeout=1000 --resume-retries=5 install "torchaudio==2.6.0"
pip --timeout=1000 --resume-retries=5 install "tensordict==0.6.2"
pip --timeout=1000 --resume-retries=5 install torchdata
pip --timeout=1000 --resume-retries=5 install "vllm==0.8.5.post1"
if [ $USE_SGLANG -eq 1 ]; then
    pip --timeout=1000 --resume-retries=5 install "sglang[all]==0.4.6.post5"
    pip --timeout=1000 --resume-retries=5 install torch-memory-saver
fi

echo "2. install basic packages"
pip --timeout=1000 --resume-retries=5 install "transformers[hf_xet]>=4.51.0"
pip --timeout=1000 --resume-retries=5 install accelerate
pip --timeout=1000 --resume-retries=5 install datasets
pip --timeout=1000 --resume-retries=5 install peft
pip --timeout=1000 --resume-retries=5 install hf-transfer
pip --timeout=1000 --resume-retries=5 install "numpy<2.0.0"
pip --timeout=1000 --resume-retries=5 install "pyarrow>=15.0.0"
pip --timeout=1000 --resume-retries=5 install pandas
pip --timeout=1000 --resume-retries=5 install ray[default]
pip --timeout=1000 --resume-retries=5 install codetiming
pip --timeout=1000 --resume-retries=5 install hydra-core
pip --timeout=1000 --resume-retries=5 install pylatexenc
pip --timeout=1000 --resume-retries=5 install qwen-vl-utils
pip --timeout=1000 --resume-retries=5 install wandb
pip --timeout=1000 --resume-retries=5 install dill
pip --timeout=1000 --resume-retries=5 install pybind11
pip --timeout=1000 --resume-retries=5 install liger-kernel
pip --timeout=1000 --resume-retries=5 install mathruler
pip --timeout=1000 --resume-retries=5 install pytest
pip --timeout=1000 --resume-retries=5 install py-spy
pip --timeout=1000 --resume-retries=5 install pyext
pip --timeout=1000 --resume-retries=5 install pre-commit
pip --timeout=1000 --resume-retries=5 install ruff

pip --timeout=1000 --resume-retries=5 install "nvidia-ml-py>=12.560.30"
pip --timeout=1000 --resume-retries=5 install "fastapi[standard]>=0.115.0"
pip --timeout=1000 --resume-retries=5 install "optree>=0.13.0"
pip --timeout=1000 --resume-retries=5 install "pydantic>=2.9"
pip --timeout=1000 --resume-retries=5 install "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip --timeout=1000 --resume-retries=5 install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip --timeout=1000 --resume-retries=5 install flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
fi


echo "5. May need to fix opencv"
pip --timeout=1000 --resume-retries=5 install opencv-python
pip --timeout=1000 --resume-retries=5 install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip --timeout=1000 --resume-retries=5 install nvidia-cudnn-cu12==9.8.0.87
fi

echo "Successfully installed all packages"
