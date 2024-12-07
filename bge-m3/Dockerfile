FROM ghcr.io/azicen/embedding-api:0.0.1

ENV DEVICE=cpu \
    CUDA_LAUNCH_BLOCKING=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

COPY ./bge-m3 /app/bge-m3
WORKDIR /app

RUN huggingface-cli download --resume-download BAAI/bge-m3 --local-dir model
