FROM library/ubuntu:22.04

ENV DEVICE=cpu \
    CUDA_LAUNCH_BLOCKING=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN set -ex; \
    apt update; \
    apt install -y --no-install-recommends \
        ca-certificates \
        iproute2 \
        iputils-ping \
        python3 \
        python3-pip; \
    apt autoremove -y; \
    apt autoclean -y; \
    apt clean; \
    rm -rf \
        /var/lib/apt/lists/* \
        /var/tmp/* \
        /tmp/*

COPY . /app
WORKDIR /app

RUN pip install \
        --no-cache-dir \
        -r /app/requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app"]
CMD ["--reload", "--host", "0.0.0.0", "--port", "8000"]
