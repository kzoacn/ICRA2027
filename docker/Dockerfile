FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv ca-certificates libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    ANCHOR_PYTHON=/opt/venv/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /opt/anchor
COPY configs/requirements.txt configs/requirements-simulator.txt configs/
COPY scripts/install.sh scripts/install.sh
ARG TORCH_BACKEND=cu128
RUN TORCH_BACKEND=${TORCH_BACKEND} bash scripts/install.sh
COPY configs/resources.lock.json configs/resources.lock.json
COPY src/ src/
COPY scripts/ scripts/
RUN chmod +x scripts/run.sh scripts/setup.sh scripts/install.sh
ENV LIBERO_CONFIG_PATH=/tmp/anchor/libero \
    HF_HOME=/tmp/anchor/huggingface
ENTRYPOINT ["/opt/anchor/scripts/run.sh"]
CMD ["--help"]
