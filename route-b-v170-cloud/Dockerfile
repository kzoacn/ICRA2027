FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv ca-certificates libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    ROUTE_B_PYTHON=/opt/venv/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /opt/route-b
COPY requirements.txt requirements-simulator.txt ./
COPY scripts/install.sh scripts/install.sh
ARG TORCH_BACKEND=cu128
RUN TORCH_BACKEND=${TORCH_BACKEND} bash scripts/install.sh
COPY . .
RUN chmod +x run.sh setup.sh scripts/install.sh
ENV LIBERO_CONFIG_PATH=/tmp/route-b/libero \
    HF_HOME=/tmp/route-b/huggingface
ENTRYPOINT ["/opt/route-b/run.sh"]
CMD ["--help"]
