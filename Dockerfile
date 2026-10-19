# 実行環境のコンテナ（CPU のみ、画面なし描画）。動作確認済みの venv 版と同じ構成
#   docker compose build
#   docker compose run --rm sim bash reproduce.sh quick
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=osmesa \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH=/opt/venv/bin:$PATH

# OSMesa（画面なし描画）、日本語フォント（動画の文字）、Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip libosmesa6 libgl1 libglib2.0-0 fonts-noto-cjk ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python パッケージ（torch は CPU 版。CUDA 版の数 GB のダウンロードを避ける）
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.14.1 --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# コードと学習済みモデル（out/・logs/ などは .dockerignore で除外）
COPY sim /app/sim
WORKDIR /app/sim
RUN mkdir -p out logs && chmod -R a+rwX /app/sim

CMD ["python", "check_env.py"]
