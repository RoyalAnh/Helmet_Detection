FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ──────────────────────────────────────────────────────────────
COPY . .

# ── Pre-download YOLOv8 weights (cached in image layer) ────────────────────
# Replace with your trained weights path via volume mount at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
