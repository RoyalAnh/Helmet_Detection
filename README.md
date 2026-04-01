# Helmet Violation Detection System

Real-time detection of motorcycle riders not wearing helmets.
Built with **YOLOv8** + **DeepSORT** tracking + **FastAPI**.


## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env: set MODEL_PATH to your trained weights

# 3. Run on webcam
python scripts/run.py

# 4. Run on video
python scripts/run.py --source traffic.mp4 --output output/result.mp4

# 5. Start API
uvicorn src.api.main:app --reload

# 6. Tests
pytest tests/ -v
```

---

## Docker

```bash
# Build & run (GPU)
docker compose up --build

# CPU only — edit docker-compose.yml: DEVICE=cpu, remove `deploy.resources`
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/detect/image` | Upload image → JSON + annotated image (base64) |
| POST | `/detect/image/raw` | Upload image → raw annotated JPEG |
| POST | `/video/frame` | Send one frame → lightweight JSON result |
| POST | `/video/file` | Upload full video → violation summary |
| GET | `/video/log` | Download violations_log.json |

Interactive docs: `http://localhost:8000/docs`

---

## Training

```bash
# Download a dataset (e.g. from Roboflow) into data/
python scripts/train.py --model yolov8m.pt --epochs 100 --batch 16
```

Best weights saved to `runs/train/helmet_exp/weights/best.pt`.

---

## Target Metrics

| Metric | Good | Target |
|--------|------|--------|
| mAP@50 | ≥ 0.85 | ≥ 0.90 |
| Precision | ≥ 0.85 | ≥ 0.90 |
| Recall | ≥ 0.80 | ≥ 0.87 |
| F1 | ≥ 0.85 | ≥ 0.88 |
| FPS (GPU) | ≥ 25 | ≥ 30 |

Run `src/evaluation.py` after training for a formatted report.
