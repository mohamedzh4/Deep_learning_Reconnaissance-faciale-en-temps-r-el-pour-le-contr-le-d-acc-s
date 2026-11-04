# Face Access Control (local, MVP)

Real-time face recognition for access control: webcam → detect/align → ArcFace embedding → cosine match → multi-frame vote → door (simulated) + audit log.
Pretrained InsightFace models (`buffalo_s` fast / `buffalo_l` accurate), downloaded automatically into `./models` on first run.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.10 or 3.11 recommended. On Windows, `insightface` may need "Microsoft C++ Build Tools" to install.
GPU: replace `onnxruntime` with `onnxruntime-gpu` and set `model.use_gpu: true` in `config.yaml`.

## Usage
```bash
python scripts/enroll_user.py --name alice          # webcam enrollment (consent required)
python scripts/enroll_user.py --name bob
# put photos of non-enrolled people in data/own_dataset/_impostors/ (impostor probes)
python scripts/calibrate_threshold.py --far 0.01 --write
python scripts/run_realtime.py                       # q to quit
python scripts/benchmark.py                          # ms/frame + FPS for the report
pytest -q                                            # unit tests (no camera/model needed)
```
Already have images from Colab? Copy them to `data/own_dataset/<person>/` and run `python scripts/enroll_from_folders.py`.

## Layout
- `src/face_engine.py` detection + alignment + embedding · `gallery.py` templates · `matcher.py` cosine match
- `src/decision.py` voting + cooldown · `door.py` simulated door · `audit_log.py` SQLite events · `liveness.py` stub (Phase 2)
- `src/pipeline.py` end-to-end · `scripts/` entry points · `evaluation/results/` ROC plots · `data/`, `models/` git-ignored

## Notes
- Only embeddings are kept in the gallery; captured images (needed for calibration) can be disabled with `--no-images`.
- Low FPS on CPU: use `buffalo_s`, raise `runtime.process_every`, lower `camera` resolution.
- Liveness is a stub: a printed photo or a phone screen will be accepted until Phase 2.
