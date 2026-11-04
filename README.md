# Face Access Control (local, MVP)

Real-time face recognition for access control: webcam → detect/align → ArcFace embedding → cosine match → multi-frame vote → door (simulated) + audit log.
Pretrained InsightFace models (`buffalo_s` fast / `buffalo_l` accurate), downloaded automatically into `./models` on first run.

---

## ⚙️ Prerequisites

| | macOS | Windows | Linux |
|---|---|---|---|
| **Python** | 3.10 or 3.11 | 3.10 or 3.11 | 3.10 or 3.11 |
| **C++ compiler** | Xcode CLT (`xcode-select --install`) | [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) ← **required for insightface** | `build-essential` (`sudo apt install build-essential`) |
| **GPU (optional)** | — (Metal not supported by ONNX) | CUDA 11.8+ + `onnxruntime-gpu` | CUDA 11.8+ + `onnxruntime-gpu` |

> **⚠️ Windows users** must install **Microsoft C++ Build Tools** *before* running `pip install -r requirements.txt`.
> During installation tick **"Desktop development with C++"**.
> Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## 🛠️ Setup

### 1 — Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (CMD)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

**GPU acceleration (Windows / Linux):** swap the ONNX package and enable GPU in config:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu        # CUDA 11.8 → onnxruntime-gpu==1.17.*
                                   # CUDA 12.x → onnxruntime-gpu==1.18.*+
```
Then set `model.use_gpu: true` in `config.yaml`.

---

## 🚀 Usage

All commands below work on **macOS, Linux, and Windows** (CMD or PowerShell).

### ▶ Launch the Web Dashboard (recommended)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

The dashboard includes:
| Page | Description |
|---|---|
| 🏠 Dashboard | KPIs, today's stats, recent events chart |
| 👥 Gallery | Enroll new users via webcam snapshots, remove users |
| 🎥 Live Control | Real-time face recognition with live camera feed |
| 📋 Audit Log | Filterable event table + CSV export |
| ⚙️ Settings | Edit all config.yaml parameters from the UI |

### ⌨️ CLI commands (alternative)

### ⌨️ CLI commands (alternative)

### Enroll users (webcam)
```bash
python scripts/enroll_user.py --name alice    # consent required
python scripts/enroll_user.py --name bob
```

### Enroll from existing images (e.g. from Colab)
Copy photos to `data/own_dataset/<person>/` then:
```bash
python scripts/enroll_from_folders.py
```

### Calibrate the similarity threshold
```bash
# put impostors' photos in  data/own_dataset/_impostors/
python scripts/calibrate_threshold.py --far 0.01 --write
```

### Run real-time access control
```bash
python scripts/run_realtime.py          # press q to quit
```

If your camera is not index 0:
```bash
python scripts/run_realtime.py --camera 1
```

### Benchmark (ms/frame + FPS)
```bash
python scripts/benchmark.py
python scripts/benchmark.py --image path/to/photo.jpg   # static image, no webcam needed
```

### Unit tests (no camera or model needed)
```bash
pytest -q
```

---

## 🪟 Windows-specific notes

| Issue | Fix |
|---|---|
| `insightface` fails to build | Install **Microsoft C++ Build Tools** (see Prerequisites) then retry |
| Camera not detected | Check another app (Teams, Zoom…) is not locking the camera. Try `--camera 1` |
| Camera permission denied | Windows Settings → Privacy → Camera → allow the terminal app |
| PowerShell activation blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| `cv2.imshow` window freezes | Known on some Windows GPU drivers; run with `--camera 0` and avoid moving the window during inference |
| `onnxruntime` GPU not found | Make sure CUDA is in `PATH`; reinstall `onnxruntime-gpu` matching your CUDA version |

---

## 🍎 macOS-specific notes

| Issue | Fix |
|---|---|
| Camera permission denied | System Settings → Privacy & Security → Camera → enable your terminal (Terminal / iTerm / VS Code / Cursor), then **quit and reopen** the app |
| `insightface` build error | Run `xcode-select --install` first |
| Metal GPU | Not supported by ONNX Runtime; keep `use_gpu: false` |

---

## 📁 Project layout

```
src/
  face_engine.py   # detection + 5-pt alignment + ArcFace embedding (InsightFace)
  gallery.py       # enrolled templates (.npz)
  matcher.py       # cosine similarity
  decision.py      # multi-frame voting + cooldown
  door.py          # simulated door relay
  audit_log.py     # SQLite event log
  liveness.py      # anti-spoofing stub (Phase 2)
  pipeline.py      # end-to-end frame processing

scripts/
  enroll_user.py          # live webcam enrollment
  enroll_from_folders.py  # batch enrollment from images
  run_realtime.py         # real-time access control loop
  calibrate_threshold.py  # ROC analysis + threshold selection
  benchmark.py            # latency / FPS measurement

data/           # git-ignored; gallery + images + SQLite log
models/         # git-ignored; InsightFace weights (auto-downloaded)
evaluation/results/  # ROC plots
```

---

## 📝 Notes

- Only **embeddings** are stored in the gallery (no raw images kept at runtime).
  Captured images (needed for threshold calibration) can be disabled with `--no-images`.
- **Low FPS on CPU?** Use `buffalo_s`, raise `runtime.process_every` in `config.yaml`, or lower the camera resolution.
- **Liveness** is a stub — a printed photo or phone screen is accepted until Phase 2 (anti-spoofing model).
