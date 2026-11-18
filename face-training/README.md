# Face recognition from scratch (MobileFaceNet + ArcFace on CASIA-WebFace)

Trains your own face-embedding network. The exported ONNX file plugs into the access-control project
(`model.recognizer` in its `config.yaml`). Works on Mac (Apple GPU via MPS, or CPU), Windows/Linux (CUDA or CPU) and Colab.

## Setup (separate virtual environment from the access-control project)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order
| Step | Command | What happens |
|---|---|---|
| 1 | `python step1_download.py` | downloads aligned CASIA-WebFace + LFW/CALFW/CPLFW/AgeDB-30 from Kaggle (needs a Kaggle API token), records their location in `data/paths.json` |
| 2 | `python step2_clean_and_index.py` | decodes every image; removes corrupt / blank / wrong-size files, duplicates, images shared by two identities, tiny identities; writes `data/index/` (manifest, stats, histogram) and measures pixel statistics |
| 2b | `python show_samples.py` | prints image -> array -> tensor -> vector shapes and saves `data/index/samples.png` |
| 3a | `python step3_train.py --selftest` | 1-minute check: output shape, loss goes down, ONNX export matches PyTorch |
| 3b | `python step3_train.py --bench` | measures images/s and estimates the time per epoch on your machine |
| 3c | `python step3_train.py --epochs 20 --out runs/exp1` | trains; evaluates LFW after each epoch; `last.pt`, `best.pt`, `log.csv`, `curves.png` |
| 4 | `python step4_evaluate.py --ckpt runs/exp1/best.pt` | LFW / CALFW / CPLFW / AgeDB-30 accuracy table |
| 5 | `python step5_export_onnx.py --ckpt runs/exp1/best.pt` | writes `exports/my_model.onnx`, checks parity and LFW accuracy of the ONNX file |

Interrupted? `python step3_train.py --resume runs/exp1/last.pt --epochs 20`.
Tests: `pytest -q tests` (the data tests need no torch; the model tests are skipped without it).

## Small laptop run first
`python step2_clean_and_index.py --max-ids 1000 --max-images 30` builds a small subset to check the whole pipeline in minutes.
Use `--bench` before any long run. If your machine is too slow for a full run, the same scripts run unchanged on a Colab GPU
(`pip install -r requirements.txt`, then steps 1-5).

## How the pieces map to the training pipeline
- **Normalization:** `(x - 127.5) / 127.5`, RGB, channels first (`fr/data.py`). Same formula as `src/embedder.py` in the access-control project.
- **Vectors:** image (112x112x3 uint8) -> tensor (3x112x112 in [-1, 1]) -> network -> 512-d embedding.
- **Layers and initialization:** `fr/model.py` (He init for convolutions, BN gamma=1/beta=0, Xavier for Linear, N(0, 0.01) for the ArcFace weights).
- **Loss:** ArcFace (scale 64, margin 0.5, margin ramped 0 -> 0.5 over 3 epochs). The head is training-only and is not exported.
- **Optimizer:** SGD momentum 0.9, weight decay 5e-4, 1 epoch warm-up then cosine decay, gradient clipping at 5.

## Notes
- Kaggle mirrors are unofficial copies. Their folder layout is auto-detected (one folder per identity; `*_ann.txt` pair files). If detection fails,
  download manually and use `--train-root` / `--val-root`.
- Cite the dataset papers in your report (CASIA-WebFace: Yi et al. 2014; ArcFace: Deng et al. 2019).
- Keep the same alignment at training and inference: the access-control project aligns with `src/align.py`; check with step 5 + its calibration.
