"""STEP 3 - build the network and train it with ArcFace loss.

  python step3_train.py --selftest                    # 1-minute sanity check (shapes, loss goes down, ONNX export)
  python step3_train.py --bench                       # measure speed and estimate the time of one epoch
  python step3_train.py --epochs 20 --out runs/exp1   # real training (checkpoints in runs/exp1)
  python step3_train.py --resume runs/exp1/last.pt --epochs 20      # continue after an interruption

Pipeline per step: batch of aligned images -> tensors in [-1, 1] -> MobileFaceNet -> 512-d vectors
-> ArcFace head (identity logits with an angular margin) -> cross-entropy -> SGD update.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fr.data import ManifestDataset
from fr.evaluate import evaluate_ann, torch_feature_fn
from fr.model import ArcFace, build_backbone, count_params
from fr.utils import INDEX_DIR, AverageMeter, get_device, load_paths, seed_everything


def lr_at(step, total, warmup, base):
    """Linear warm-up for `warmup` steps, then cosine decay to 0."""
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def selftest(device):
    from fr.export import export_onnx, parity

    net = build_backbone(512).to(device).train()
    out = net(torch.randn(4, 3, 112, 112, device=device))
    print(f"1. forward: output {tuple(out.shape)} (expected (4, 512)), parameters: {count_params(net) / 1e6:.2f} M")
    ok = tuple(out.shape) == (4, 512)

    C, per = 8, 4  # can the network memorize 32 random images of 8 "identities"? the loss must go down
    imgs = torch.randn(C * per, 3, 112, 112, device=device)
    labels = torch.arange(C, device=device).repeat_interleave(per)
    head = ArcFace(512, C, 64.0, 0.2).to(device)
    opt = torch.optim.SGD(list(net.parameters()) + list(head.parameters()), lr=0.05, momentum=0.9, weight_decay=5e-4)
    first = last = None
    for _ in range(60):
        loss = F.cross_entropy(head(net(imgs), labels), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    good = last < 0.9 * first
    print(f"2. overfit test: loss {first:.2f} -> {last:.2f}  {'OK' if good else 'FAIL: loss did not go down'}")
    ok = ok and good

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        print("3. ONNX parity: skipped (pip install onnx onnxruntime)")
    else:
        Path("exports").mkdir(exist_ok=True)
        export_onnx(net, "exports/selftest.onnx")
        diff = parity(net, "exports/selftest.onnx")
        good = diff < 1e-3
        print(f"3. ONNX parity: max |torch - onnxruntime| = {diff:.2e}  {'OK' if good else 'FAIL'}")
        ok = ok and good
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="default: 0.1 * batch / 256")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--emb", type=int, default=512, help="embedding size")
    ap.add_argument("--scale", type=float, default=64.0, help="ArcFace scale s")
    ap.add_argument("--margin", type=float, default=0.5, help="ArcFace angular margin m (radians)")
    ap.add_argument("--margin-warmup", type=int, default=3, help="epochs to ramp the margin from 0 to m")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--augment", action="store_true", help="brightness/contrast jitter + random erasing")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (CUDA only)")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--resume")
    ap.add_argument("--val", default="lfw", help="validation set evaluated after each epoch (lfw, calfw, cplfw, agedb_30, none)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    seed_everything(args.seed)
    print("device:", device)
    if args.selftest:
        selftest(device)

    paths = load_paths()
    manifest = INDEX_DIR / "train_manifest.csv"
    if not manifest.exists():
        raise SystemExit("Run step2_clean_and_index.py first.")
    ds = ManifestDataset(paths["train_root"], manifest, augment=args.augment)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    print(f"{len(ds)} images, {ds.n_classes} identities, {len(dl)} steps per epoch")

    net = build_backbone(args.emb).to(device)
    head = ArcFace(args.emb, ds.n_classes, args.scale, args.margin).to(device)
    print(f"backbone parameters: {count_params(net) / 1e6:.2f} M | ArcFace head: {count_params(head) / 1e6:.2f} M (training only)")
    base_lr = args.lr or 0.1 * args.batch / 256
    opt = torch.optim.SGD([{"params": net.parameters()}, {"params": head.parameters()}],
                          lr=base_lr, momentum=0.9, weight_decay=args.wd)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    all_params = list(net.parameters()) + list(head.parameters())

    def train_step(x, y):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext():
            emb = net(x)
        logits = head(emb.float(), y)  # the head always runs in fp32 (logits reach +-64)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            acc = (head.cos_logits(emb.detach().float()).argmax(1) == y).float().mean().item()
        return loss.item(), acc

    if args.bench:
        net.train()
        it, t0, n = iter(dl), None, 0
        for i in range(30):
            if i == 5:
                t0 = time.time()
            train_step(*next(it))
            n += args.batch if i >= 5 else 0
        ips = n / (time.time() - t0)
        print(f"speed: {ips:.0f} images/s -> one epoch of {len(ds)} images = {len(ds) / ips / 60:.1f} min, "
              f"{args.epochs} epochs = {args.epochs * len(ds) / ips / 3600:.1f} h")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["net"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", 0.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    def save(name, epoch):
        torch.save({"net": net.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best": best, "args": vars(args), "n_classes": ds.n_classes}, out / name)

    log_path = out / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,loss,train_acc,val_acc,lr,minutes\n")
    total_steps, warmup = args.epochs * len(dl), len(dl)
    step = start_epoch * len(dl)
    val_root = paths.get("val_root") if args.val != "none" else None

    for epoch in range(start_epoch, args.epochs):
        head.set_margin(args.margin * min(1.0, epoch / max(1, args.margin_warmup)))
        net.train()
        loss_m, acc_m, t0 = AverageMeter(), AverageMeter(), time.time()
        for x, y in dl:
            lr = lr_at(step, total_steps, warmup, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            loss, acc = train_step(x, y)
            loss_m.update(loss, x.size(0))
            acc_m.update(acc, x.size(0))
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch + 1}/{args.epochs} step {step} loss {loss_m.avg:.3f} acc {acc_m.avg:.3f} "
                      f"lr {lr:.4f} margin {head.m:.2f} | {loss_m.n / (time.time() - t0):.0f} img/s", flush=True)
        val_acc = float("nan")
        if val_root:
            val_acc, val_std, _ = evaluate_ann(torch_feature_fn(net, device), val_root, f"{args.val}_ann.txt")
            print(f"== epoch {epoch + 1}: {args.val} accuracy {val_acc:.4f} +- {val_std:.4f}")
            if val_acc > best:
                best = val_acc
                save("best.pt", epoch)
        save("last.pt", epoch)
        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{loss_m.avg:.4f},{acc_m.avg:.4f},{val_acc:.4f},{lr:.5f},{(time.time() - t0) / 60:.1f}\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(open(log_path)))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot([float(r["loss"]) for r in rows])
        ax[0].set_title("training loss")
        ax[1].plot([float(r["val_acc"]) for r in rows])
        ax[1].set_title(f"{args.val} accuracy")
        fig.savefig(out / "curves.png", dpi=120, bbox_inches="tight")
    except Exception as e:  # plotting must never hide a finished training
        print("could not plot curves:", e)
    print("done. best", args.val, "accuracy:", best, "| checkpoints in", out)


if __name__ == "__main__":
    main()
