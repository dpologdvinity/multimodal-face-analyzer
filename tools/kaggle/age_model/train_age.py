"""Fine-tune ConvNeXt-Tiny on FairFace TRAIN for the nine FairFace age buckets; export ONNX.

Runs in the GPU training kernel (see kernel_train.py). Inputs are the prep kernel's labels
(prep_faces.py) and the crops kernel's uint8 array of the app's own aligned 224x224 faces
(build_crops.py, rebuilt with faceprep.aligned_face from FairFace 1.25-padding TRAIN images),
so training sees what the app feeds the model. The crops are memory-mapped and augmented on
the GPU; the CPU only gathers batches.

A stratified slice of TRAIN (5%, by age and race) is held back for model selection and early
stopping. FairFace VALIDATION is never read here: it is the held-out test set.

    python train_age.py --prep DIR --crops DIR --out DIR [--epochs 14] [--max-steps N] [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faceprep  # noqa: E402
from fetch_fairface import REPO_ID, REVISION  # noqa: E402

from face_analyzer.core.constants import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

N_AGE, N_GENDER, N_RACE = 9, 2, 7
SIZE = 224
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
BACKBONE_REPO = "timm/convnext_tiny.fb_in22k_ft_in1k"
BACKBONE_REVISION = "bc48a87f119bae1b06418f5340bfa97ef75d4609"
BACKBONE_LICENSE = ("Apache-2.0 (Hugging Face model card of timm/convnext_tiny.fb_in22k_ft_in1k); "
                    "upstream facebookresearch/ConvNeXt weights are MIT")
SELECT_FRACTION = 0.05
SEED = 0


# --------------------------------------------------------------------------- data


def selection_split(age: np.ndarray, race: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into (train, select): SELECT_FRACTION of every (age, race) stratum is held back."""
    rng = np.random.default_rng([seed, 7])
    train, select = [], []
    for stratum in sorted(set(zip(age.tolist(), race.tolist(), strict=True))):
        members = np.flatnonzero((age == stratum[0]) & (race == stratum[1]))
        members = rng.permutation(members)
        k = int(round(SELECT_FRACTION * len(members)))
        select.extend(members[:k].tolist())
        train.extend(members[k:].tolist())
    return np.sort(np.asarray(train)), np.sort(np.asarray(select))


class CropBatches:
    """Batches of (uint8 BGR NHWC crops, age, gender, race), gathered by a background thread."""

    def __init__(self, crops: np.ndarray, meta: dict, indices: np.ndarray, batch: int,
                 shuffle: bool, seed: int = SEED, prefetch: int = 6):
        self.crops, self.meta, self.indices = crops, meta, indices
        self.batch, self.shuffle, self.prefetch = batch, shuffle, prefetch
        self.rng = np.random.default_rng([seed, 11])

    def __len__(self) -> int:
        return len(self.indices) // self.batch if self.shuffle else -(-len(self.indices) // self.batch)

    def _batches(self):
        order = self.rng.permutation(self.indices) if self.shuffle else self.indices
        for k in range(len(self)):
            # Sorted within the batch so the memory-mapped reads go forward through the file.
            idx = np.sort(order[k * self.batch:(k + 1) * self.batch])
            x = torch.from_numpy(np.ascontiguousarray(self.crops[idx]))
            if torch.cuda.is_available():
                x = x.pin_memory()
            labels = [torch.from_numpy(self.meta[key][idx].astype(np.int64)) for key in ("age", "gender", "race")]
            yield (x, *labels)

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        done = object()

        def fill():
            for item in self._batches():
                q.put(item)
            q.put(done)

        threading.Thread(target=fill, daemon=True).start()
        while (item := q.get()) is not done:
            yield item


class GpuInput:
    """Turn uint8 BGR NHWC crops into the app's normalized RGB input, with optional GPU jitter."""

    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(SEED)

    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, device=self.device, generator=self.gen)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Mild geometric and photometric jitter on 0-255 float BGR NCHW faces."""
        n = x.shape[0]
        x = torch.where((self._rand(n) < 0.5)[:, None, None, None], x.flip(3), x)
        angle = torch.deg2rad(self._rand(n) * 20 - 10)
        scale = 0.9 + 0.2 * self._rand(n)
        theta = torch.zeros(n, 2, 3, device=self.device)
        theta[:, 0, 0] = torch.cos(angle) / scale
        theta[:, 0, 1] = -torch.sin(angle) / scale
        theta[:, 1, 0] = torch.sin(angle) / scale
        theta[:, 1, 1] = torch.cos(angle) / scale
        theta[:, :, 2] = (torch.rand(n, 2, device=self.device, generator=self.gen) - 0.5) * 0.2
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        x = x * (0.7 + 0.6 * self._rand(n))[:, None, None, None] + (self._rand(n) * 50 - 25)[:, None, None, None]
        gray = x.mean(1, keepdim=True)
        x = gray + (x - gray) * (0.6 + 0.8 * self._rand(n))[:, None, None, None]
        x = torch.where((self._rand(n) < 0.05)[:, None, None, None], x.mean(1, keepdim=True).expand_as(x), x)
        x = x.clamp(0, 255)
        low = torch.nonzero(self._rand(n) < 0.15).flatten().tolist()  # webcams, small faces
        if low:
            sizes = (48 + self._rand(len(low)) * 64).long().tolist()
            x = x.clone()
            for i, small in zip(low, sizes, strict=True):
                shrunk = F.interpolate(x[i:i + 1], size=(small, small), mode="area")
                x[i:i + 1] = F.interpolate(shrunk, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        return x

    def __call__(self, crops: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """Return the normalized NCHW float32 batch (identical to faceprep.imagenet_blob unaugmented)."""
        x = crops.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()
        if augment:
            x = self.augment(x)
        x = x.flip(1)  # BGR to RGB
        return ((x / 255.0 - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------- model


class AgeNet(nn.Module):
    """ConvNeXt-Tiny trunk with an age head and optional gender/race auxiliary heads."""

    def __init__(self, trunk: nn.Module, features: int, aux: bool):
        super().__init__()
        self.trunk = trunk
        self.age = nn.Linear(features, N_AGE)
        self.aux = nn.Linear(features, N_GENDER + N_RACE) if aux else None

    def forward(self, x):
        z = self.trunk(x)
        age = self.age(z)
        if self.aux is None:
            return age, None
        return age, self.aux(z)


class AgeOnly(nn.Module):
    """The exported graph: age logits only."""

    def __init__(self, net: AgeNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[0]


def build_model(aux: bool, drop_path: float) -> AgeNet:
    """Load the ImageNet weights into a 1x1-conv-MLP ConvNeXt (cv2.dnn cannot import the Linear MLP)."""
    import timm
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(BACKBONE_REPO, "model.safetensors", revision=BACKBONE_REVISION)
    reference = timm.create_model(BACKBONE, pretrained=True, num_classes=0,
                                  pretrained_cfg_overlay={"file": weights}).eval()
    trunk = timm.create_model(BACKBONE, pretrained=False, num_classes=0, conv_mlp=True,
                              drop_path_rate=drop_path)
    target = trunk.state_dict()
    source = reference.state_dict()
    if set(source) != set(target):
        raise RuntimeError(f"state dict keys differ: {sorted(set(source) ^ set(target))[:10]}")
    trunk.load_state_dict({k: v.reshape(target[k].shape) for k, v in source.items()})
    trunk.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        diff = float((trunk(x) - reference(x)).abs().max())
    if diff > 1e-3:
        raise RuntimeError(f"conv-MLP conversion changed the features (max diff {diff})")
    print(f"pretrained trunk converted to conv MLP, max feature diff {diff:.2e}", flush=True)
    return AgeNet(trunk, trunk.num_features, aux)


def ordinal_targets(age: torch.Tensor, sigma: float) -> torch.Tensor:
    """Soft labels: a Gaussian over bucket indices centred on the true bucket (sigma in buckets)."""
    grid = torch.arange(N_AGE, device=age.device, dtype=torch.float32)
    logits = -((grid[None, :] - age[:, None].float()) ** 2) / (2 * sigma ** 2)
    return torch.softmax(logits, dim=1)


# --------------------------------------------------------------------------- loops


@torch.no_grad()
def predict(model: nn.Module, batches: CropBatches, to_input: GpuInput) -> tuple[np.ndarray, np.ndarray]:
    """Return (age probabilities, true buckets) over unaugmented batches."""
    model.eval()
    probs, truth = [], []
    for x, age, _, _ in batches:
        with torch.autocast("cuda", dtype=torch.float16, enabled=to_input.device.type == "cuda"):
            logits = model(to_input(x))[0]
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        truth.append(age.numpy())
    return np.concatenate(probs), np.concatenate(truth)


def metrics(probs: np.ndarray, truth: np.ndarray) -> dict:
    """Bucket accuracy, within-one-bucket rate and mean bucket offset."""
    offset = np.abs(probs.argmax(1) - truth)
    return {"accuracy": round(float((offset == 0).mean()), 4),
            "within_one": round(float((offset <= 1).mean()), 4),
            "mean_offset": round(float(offset.mean()), 4)}


def export_onnx(model: AgeNet, path: Path) -> None:
    """Export the age head to ONNX (opset 17, the TorchScript exporter: cv2.dnn imports it)."""
    wrapped = AgeOnly(copy.deepcopy(model).float().cpu().eval())
    x = torch.randn(1, 3, 224, 224)
    kwargs = {"opset_version": 17, "input_names": ["input"], "output_names": ["age_logits"]}
    try:
        torch.onnx.export(wrapped, x, str(path), dynamo=False, **kwargs)
    except TypeError:  # torch releases before the dynamo switch only have the TorchScript exporter
        torch.onnx.export(wrapped, x, str(path), **kwargs)


CV2_CHECK = r"""
import json, sys, numpy as np, cv2
net = cv2.dnn.readNetFromONNX(sys.argv[1])
x = np.load(sys.argv[2]); ref = np.load(sys.argv[3])
out = []
for i in range(len(x)):
    net.setInput(x[i:i + 1]); out.append(net.forward().reshape(-1))
out = np.stack(out)
print(json.dumps({"cv2": cv2.__version__, "max_abs_diff": float(np.abs(out - ref).max()),
                  "argmax_agree": float((out.argmax(1) == ref.argmax(1)).mean())}))
"""




def diagnose(crops: np.ndarray, meta: dict, train_idx: np.ndarray, to_input: GpuInput, batch: int,
             emit) -> None:
    """Time the input pipeline and the training step variants separately, then return."""
    def timed(fn, warmup: int, runs: int) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(runs):
            fn()
        torch.cuda.synchronize()
        return round((time.time() - t) / runs, 3)

    batches = iter(CropBatches(crops, meta, train_idx, batch, shuffle=True))
    t = time.time()
    for _ in range(30):
        x, *_ = next(batches)
    emit({"event": "diag_data", "s_per_batch": round((time.time() - t) / 30, 3), **memory_note()})
    emit({"event": "diag_input", "augment_s": timed(lambda: to_input(x, augment=True), 2, 10),
          "plain_s": timed(lambda: to_input(x), 2, 10)})
    model = build_model(False, 0.1).cuda()
    targets = torch.randint(0, N_AGE, (batch,), device="cuda")
    for bench in (False, True):
        torch.backends.cudnn.benchmark = bench
        for layout in ("channels_last", "contiguous"):
            fmt = torch.channels_last if layout == "channels_last" else torch.contiguous_format
            model = model.to(memory_format=fmt).train()
            for gpus in (1, 2):
                net = nn.DataParallel(model) if gpus == 2 else model
                opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
                scaler = torch.amp.GradScaler("cuda")
                inp = to_input(x).contiguous(memory_format=fmt)

                def step(net=net, inp=inp, opt=opt, scaler=scaler):
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = F.cross_entropy(net(inp)[0].float(), targets)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                emit({"event": "diag_step", "cudnn_benchmark": bench, "layout": layout, "gpus": gpus,
                      "s_per_step": timed(step, 2, 5), **memory_note()})


def memory_note() -> dict:
    """Host RAM and GPU memory in GB, for the log."""
    out = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        out.update(ram_used_gb=round(vm.used / 2**30, 1), ram_available_gb=round(vm.available / 2**30, 1))
    except ImportError:
        pass
    if torch.cuda.is_available():
        out["gpu_max_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--prep", type=Path, required=True, help="prep kernel output directory")
    parser.add_argument("--crops", type=Path, required=True, help="crops kernel output directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4, help="peak learning rate of the trunk")
    parser.add_argument("--head-lr-mult", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.5, help="ordinal label spread, in buckets")
    parser.add_argument("--aux-weight", type=float, default=0.0, help="gender+race loss weight; 0 = off")
    parser.add_argument("--ema", type=float, default=0.9995)
    parser.add_argument("--time-budget-min", type=float, default=150.0,
                        help="wall-clock budget for setup plus training; the schedule shrinks to fit")
    parser.add_argument("--diagnose", action="store_true", help="time pipeline parts and step variants")
    parser.add_argument("--max-steps", type=int, default=0, help="timing check: stop after N steps, no export")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-safe run to check the plumbing")
    args = parser.parse_args()

    started = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = (args.out / "train_log.jsonl").open("w")

    def emit(event: dict) -> None:
        event = {"t_min": round((time.time() - started) / 60, 2), **event}
        print(json.dumps(event), flush=True)
        log.write(json.dumps(event) + "\n")
        log.flush()

    prep = np.load(args.prep / "faces_train.npz")
    meta = {key: prep[key] for key in ("age", "gender", "race")}
    files = [str(f) for f in prep["files"]]
    if any("validation" in f or not f.startswith("1.25/train-") for f in files):
        raise SystemExit(f"refusing non-TRAIN inputs: {files}")
    crops = np.load(args.crops / "crops_train.npy", mmap_mode="r")
    if crops.shape != (len(prep["row"]), SIZE, SIZE, 3):
        raise SystemExit(f"crops shape {crops.shape} does not match {len(prep['row'])} records")
    detected = np.flatnonzero(prep["detected"])
    train_idx, select_idx = selection_split(meta["age"][detected], meta["race"][detected], SEED)
    train_idx, select_idx = detected[train_idx], detected[select_idx]
    if args.smoke:
        train_idx, select_idx = train_idx[:256], select_idx[:128]
        args.epochs, args.batch = 2, 32
    emit({"event": "split", "train_images": len(prep["row"]), "detected": len(detected),
          "train": len(train_idx), "select": len(select_idx)})

    # Parity: the crops must be the app-built references, and the unaugmented GPU input must be
    # the app's own blob (faceprep.imagenet_blob) of each crop.
    to_input = GpuInput(device)
    refs = np.load(args.prep / "reference_crops.npz")
    if not all(np.array_equal(crops[int(i)], crop) for i, crop in zip(refs["index"], refs["crops"], strict=True)):
        raise SystemExit("crops differ from the app-built reference crops")
    blob_diff = max(float(np.abs(to_input(torch.from_numpy(crop[None])).cpu().float().numpy()
                                 - faceprep.imagenet_blob(crop)).max()) for crop in refs["crops"][:16])
    emit({"event": "parity", "reference_crops_identical": len(refs["index"]), "input_max_abs_diff": blob_diff})
    if blob_diff > 1e-5:
        raise SystemExit("GPU input normalization differs from the app's")

    if args.diagnose:
        diagnose(crops, meta, train_idx, to_input, args.batch, emit)
        return
    train_batches = CropBatches(crops, meta, train_idx, args.batch, shuffle=True)
    select_batches = CropBatches(crops, meta, select_idx, 256, shuffle=False)

    from timm.utils import ModelEmaV3

    model = build_model(args.aux_weight > 0, args.drop_path).to(device).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=args.ema)
    trunk_params = list(model.trunk.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("trunk.")]
    no_decay = {id(p) for p in trunk_params if p.ndim <= 1}
    groups = [
        {"params": [p for p in trunk_params if id(p) not in no_decay], "lr": args.lr},
        {"params": [p for p in trunk_params if id(p) in no_decay], "lr": args.lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ]
    for group in groups:
        group["base_lr"] = group["lr"]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    parallel = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    steps_per_epoch = len(train_batches)
    total_steps, warmup = steps_per_epoch * args.epochs, steps_per_epoch
    emit({"event": "start", "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
          "gpus": torch.cuda.device_count(), "steps_per_epoch": steps_per_epoch, **memory_note(), "args": {
              k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})

    best = {"accuracy": -1.0}
    best_state = None
    stale = step = epoch = 0
    sized = False
    loop_started = time.time()
    tick = time.time()
    while epoch < args.epochs:
        epoch += 1
        model.train()
        running, seen = 0.0, 0
        for x, age, gender, race in train_batches:
            factor = step / warmup if step < warmup else 0.5 * (1 + math.cos(
                math.pi * (step - warmup) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * max(factor, 0.01)
            x = to_input(x, augment=True)
            age, gender, race = age.to(device), gender.to(device), race.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                age_logits, aux_logits = parallel(x)
            loss = torch.sum(-ordinal_targets(age, args.sigma) * F.log_softmax(age_logits.float(), 1), 1).mean()
            if aux_logits is not None:
                aux_logits = aux_logits.float()
                loss = loss + args.aux_weight * (F.cross_entropy(aux_logits[:, :N_GENDER], gender)
                                                 + F.cross_entropy(aux_logits[:, N_GENDER:], race))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            running += float(loss.detach()) * len(x)
            seen += len(x)
            step += 1
            if step % 50 == 0:
                per_step = (time.time() - tick) / 50
                tick = time.time()
                emit({"event": "step", "step": step, "loss": round(running / seen, 4),
                      "s_per_step": round(per_step, 3), **memory_note()})
                if not sized and step >= 100:
                    # Size the cosine schedule from the measured speed so the run fits the budget.
                    sized = True
                    epoch_s = per_step * steps_per_epoch + 60
                    remaining = args.time_budget_min * 60 - (time.time() - started) + (time.time() - loop_started)
                    fits = max(1, int(remaining // epoch_s))
                    if fits < args.epochs:
                        args.epochs = fits
                        total_steps = steps_per_epoch * args.epochs
                    emit({"event": "schedule", "epochs": args.epochs, "epoch_minutes": round(epoch_s / 60, 1)})
            if args.max_steps and step >= args.max_steps:
                emit({"event": "timing_check_done", "steps": step, **memory_note()})
                return

        scores = {}
        for name, net in (("raw", model), ("ema", ema.module)):
            probs, truth = predict(net, select_batches, to_input)
            scores[name] = metrics(probs, truth)
        emit({"event": "epoch", "epoch": epoch, "train_loss": round(running / max(seen, 1), 4), **scores,
              **memory_note()})
        tick = time.time()
        name = max(scores, key=lambda key: (scores[key]["accuracy"], scores[key]["within_one"]))
        if scores[name]["accuracy"] > best["accuracy"]:
            best = {**scores[name], "epoch": epoch, "weights": name}
            best_state = copy.deepcopy((model if name == "raw" else ema.module).state_dict())
            torch.save(best_state, args.out / "age_model_best.pt")
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                emit({"event": "early_stop", "epoch": epoch})
                break

    final = copy.deepcopy(model)
    final.load_state_dict(best_state)
    final = final.to(device).to(memory_format=torch.channels_last).eval()
    probs, truth = predict(final, select_batches, to_input)
    np.savez_compressed(args.out / "select_predictions.npz", index=select_idx, probs=probs, truth=truth)

    onnx_path = args.out / "age_convnext_fairface.onnx"
    export_onnx(final, onnx_path)
    batch = np.concatenate([faceprep.imagenet_blob(crops[int(i)]) for i in select_idx[:64]])
    with torch.no_grad():
        reference = AgeOnly(copy.deepcopy(final).float().cpu().eval())(torch.from_numpy(batch)).numpy()
    np.save(args.out / "check_x.npy", batch)
    np.save(args.out / "check_ref.npy", reference)
    check = subprocess.run([sys.executable, "-c", CV2_CHECK, str(onnx_path), str(args.out / "check_x.npy"),
                            str(args.out / "check_ref.npy")], capture_output=True, text=True)
    cv2_check = json.loads(check.stdout.strip().splitlines()[-1]) if check.returncode == 0 else {
        "error": check.stderr[-2000:]}
    (args.out / "check_x.npy").unlink()
    (args.out / "check_ref.npy").unlink()

    from importlib import metadata

    import timm

    summary = {
        "backbone": {"timm_name": BACKBONE, "hf_repo": BACKBONE_REPO, "revision": BACKBONE_REVISION,
                     "license": BACKBONE_LICENSE, "mlp": "1x1 conv (conv_mlp=True), weights reshaped"},
        "data": {"dataset": REPO_ID, "revision": REVISION, "files": files, "license": "CC BY 4.0",
                 "train_images": int(len(prep["row"])), "detected": int(len(detected)),
                 "train": int(len(train_idx)), "select": int(len(select_idx)),
                 "select_rule": f"{SELECT_FRACTION:.0%} of every (age, race) stratum of TRAIN, seed {SEED}",
                 "validation_used": False},
        "loss": f"cross-entropy on Gaussian ordinal soft labels (sigma {args.sigma} buckets)"
                + (f" + {args.aux_weight} x (gender + race CE)" if args.aux_weight else ""),
        "augmentation": "on GPU: flip, rotation +-10 deg, scale 0.9-1.1, shift +-5%, contrast 0.7-1.3, "
                        "brightness +-25, saturation 0.6-1.4, 5% grayscale, 15% down-up resample to 48-112 px",
        "best": best, "select_final": metrics(probs, truth), "cv2_check": cv2_check,
        "onnx": {"file": onnx_path.name, "opset": 17, "input": "input 1x3x224x224 RGB ImageNet-normalized",
                 "output": "age_logits 1x9 (FairFace buckets 0-2 ... 70+)"},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "timm": timm.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__, "onnx": metadata.version("onnx")},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "gpus": torch.cuda.device_count(), "minutes": round((time.time() - started) / 60, 1),
    }
    (args.out / "selection_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit({"event": "done", **{k: summary[k] for k in ("best", "select_final", "cv2_check", "minutes")}})


if __name__ == "__main__":
    main()
