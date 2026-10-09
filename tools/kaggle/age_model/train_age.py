"""Fine-tune ConvNeXt-Tiny on FairFace TRAIN for the nine FairFace age buckets; export ONNX.

Runs in the GPU training kernel (see kernel_train.py). Inputs are the prep kernel's crop records
(prep_faces.py) and the FairFace 1.25-padding TRAIN parquets, downloaded from the same Hugging
Face mirror and revision as tools/fetch_fairface.py. Every image is rebuilt into the app's own
aligned 224x224 face by faceprep.aligned_face, so training sees what the app feeds the model.

A stratified slice of TRAIN (5%, by age and race) is held back for model selection and early
stopping. FairFace VALIDATION is never downloaded here: it is the held-out test set.

    python train_age.py --prep DIR --out DIR [--epochs 14] [--aux-weight 0] [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faceprep  # noqa: E402
from fetch_fairface import REPO_ID, REVISION  # noqa: E402

N_AGE, N_GENDER, N_RACE = 9, 2, 7
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


class ImageStore:
    """Every needed JPEG in one flat uint8 buffer, so forked loader workers share it untouched."""

    def __init__(self, parquets: list[Path], shards: np.ndarray, rows: np.ndarray):
        import pyarrow.parquet as pq

        columns = [pq.read_table(path, columns=["image"]).column("image").combine_chunks().field("bytes")
                   for path in parquets]
        chunks = [np.frombuffer(columns[int(shard)][int(row)].as_py(), np.uint8)
                  for shard, row in zip(shards, rows, strict=True)]
        self.offsets = np.zeros(len(chunks) + 1, np.int64)
        self.offsets[1:] = np.cumsum([len(chunk) for chunk in chunks])
        self.buffer = np.concatenate(chunks)

    def frame(self, i: int) -> np.ndarray:
        """Decode sample i's image as BGR, exactly as the eval reads FairFace (cv2.imdecode)."""
        return cv2.imdecode(self.buffer[self.offsets[i]:self.offsets[i + 1]], cv2.IMREAD_COLOR)


def augment(face: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild photometric and geometric jitter on the aligned uint8 BGR face."""
    if rng.random() < 0.5:
        face = face[:, ::-1]
    size = face.shape[0]
    angle, scale = rng.uniform(-10, 10), rng.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    m[:, 2] += rng.uniform(-0.05, 0.05, 2) * size
    face = cv2.warpAffine(np.ascontiguousarray(face), m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    img = face.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)  # contrast, brightness
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * rng.uniform(0.6, 1.4)  # saturation
    if rng.random() < 0.05:
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    face = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:  # low-resolution sources (webcams, small faces)
        small = int(rng.uniform(48, 112))
        face = cv2.resize(cv2.resize(face, (small, small), interpolation=cv2.INTER_AREA), (size, size))
    return face


def crop_record(meta: dict, i: int) -> dict:
    """Return sample i's crop record in the form faceprep.aligned_face takes."""
    angle = meta["angle"][i]
    return {"box": meta["box"][i], "angle": None if np.isnan(angle) else float(angle),
            "landmarks": meta["landmarks"][i]}


class FaceDataset(Dataset):
    """Rebuild the app's aligned face per sample; augment only when training."""

    def __init__(self, store: ImageStore, meta: dict, indices: np.ndarray, train: bool):
        self.store, self.meta, self.indices, self.train = store, meta, indices, train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        face = faceprep.aligned_face(self.store.frame(i), crop_record(self.meta, i))
        if self.train:
            rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
            face = augment(face, rng)
        blob = faceprep.imagenet_blob(face)[0]
        return (torch.from_numpy(blob), int(self.meta["age"][i]), int(self.meta["gender"][i]),
                int(self.meta["race"][i]))


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
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (age probabilities, true buckets) over a loader."""
    model.eval()
    probs, truth = [], []
    for x, age, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last))[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--prep", type=Path, required=True, help="prep kernel output directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/tmp/age/fairface"))
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-budget-min", type=float, default=170.0,
                        help="wall-clock budget for setup plus training; the schedule shrinks to fit")
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
    meta = {key: prep[key] for key in ("shard", "row", "age", "gender", "race", "box", "angle", "landmarks")}
    files = [str(f) for f in prep["files"]]
    if any("validation" in f or not f.startswith("1.25/train-") for f in files):
        raise SystemExit(f"refusing non-TRAIN inputs: {files}")
    detected = np.flatnonzero(prep["detected"])
    train_idx, select_idx = selection_split(meta["age"][detected], meta["race"][detected], SEED)
    train_idx, select_idx = detected[train_idx], detected[select_idx]
    if args.smoke:
        train_idx, select_idx = train_idx[:256], select_idx[:128]
        args.epochs, args.batch, args.workers = 2, 32, 2
    emit({"event": "split", "train_images": len(prep["row"]), "detected": len(detected),
          "train": len(train_idx), "select": len(select_idx)})

    from huggingface_hub import hf_hub_download

    parquets = [Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION,
                                     local_dir=args.data)) for name in files]
    store = ImageStore(parquets, meta["shard"], meta["row"])

    # Parity: rebuilt crops must equal the aligned faces the app built in the prep kernel.
    refs = np.load(args.prep / "reference_crops.npz")
    for i, crop in zip(refs["index"], refs["crops"], strict=True):
        if not np.array_equal(faceprep.aligned_face(store.frame(int(i)), crop_record(meta, int(i))), crop):
            raise SystemExit(f"parity check failed for sample {i}")
    emit({"event": "parity", "reference_crops_identical": len(refs["index"])})

    train_loader = DataLoader(FaceDataset(store, meta, train_idx, True), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    select_loader = DataLoader(FaceDataset(store, meta, select_idx, False), batch_size=256,
                               num_workers=args.workers, pin_memory=True)

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
    steps_per_epoch = len(train_loader)
    total_steps, warmup = steps_per_epoch * args.epochs, steps_per_epoch
    emit({"event": "start", "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
          "gpus": torch.cuda.device_count(), "steps_per_epoch": steps_per_epoch, "args": {
              k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})

    best = {"accuracy": -1.0}
    best_state = None
    stale = 0
    step = 0
    epoch = 0
    loop_started = time.time()
    while epoch < args.epochs:
        epoch += 1
        model.train()
        running, seen = 0.0, 0
        for x, age, gender, race in train_loader:
            factor = step / warmup if step < warmup else 0.5 * (1 + math.cos(
                math.pi * (step - warmup) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * max(factor, 0.01)
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
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
            running += float(loss) * len(x)
            seen += len(x)
            step += 1
            if step % 200 == 0:
                emit({"event": "step", "step": step, "loss": round(running / seen, 4)})

        scores = {}
        for name, net in (("raw", model), ("ema", ema.module)):
            probs, truth = predict(net, select_loader, device)
            scores[name] = metrics(probs, truth)
        emit({"event": "epoch", "epoch": epoch, "train_loss": round(running / max(seen, 1), 4), **scores})
        if epoch == 1:
            # Shorten the cosine schedule up front if the planned epochs would overrun the GPU budget.
            per_epoch = time.time() - loop_started
            remaining = args.time_budget_min * 60 - (time.time() - started)
            fits = max(2, 1 + int(remaining // per_epoch))
            if fits < args.epochs:
                emit({"event": "schedule_shortened", "epochs": fits, "planned": args.epochs,
                      "epoch_minutes": round(per_epoch / 60, 1)})
                args.epochs = fits
                total_steps = steps_per_epoch * args.epochs
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
    probs, truth = predict(final, select_loader, device)
    np.savez_compressed(args.out / "select_predictions.npz", index=select_idx, probs=probs, truth=truth)
    torch.save(best_state, args.out / "age_model_best.pt")

    onnx_path = args.out / "age_convnext_fairface.onnx"
    export_onnx(final, onnx_path)
    batch = torch.stack([select_loader.dataset[k][0] for k in range(min(64, len(select_idx)))])
    with torch.no_grad():
        reference = AgeOnly(copy.deepcopy(final).float().cpu().eval())(batch).numpy()
    np.save(args.out / "check_x.npy", batch.numpy())
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
        "best": best, "select_final": metrics(probs, truth), "cv2_check": cv2_check,
        "onnx": {"file": onnx_path.name, "opset": 17, "input": "input 1x3x224x224 RGB ImageNet-normalized",
                 "output": "age_logits 1x9 (FairFace buckets 0-2 ... 70+)"},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "timm": timm.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__,
                     "onnx": metadata.version("onnx")},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "gpus": torch.cuda.device_count(), "minutes": round((time.time() - started) / 60, 1),
    }
    (args.out / "selection_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit({"event": "done", **{k: summary[k] for k in ("best", "select_final", "cv2_check", "minutes")}})


if __name__ == "__main__":
    main()
