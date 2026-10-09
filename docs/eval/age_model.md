# A new age model: ConvNeXt-Tiny fine-tuned on FairFace (not adopted)

A ConvNeXt-Tiny was fine-tuned on the FairFace training split for FairFace's nine age buckets,
on exactly the face crop the app gives its FairFace-style backends, to try to beat the best age
backend, `mivolo`. The rule set in advance: integrate it only if it beats `mivolo` on the
held-out FairFace test half (paired-bootstrap 95% interval of the difference above zero) and is
not clearly worse out of distribution (on UTKFace, the interval's upper end not below zero).
**It failed both conditions, so it is not an app backend and no weights are shipped.** The
numbers are in [age_model.json](age_model.json) (UTKFace, the FairFace extract and the training
record) and [heldout_fairface.json](heldout_fairface.json) (FairFace, every backend).

## Results

### FairFace validation, test half (in-distribution for the new model)

The same 1,002 test faces as [heldout_fairface.md](heldout_fairface.md); 95% bootstrap intervals
in brackets, and the last column is the paired per-face difference:

| Backend | Age bucket | Within one bucket | Mean bucket offset | Bucket acc. - `mivolo` |
| ------- | ---------- | ----------------- | ------------------ | ---------------------- |
| `mivolo` (shipped headline) | 62.3% (59.2-65.1) | 97.0% (95.9-98.0) | 0.42 (0.38-0.45) | - |
| new ConvNeXt-Tiny (`convnext`) | 60.0% (56.7-63.1) | 96.6% (95.4-97.7) | 0.44 (0.40-0.47) | -2.3 pp (-5.6 to +1.2) |
| `fairface` | 60.3% (57.2-63.3) | 95.4% (94.1-96.6) | 0.47 (0.42-0.51) | -2.0 pp (-5.2 to +1.7) |

Within one bucket, the new model minus `mivolo` is -0.4 pp (-1.6 to +0.8). The new model is
statistically indistinguishable from both `mivolo` and the FairFace authors' own ResNet-34
(`fairface`), which was trained on the same split. Even with in-distribution data, it does not
clear the bar.

### UTKFace (out of distribution for the new model)

1,498 detected faces of 1,500 drawn (seed 0, stratified by age decade); UTKFace gives exact
ages, so the year error is reported for the backends that output a number. Bucket accuracy maps
the true age onto FairFace's buckets the same way a predicted age is mapped.

| Backend | Age bucket | Within one bucket | MAE (years) | Bucket acc. - `mivolo` | Gender |
| ------- | ---------- | ----------------- | ----------- | ---------------------- | ------ |
| `mivolo` (shipped headline) | 67.5% (65.1-69.8) | 98.4% (97.8-99.0) | 3.7 (3.5-3.9) | - | 98.4% (97.7-99.0) |
| new ConvNeXt-Tiny (`convnext`) | 58.8% (56.3-61.4) | 95.2% (94.1-96.3) | - | -8.7 pp (-11.2 to -6.3) | - |
| `fairface` | 57.5% (55.1-60.2) | 93.1% (91.7-94.3) | - | -10.0 pp (-12.6 to -7.3) | 92.7% (91.3-93.9) |
| `dex` | 40.0% (37.5-42.7) | 82.4% (80.4-84.2) | 8.4 (8.1-8.8) | -27.5 pp (-30.6 to -24.4) | - |
| `caffe` | 35.9% (33.4-38.3) | 66.5% (63.9-68.9) | - | -31.6 pp (-34.7 to -28.4) | 79.0% (76.7-81.0) |

The new model is clearly worse than `mivolo` here (the whole interval is below zero), and only
slightly ahead of `fairface`. By true age, it matches or beats `mivolo` on the 0-2 and 20-29
buckets and falls behind from 30 upwards, by up to 30 points (30-39: 46.9% against 67.8%; 70+:
34.5% against 64.4%; `per_true_bucket` in the JSON), which is where FairFace's own labels are hardest. Within
one bucket, it trails `mivolo` by 3.2 pp (2.2 to 4.1).

### Gate decision

| Condition | Result | Passed |
| --------- | ------ | ------ |
| FairFace test half: interval of (new - `mivolo`) above zero | -2.3 pp (-5.6 to +1.2) | no |
| UTKFace: interval of (new - `mivolo`) not wholly below zero | -8.7 pp (-11.2 to -6.3) | no |

The age headline stays `mivolo`. Because a learned stacker over the existing backends also gained
nothing ([heldout_fairface.md](heldout_fairface.md#learned-stacker)), the evidence points to
FairFace's training labels as the limit: two different networks trained on them (ResNet-34 and
ConvNeXt-Tiny) land on the same accuracy, below a model trained on other data.

## What these numbers can and cannot show

- **FairFace numbers are in-distribution for the new model.** It was trained on FairFace's
  training split, with the same collection, label protocol and crop style as the validation
  images it is tested on. UTKFace is the fairer comparison with `mivolo`.
- **UTKFace may not be out of distribution for `mivolo`.** Its
  [model card](https://huggingface.co/iitolstykh/mivolo_v2) says only "proprietary and
  open-source datasets", so UTKFace overlap cannot be ruled out; `mivolo`'s lead there may be
  partly in-distribution. `dex` and `caffe` were trained on IMDB-WIKI and Adience.
- **UTKFace faces are tight crops, padded with black.** Kaggle's UTKFace copies hold the 200x200
  aligned-and-cropped faces only. A face that fills the frame defeats the SSD detector (see the
  padding note in [heldout_fairface.md](heldout_fairface.md#method)), so each face is centred in a
  black border of two thirds of its size on every side (466x466), about the face-to-frame ratio
  of the FairFace 1.25 images. Every backend sees the same padded image, but the black
  surround is itself unlike real photos.
- **UTKFace labels** come from its authors' annotation of ages and genders; its five file names
  without a full label (of 23,708) are skipped.
- **One training run.** One recipe (below) was trained once; a different recipe might do
  somewhat better. The training split's own held-back slice predicted the outcome well (61.0%
  there, 60.0% on the FairFace test half), so the limit is not overfitting to that slice.

## Training

- **Data.** FairFace TRAIN, 1.25 padding, from the Hugging Face mirror
  [`HuggingFaceM4/FairFace`](https://huggingface.co/datasets/HuggingFaceM4/FairFace) at the
  revision `tools/fetch_fairface.py` pins (Karkkainen and Joo, WACV 2021; CC BY 4.0). Of
  86,744 images, 86,654 (99.9%) had a face found by the app's detector; 5% of every (age,
  race) stratum of those (4,332 images, seed 0) was held back for choosing the epoch, leaving
  82,322 for training. FairFace VALIDATION was never read during training or selection.
- **Input.** The app's own crop: SSD detection with the largest in-frame box (as the eval keeps),
  the app's roll leveling and MediaPipe landmarks, then the landmark-aligned (or margin-cropped)
  224x224 face with ImageNet normalization, exactly what `fairface` receives. A CPU kernel ran
  the app's code (pinned commit, the app's package pins, Python 3.11) on every training image and
  stored the few numbers that fix each crop (box, roll angle, landmarks); 89.7% of the found
  faces were landmark-aligned, against 89.9% in the eval. A second CPU kernel rebuilt the crops from those
  numbers with the app's functions, and the GPU kernel trained from them. The rebuilt crops of
  64 training images are byte-identical to the ones the app built, and
  `tests/test_age_model_parity.py` checks that the rebuild equals the tensor `analyze_frame`
  feeds the model (including the roll-leveling and landmark paths).
- **Model.** `timm/convnext_tiny.fb_in22k_ft_in1k` (ImageNet-22k pre-trained, ImageNet-1k
  fine-tuned) at revision `bc48a87`, licensed Apache-2.0 on its model card; the upstream
  facebookresearch/ConvNeXt weights are MIT. The MLP's linear layers were reshaped into
  equivalent 1x1 convolutions (`conv_mlp=True`, features unchanged to 3e-5), because OpenCV's
  ONNX importer cannot load ConvNeXt's channels-last linear layers. A single linear head
  predicts the nine buckets; gender and race auxiliary heads were not tried (no GPU budget was
  left for the comparison).
- **Recipe.** Cross-entropy on ordinal soft labels (a Gaussian over bucket indices, sigma 0.5
  buckets, so the neighbouring buckets get about 0.1 each); AdamW, peak learning rate 2e-4
  (5x for the head), weight decay 0.05, one warm-up epoch then cosine decay, drop path 0.1,
  batch 128, mixed precision, 12 epochs; on-GPU jitter (flip, rotation up to 10 degrees, scale
  0.9-1.1, shift up to 5%, contrast, brightness, saturation, 5% grayscale, 15% down-up resampling
  to 48-112 px). The exponential moving average of the weights (decay 0.9995) at epoch 11 was
  the best on the held-back slice: 61.0% bucket accuracy, 96.6% within one bucket.
- **Export.** ONNX, opset 17, input `1x3x224x224`, output nine logits. OpenCV 4.14's
  `cv2.dnn` loads it and matches PyTorch to 8e-6 on 64 held-back faces (same argmax on all);
  it takes about 0.2 s per face on two CPU cores. The weights were kept out of the repository
  and off the model host, since the model was not adopted.
- **Compute.** Kaggle, one T4 for 93 minutes of training; about 3 GPU-hours in total including
  a first attempt that was killed after 49 minutes (most likely out of host memory) and two short
  timing runs (DataParallel over two T4s
  ran at 6-10 s per step against 0.55 s on one GPU).

## Reproduce

The kernels live in `tools/kaggle/age_model/` (each pins the repository commit it ran):
`prep/` records the crops (CPU), `crops/` rebuilds them (CPU), and the top level trains (GPU).
The training kernel is public at
<https://www.kaggle.com/code/kaitlynbassford/face-age-convnext-train>; the run reported here is
its **version 5** (open the version history, then the Output tab, for the ONNX and the
selection metrics). Its latest version is a publish-only save with no output, so
`kaggle kernels output` returns files only after you push and run your own copy.

```bash
kaggle kernels push -p tools/kaggle/age_model/prep     # then crops/, then the training kernel
kaggle kernels output kaitlynbassford/face-age-convnext-train -p data/kaggle_age
FACE_ANALYZER_MODEL_DIR=$PWD/models .venv-eval/bin/python tools/eval_heldout.py --n 2000 \
    --extra-age convnext=data/kaggle_age/age_convnext_fairface.onnx
kaggle datasets download moritzm00/utkface-cropped -p data/utkface --unzip
FACE_ANALYZER_MODEL_DIR=$PWD/models .venv-eval/bin/python tools/eval_utkface.py --n 1500 \
    --extra-age convnext=data/kaggle_age/age_convnext_fairface.onnx
```

`--extra-age` runs a model that is not an app backend on the same detections and aligned faces
as the cached run (the JSONs' `crop_mismatches` are zero) and caches it separately, so the other
backends are not re-run. The UTKFace pass of the four existing age backends took 42 minutes on
two cores. UTKFace (Zhang, Song and Qi, CVPR 2017) is for non-commercial research only; it is
used here for evaluation alone, kept in the untracked `data/utkface/`, and not redistributed.
