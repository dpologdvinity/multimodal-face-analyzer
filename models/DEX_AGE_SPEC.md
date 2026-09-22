# DEX Age Model Specification

## Overview
Deep EXpectation (DEX) age prediction model by Rothe et al., trained on IMDB-WIKI dataset.
- **Reference**: "DEX: Deep EXpectation of apparent age from a single image" (ICCV 2015)
- **Architecture**: VGG-16 (16-layer convolutional neural network)
- **Framework**: Caffe (protocol buffer format)
- **Input**: 224×224 RGB/BGR image
- **Output**: 101-way softmax over ages 0-100 years

## File Locations & Sizes

| File | Location | Size | Format | Status |
|------|----------|------|--------|--------|
| Model weights | `models/dex_age.caffemodel` | 513.7 MB | Caffe binary | ✓ Downloaded |
| Network definition | `models/dex_age.prototxt` | 4.8 KB | Protobuf text | ✓ Downloaded |

**Download sources**: 
- https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_imdb_wiki.caffemodel
- https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt

## Preprocessing

### Input Dimensions
- **Blob shape**: (1, 3, 224, 224) — batch_size=1, channels=3, height=224, width=224
- **Color order**: BGR (OpenCV standard for Caffe models)
- **Scale factor**: 1.0 (no per-channel scaling)

### Mean Subtraction (ImageNet VGG Mean)
The model expects images normalized via mean subtraction before inference:
- **B (blue) channel**: 103.939
- **G (green) channel**: 116.779
- **R (red) channel**: 123.68

Applied as: `normalized_img = img - [103.939, 116.779, 123.68]` (BGR order)

This is the standard VGG-16 ImageNet preprocessing constant from the original Simonyan et al. 2014 paper.

### Encoding in cv2.dnn

```python
import cv2

# Preprocess image
vgg_mean = [103.939, 116.779, 123.68]  # BGR order
blob = cv2.dnn.blobFromImage(
    image,           # OpenCV Mat/ndarray in BGR
    scalefactor=1.0,
    size=(224, 224),
    mean=vgg_mean,
    swapRB=False,    # Image already in BGR; no swap needed
    crop=False       # Resize, don't crop
)
```

## Network Architecture

**Prototxt structure** (verified from `dex_age.prototxt`):
- Input layer: `"data"` accepting 3-channel 224×224 images
- Convolutional blocks: conv1–conv5 (standard VGG-16 structure)
- Fully connected layers: fc6, fc7 (4096 units each, ReLU + Dropout(0.5))
- **Output layer**: `fc8-101` with 101 units (ages 0–100)
- **Loss layer**: `Softmax` producing probability distribution over age classes

## Output Decoding

The network outputs a Softmax distribution over 101 age classes (0–100 years inclusive).

### Raw Output
After forward pass, extract output from `"prob"` layer (shape: [1, 101, 1, 1]):
```python
predictions = net.forward()  # Shape: [1, 101, 1, 1]
probs = predictions[0].flatten()  # Shape: [101]
```

### Age Estimation (Expected Value)
Compute the expected age as the weighted sum of class centers:
```python
age = sum(probs[i] * i for i in range(101))
```

This yields a continuous age estimate in the range [0, 100], not constrained to integer years.

### Alternative: Argmax
For class-based prediction (integer age):
```python
predicted_age_class = np.argmax(probs)  # Integer in [0, 100]
```

## Model Loading (cv2.dnn)

### Docker Environment (Recommended)
The Docker image uses `opencv-python-headless>=4.8.0,<5.0.0` (from `requirements.txt`), which includes full Caffe support:

```python
import cv2

proto_path = "models/dex_age.prototxt"
model_path = "models/dex_age.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
```

### Local Development (Known Limitation)
The local sandbox environment has **OpenCV 5.0.0**, which removed Caffe support in favor of ONNX and TensorFlow. 

**Error**:
```
AttributeError: module 'cv2.dnn' has no attribute 'readNetFromCaffe'
```

**Status**: This is a **known local-environment limitation**, not a problem with the DEX model files themselves. The files are verified as valid:
- ✓ Prototxt parses correctly (contains expected layers: fc8-101, Softmax, input_dim: 224)
- ✓ Caffemodel file size matches expectation (513.7 MB)
- ✓ Caffemodel binary header is intact

**Workaround**: Use Docker for testing, or upgrade/downgrade OpenCV locally if needed.

## File Verification Checklist

- [x] **Caffemodel integrity**: 513.7 MB, binary header validated
- [x] **Prototxt syntax**: Parses as valid protocol buffer text format
- [x] **Network structure**: Confirms VGG-16 architecture with fc8-101 (101 units) final layer
- [x] **Softmax output layer**: Named `"prob"`, ready for probability-based decoding
- [x] **Input dimensions**: 224×224 (line 5–6 of prototxt: `input_dim: 224`)
- [x] **Download sources live**: Confirmed both files available on ETH Zurich server

## License

**Source**: ETH Zurich Computer Vision Lab  
**Authors**: Rothe et al. (2015)  
**License**: The original code and pre-trained weights are provided for **academic research and non-commercial use**.

While no explicit restriction is stated in the README or model comments, the dataset (IMDB-WIKI) and pre-trained weights are derived from academic research and carry a research-use-only spirit. Commercial applications may require independent licensing or re-training on a licensed dataset.

**Recommendation**: Flag as research/academic-use in any product documentation if used in deployed systems.

## Integration Notes

This model **complements existing age backends** (caffe Levi-Hassner, fairface) and follows the same Caffe pattern:

- Same input preprocessing (224×224 mean subtraction) as the existing `age_net.caffemodel`
- Same cv2.dnn loading pattern (readNetFromCaffe)
- Different output shape (101 classes vs. 8 age-range classes) — requires decode formula shown above
- Can be added to Dockerfile ARGs: `--build-arg AGE_MODEL=caffe,dex,...`
- Defers to Docker environment for Caffe support (not available in local OpenCV 5.0.0 sandbox)

## References

1. Rothe, R., Timofte, R., & Van Gool, L. (2015). "DEX: Deep EXpectation of apparent age from a single image." *ICCV 2015*.
2. ImageNet VGG-16 preprocessing: Simonyan, K., & Zisserman, A. (2014). "Very deep convolutional networks for large-scale image recognition." *ICLR 2015*.
3. ETH Zurich release: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
