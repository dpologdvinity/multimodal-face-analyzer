# Model licenses

The code in this repository is MIT-licensed (see [LICENSE](LICENSE)), except the vendored model
architectures in `src/face_analyzer/nets/`, which keep their upstream licenses (listed per model
below; note the CC BY-NC-SA 4.0 BlurPool code in `src/face_analyzer/nets/face_reaging_model.py`).
The model weights are third-party files under their own terms. Apart from the required SSD
detector, none of them are stored in this repository's git tree any more (older commits kept them
in git-lfs). `tools/fetch_models.py`
downloads them on demand, as recorded in [models/manifest.json](models/manifest.json), which
carries the same license and source for every file. "Obtained from" below says where each file now
comes from:

- **HF mirror:** the license clearly allows redistribution (MIT, Apache-2.0, BSD, CC BY), so the
  file is mirrored in [kaitlynbassford/face-analyzer-weights](https://huggingface.co/kaitlynbassford/face-analyzer-weights),
  whose model card carries the attributions and license notices.
- **Upstream:** the license is restrictive or not stated, so the file is not re-hosted. The fetch
  script downloads it from the original project's own release, file, or site.
- **Bring your own:** no direct download exists, so the script prints where to get the file.
- **In git:** a few small, permissively licensed configs stay in git so a fresh clone starts
  without downloads. The required SSD detector weights also stay in git, although no license is
  stated for them, because the app cannot start without a face detector.

Downloading a file from any of these sources is not a grant of any right the upstream authors did
not give. Several are non-commercial or have no stated license.

Entries were compiled from the linked upstream sources in October 2026. "Not stated upstream"
means the linked source has no license file or license statement covering that file; it does
not mean the file is unrestricted.

## Model files

| Feature | Model key | File(s) | Upstream source (URL) | License | Redistribution | Obtained from |
| ------- | --------- | ------- | --------------------- | ------- | -------------- | ------------- |
| Face detection | `ssd` | `opencv_face_detector_uint8.pb`, `opencv_face_detector.pbtxt` | [opencv/opencv samples/dnn/face_detector](https://github.com/opencv/opencv/tree/4.x/samples/dnn/face_detector), weights from [opencv_3rdparty](https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20180220_uint8) | OpenCV repo: Apache-2.0. Weight file: not stated upstream | Unclear for the weights | In git (required detector; see note below). Upstream copy: GitHub raw, opencv_3rdparty |
| Face detection | `yolo` | `yolov8n_face.onnx` | [yakhyo/yolov8-face-onnx-inference](https://github.com/yakhyo/yolov8-face-onnx-inference) | Not stated upstream (README license badge points at a LICENSE file that does not exist). Credits [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face) (GPL-3.0) and [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Unclear; upstream lineage is copyleft | Upstream: GitHub release, yakhyo/yolov8-face-onnx-inference |
| Face detection | `scrfd` | `scrfd_2.5g_bnkps.onnx` | [deepinsight/insightface](https://github.com/deepinsight/insightface#license) | Code MIT; pretrained models "available for non-commercial research purposes only" | Non-commercial research only | Upstream: GitHub release, insightface `buffalo_m.zip` (`det_2.5g.onnx`) |
| Face detection | `retinaface` | `retinaface_mobilenet0.25.onnx` | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), ONNX re-export [amd/retinaface](https://huggingface.co/amd/retinaface) | MIT (repo); Apache-2.0 (model card) | Permitted with notice | HF mirror |
| Age | `caffe` | `age_net.caffemodel`, `age_deploy.prototxt` | [GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning) | Not stated upstream | Unclear | Upstream: GitHub raw, GilLevi/AgeGenderDeepLearning |
| Gender | `caffe` | `gender_net.caffemodel`, `gender_deploy.prototxt` | [GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning) | Not stated upstream | Unclear | Upstream: GitHub raw, GilLevi/AgeGenderDeepLearning |
| Age, gender, race | `fairface` | `fairface_7class.onnx` | [dchen236/FairFace](https://github.com/dchen236/FairFace) | Upstream README: "License: CC BY 4.0". The ONNX conversion's source is not recorded in this repo, so its own terms are unverified | Attribution required (CC BY 4.0) | HF mirror (attribution on the model card) |
| Age | `dex` | `dex_age.caffemodel`, `dex_age.prototxt` | [IMDB-WIKI / DEX (ETH Zurich)](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) | Dataset: "academic research purpose only". Pretrained models: not stated upstream | Treat as research use only | Upstream: data.vision.ee.ethz.ch |
| Age, gender | `mivolo` | `mivolo_v2.safetensors`, `mivolo_v2_config.json` | [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO), [iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2) | Apache-2.0 (repo and model card; vendored code license in `src/face_analyzer/nets/mivolo/LICENSE_MIVOLO`) | Permitted with notice | HF mirror; the config is also in git |
| Gender | `deepface` | `deepface_gender.h5` | [serengil/deepface_models](https://github.com/serengil/deepface_models) (`gender_model_weights.h5`) | Repo MIT. Fine-tuned from VGG-Face, whose authors allow [non-commercial research use only](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) (CC BY-NC 4.0) | Non-commercial (VGG-Face base) | Upstream: GitHub release, serengil/deepface_models |
| Race | `deepface` | `deepface_race.h5` | [serengil/deepface_models](https://github.com/serengil/deepface_models) (`race_model_single_batch.h5`) | Same as deepface gender | Non-commercial (VGG-Face base) | Upstream: GitHub release, serengil/deepface_models |
| Recognition, identity search | `vggface` | `deepface_vgg.h5` | [serengil/deepface_models](https://github.com/serengil/deepface_models) (`vgg_face_weights.h5`) | Repo MIT. Original VGG-Face weights: [non-commercial research use](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) (CC BY-NC 4.0) | Non-commercial (VGG-Face) | Upstream: GitHub release, serengil/deepface_models |
| Emotion | `dan` | `dan_affecnet7.pth` | [yaoing/DAN](https://github.com/yaoing/DAN) | Code MIT. No separate weight license. Trained on AffectNet, whose [license](http://mohammadmahoor.com/wp-content/uploads/2023/03/AffectNet-Agreement-v2-30Mar2023.pdf) is non-commercial research and education only | Unclear; treat as non-commercial | Bring your own (Google Drive) |
| Emotion | `hsemotion` | `hsemotion_enet_b0_8_best_vgaf.onnx` | [HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition) (`models/affectnet_emotions/onnx`) | Code Apache-2.0. No separate weight license. Fine-tuned on AffectNet (non-commercial dataset license, as above) | Unclear; treat as non-commercial | Upstream: GitHub raw, HSE-asavchenko/face-emotion-recognition |
| Emotion | `ferplus` | `emotion_ferplus.onnx` | [onnx/models emotion_ferplus](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) | MIT (model README) | Permitted with notice | HF mirror |
| Emotion | `mini_xception` | `mini_xception_fer.h5` | [oarriaga/face_classification](https://github.com/oarriaga/face_classification) | MIT | Permitted with notice | HF mirror |
| Face landmarks, gaze, liveness | `mediapipe` | `face_landmarker.task` | [Face Mesh V2 model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf) | Apache-2.0 | Permitted with notice | HF mirror |
| Hand landmarks | `mediapipe` | `hand_landmarker.task` | [Hand Tracking model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf) | Apache-2.0 | Permitted with notice | HF mirror |
| Glasses | `mobilenet` | `glasses_detector.onnx` | [Sorour190/Glasses-Detector](https://github.com/Sorour190/Glasses-Detector) | Not stated upstream (no LICENSE file) | Unclear | Upstream: GitHub raw, Sorour190/Glasses-Detector |
| Mask | `mobilenetv2` | `mask_detector.h5` | [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) | MIT | Permitted with notice | HF mirror |
| Colorization | `eccv16` | `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, `pts_in_hull.npy` | [richzhang/colorization](https://github.com/richzhang/colorization) | BSD-2-Clause | Permitted with notice | HF mirror; the prototxt and `.npy` are also in git |
| Eye color, roll alignment | `colorimetric` | `haarcascade_eye.xml` | [opencv/opencv data/haarcascades](https://github.com/opencv/opencv/tree/4.x/data/haarcascades) | Intel License Agreement (BSD-style, in the file header) | Permitted with notice | In git; also on the HF mirror |
| 3D reconstruction | `deep3d` | `deep3d_recon_resnet50.pth`, `BFM/similarity_Lm3D_all.mat` | [sicxu/Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) | Repo MIT. The checkpoint (Google Drive release) has no separate license statement | Unclear for the checkpoint | Checkpoint: bring your own (Google Drive). `.mat` template: in git and on the HF mirror |
| Age progression | `franunet` | `face_reaging_unet.pth` | [timroelofs123/face_reaging](https://github.com/timroelofs123/face_reaging), [timroelofs123/face_re-aging](https://huggingface.co/timroelofs123/face_re-aging) | Code and model card MIT. The network's BlurPool layers are vendored from [adobe/antialiased-cnns](https://github.com/adobe/antialiased-cnns), CC BY-NC-SA 4.0 | Non-commercial, share-alike (BlurPool) | Upstream: huggingface.co, timroelofs123/face_re-aging |

## Required but not downloadable

| Feature | File | Upstream source (URL) | License | Redistribution | Obtained from |
| ------- | ---- | --------------------- | ------- | -------------- | ------------- |
| 3D reconstruction | `models/BFM/BFM_model_front.mat` (converted from BFM09) | [Basel Face Model](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) | Registration required; "internal, non-commercial research, evaluation or testing purposes only" | Prohibited; each user must obtain it directly | Bring your own (registration) |

`lbph` recognition, hair color, eye color, eigenfaces, and the image tools have no weight files.
