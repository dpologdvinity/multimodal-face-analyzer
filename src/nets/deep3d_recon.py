# Vendored from sicxu/Deep3DFaceRecon_pytorch (MIT license): models/networks.py's
# ReconNetWrapper (ResNet50 + per-coefficient 1x1-conv heads), models/bfm.py's
# ParametricFaceModel (BFM coefficient -> mesh math), and util/preprocess.py's face
# alignment (POS + resize_n_crop_img), reproduced faithfully from the upstream source so a
# user-supplied checkpoint/BFM file loads and behaves the same way it would in the original
# repo. Two files this module needs are NOT bundled with this app and cannot be auto-fetched:
#
#   models/BFM/BFM_model_front.mat  -- derived from the Basel Face Model (BFM09), which is
#       under Basel University's own non-commercial research license and requires
#       registering on their site (https://faces.dmi.unibas.ch/bfm/) to obtain. There is no
#       redistributable/direct-download URL for this file.
#   models/deep3d_recon_resnet50.pth -- the fine-tuned coefficient-regression checkpoint,
#       distributed only via a Google Drive folder (upstream repo's own README), which has
#       no scriptable/direct download URL either.
#
# models/BFM/similarity_Lm3D_all.mat IS bundled -- it's a small (~1KB) landmark alignment
# template from the same MIT-licensed upstream repo, not derived from BFM09 itself.
#
# UNVERIFIED AGAINST A REAL CHECKPOINT: neither gated file could be obtained this session
# (see README), so this code has not been exercised end-to-end against the actual upstream
# checkpoint/BFM data -- only against its own math in isolation. It's a faithful line-for-line
# port of the published source, not a guess, but treat it as unverified until tested with the
# real files.
#
# Also note: this repo has no face-landmark detector of its own for the 5-point alignment
# step upstream's own pipeline expects (they use an external MTCNN-based tool, not bundled
# here either). This module instead derives 5-point landmarks from this app's existing
# MediaPipe FaceLandmarker (see landmarks_5pt_from_mediapipe) -- an approximation of
# upstream's own landmark source, not a re-implementation of it.
from __future__ import annotations

import numpy as np
from scipy.io import loadmat

try:
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50
    TORCHVISION_SUPPORTED = True
except ImportError:
    TORCHVISION_SUPPORTED = False

DEEP3D_TARGET_SIZE = 224.0
DEEP3D_RESCALE_FACTOR = 102.0
DEEP3D_COEFF_DIM = 257

# MediaPipe FaceLandmarker index -> Deep3DFaceRecon's 5-point convention (nose, left eye,
# right eye, left mouth corner, right mouth corner -- must match load_lm3d_template's order).
# Approximate correspondence, not upstream's own landmark source (see module docstring).
_MEDIAPIPE_NOSE_TIP = 1
_MEDIAPIPE_LEFT_EYE = (33, 133)
_MEDIAPIPE_RIGHT_EYE = (362, 263)
_MEDIAPIPE_MOUTH_LEFT = 61
_MEDIAPIPE_MOUTH_RIGHT = 291


def landmarks_5pt_from_mediapipe(points_normalized: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """Reduce MediaPipe FaceLandmarker's 468 normalized (x, y) points to the 5-point
    [nose, left_eye, right_eye, left_mouth, right_mouth] pixel-coordinate array this module's
    alignment expects, with the y-axis flipped to match align_img's convention (upstream's
    own test.py does the same flip on its landmark files: `lm[:, -1] = H - 1 - lm[:, -1]`)."""
    def px(idx: int) -> np.ndarray:
        x, y = points_normalized[idx]
        return np.array([x * width, height - 1 - y * height])

    def px_mean(indices: tuple[int, int]) -> np.ndarray:
        return np.mean([px(i) for i in indices], axis=0)

    return np.stack([
        px(_MEDIAPIPE_NOSE_TIP),
        px_mean(_MEDIAPIPE_LEFT_EYE),
        px_mean(_MEDIAPIPE_RIGHT_EYE),
        px(_MEDIAPIPE_MOUTH_LEFT),
        px(_MEDIAPIPE_MOUTH_RIGHT),
    ]).astype(np.float32)


def load_lm3d_template(bfm_folder: str) -> np.ndarray:
    """Vendored from util/load_mats.py's load_lm3d: reduce the bundled 68-point 3D landmark
    template to the same 5-point [nose, left_eye, right_eye, mouth1, mouth2] layout."""
    lm3d = loadmat(f"{bfm_folder}/similarity_Lm3D_all.mat")["lm"]
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    return np.stack([
        lm3d[lm_idx[0], :],
        np.mean(lm3d[lm_idx[[1, 2]], :], 0),
        np.mean(lm3d[lm_idx[[3, 4]], :], 0),
        lm3d[lm_idx[5], :],
        lm3d[lm_idx[6], :],
    ], axis=0)


def _pos(xp: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Vendored verbatim from util/preprocess.py's POS: least-squares similarity transform
    (translation + uniform scale) mapping 3D template points x onto 2D landmarks xp."""
    npts = xp.shape[1]
    A = np.zeros([2 * npts, 8])
    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1
    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1
    b = np.reshape(xp.transpose(), [2 * npts, 1])
    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    r1, r2 = k[0:3], k[4:7]
    s_tx, s_ty = k[3], k[7]
    s = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2
    t = np.array([s_tx.item(), s_ty.item()])
    return t, float(s)


def align_face_for_recon(face_bgr: np.ndarray, landmarks_5pt: np.ndarray, lm3d_template: np.ndarray) -> np.ndarray:
    """Vendored from util/preprocess.py's align_img + resize_n_crop_img, adapted to plain
    cv2/numpy (no PIL) since this app doesn't otherwise depend on it. Returns a
    DEEP3D_TARGET_SIZE x DEEP3D_TARGET_SIZE BGR image, aligned/scaled the same way upstream's
    own pipeline prepares its ResNet50 input."""
    import cv2

    h0, w0 = face_bgr.shape[:2]
    t, s = _pos(landmarks_5pt.transpose(), lm3d_template.transpose())
    s = DEEP3D_RESCALE_FACTOR / s

    w, h = int(w0 * s), int(h0 * s)
    resized = cv2.resize(face_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    target = DEEP3D_TARGET_SIZE
    left = int(w / 2 - target / 2 + float((t[0] - w0 / 2) * s))
    up = int(h / 2 - target / 2 + float((h0 / 2 - t[1]) * s))
    right, below = left + int(target), up + int(target)

    # Upstream crops directly (assumes the aligned face always lands in-bounds for their
    # curated test sets); pad first here since an arbitrary crop from this app's own face
    # detector is more likely to land partially outside the resized frame.
    pad = int(target)
    padded = cv2.copyMakeBorder(resized, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    left, up, right, below = left + pad, up + pad, right + pad, below + pad
    return padded[up:below, left:right]


class ReconNetWrapper(nn.Module if TORCHVISION_SUPPORTED else object):
    """Vendored from models/networks.py's ReconNetWrapper (use_last_fc=False path): a
    torchvision ResNet50 backbone (up to its final avgpool, no fc layer) followed by 7
    separate 1x1-conv heads, one per coefficient group. Output is the same 257-d layout as
    upstream's split_coeff: id[0:80], exp[80:144], tex[144:224], angle[224:227],
    gamma[227:254], trans[254:257]."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # drop avgpool + fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.heads = nn.ModuleList([
            nn.Conv2d(2048, 80, kernel_size=1, bias=True),  # id
            nn.Conv2d(2048, 64, kernel_size=1, bias=True),  # exp
            nn.Conv2d(2048, 80, kernel_size=1, bias=True),  # tex
            nn.Conv2d(2048, 3, kernel_size=1, bias=True),   # angle
            nn.Conv2d(2048, 27, kernel_size=1, bias=True),  # gamma (SH lighting)
            nn.Conv2d(2048, 2, kernel_size=1, bias=True),   # tx, ty
            nn.Conv2d(2048, 1, kernel_size=1, bias=True),   # tz
        ])

    def forward(self, x):
        features = self.avgpool(self.backbone(x))
        return torch.cat([head(features) for head in self.heads], dim=1).flatten(1)


def build_deep3d_recon_model(weights_path: str):
    model = ReconNetWrapper()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class ParametricFaceModel:
    """Vendored from models/bfm.py's ParametricFaceModel: BFM coefficient -> 3D mesh math
    (shape/texture/normals/lighting/pose), numpy in, numpy out (upstream is batched/torch;
    this app only ever reconstructs one face at a time, so the batch dimension is dropped).
    Needs models/BFM/BFM_model_front.mat -- not bundled, see module docstring."""

    def __init__(self, bfm_model_path: str, camera_distance: float = 10.0, focal: float = 1015.0, center: float = 112.0) -> None:
        model = loadmat(bfm_model_path)
        self.mean_shape = model["meanshape"].astype(np.float32).reshape(-1, 3)
        self.mean_shape -= self.mean_shape.mean(axis=0, keepdims=True)
        self.id_base = model["idBase"].astype(np.float32)
        self.exp_base = model["exBase"].astype(np.float32)
        self.mean_tex = model["meantex"].astype(np.float32)
        self.tex_base = model["texBase"].astype(np.float32)
        self.point_buf = model["point_buf"].astype(np.int64) - 1
        self.face_buf = model["tri"].astype(np.int64) - 1
        self.camera_distance = camera_distance
        self.persc_proj = np.array([focal, 0, center, 0, focal, center, 0, 0, 1], dtype=np.float32).reshape(3, 3).T
        self.init_lit = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self._sh_a = [np.pi, 2 * np.pi / np.sqrt(3.0), 2 * np.pi / np.sqrt(8.0)]
        self._sh_c = [1 / np.sqrt(4 * np.pi), np.sqrt(3.0) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)]

    def split_coeff(self, coeffs: np.ndarray) -> dict:
        return {
            "id": coeffs[:80], "exp": coeffs[80:144], "tex": coeffs[144:224],
            "angle": coeffs[224:227], "gamma": coeffs[227:254], "trans": coeffs[254:257],
        }

    def compute_shape(self, id_coeff: np.ndarray, exp_coeff: np.ndarray) -> np.ndarray:
        flat = (self.id_base @ id_coeff) + (self.exp_base @ exp_coeff) + self.mean_shape.reshape(-1)
        return flat.reshape(-1, 3)

    def compute_texture(self, tex_coeff: np.ndarray) -> np.ndarray:
        flat = (self.tex_base @ tex_coeff) + self.mean_tex.reshape(-1)
        return (flat.reshape(-1, 3) / 255.0).clip(0, 1)

    def compute_norm(self, face_shape: np.ndarray) -> np.ndarray:
        v1, v2, v3 = face_shape[self.face_buf[:, 0]], face_shape[self.face_buf[:, 1]], face_shape[self.face_buf[:, 2]]
        face_norm = np.cross(v1 - v2, v2 - v3)
        face_norm /= np.linalg.norm(face_norm, axis=-1, keepdims=True).clip(min=1e-8)
        face_norm = np.vstack([face_norm, np.zeros((1, 3), dtype=face_norm.dtype)])
        vertex_norm = face_norm[self.point_buf].sum(axis=1)
        return vertex_norm / np.linalg.norm(vertex_norm, axis=-1, keepdims=True).clip(min=1e-8)

    def compute_color(self, face_texture: np.ndarray, face_norm: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        a, c = self._sh_a, self._sh_c
        gamma = gamma.reshape(3, 9) + self.init_lit.reshape(1, 9)
        gamma = gamma.T  # (9, 3)
        nx, ny, nz = face_norm[:, 0:1], face_norm[:, 1:2], face_norm[:, 2:3]
        Y = np.concatenate([
            a[0] * c[0] * np.ones_like(nx), -a[1] * c[1] * ny, a[1] * c[1] * nz, -a[1] * c[1] * nx,
            a[2] * c[2] * nx * ny, -a[2] * c[2] * ny * nz, 0.5 * a[2] * c[2] / np.sqrt(3.0) * (3 * nz ** 2 - 1),
            -a[2] * c[2] * nx * nz, 0.5 * a[2] * c[2] * (nx ** 2 - ny ** 2),
        ], axis=-1)  # (N, 9)
        rgb = Y @ gamma  # (N, 3)
        return (rgb * face_texture).clip(0, 1)

    def compute_rotation(self, angles: np.ndarray) -> np.ndarray:
        x, y, z = angles
        rot_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        rot_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rot_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return (rot_z @ rot_y @ rot_x).T.astype(np.float32)

    def reconstruct(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (vertices, faces, per-vertex RGB colors in [0, 1]) for the 257-d
        coefficient vector predicted by ReconNetWrapper."""
        c = self.split_coeff(coeffs)
        face_shape = self.compute_shape(c["id"], c["exp"])
        rotation = self.compute_rotation(c["angle"])
        face_shape = face_shape @ rotation + c["trans"]
        face_shape[:, 2] = self.camera_distance - face_shape[:, 2]  # to_camera

        face_texture = self.compute_texture(c["tex"])
        face_norm = self.compute_norm(self.compute_shape(c["id"], c["exp"])) @ rotation
        face_color = self.compute_color(face_texture, face_norm, c["gamma"])
        return face_shape, self.face_buf, face_color


def reconstruct_face_3d(recon_net, bfm_model: ParametricFaceModel, face_bgr: np.ndarray, landmarks_5pt: np.ndarray, lm3d_template: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end: align -> ResNet50 coefficient regression -> BFM mesh reconstruction.
    Returns (vertices, faces, per-vertex RGB colors in [0, 1])."""
    import cv2

    aligned_bgr = align_face_for_recon(face_bgr, landmarks_5pt, lm3d_template)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(aligned_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        coeffs = recon_net(tensor)[0].numpy()
    return bfm_model.reconstruct(coeffs)


def mesh_to_obj_str(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> str:
    """Wavefront .obj with per-vertex color (the widely-supported `v x y z r g b` extension --
    MeshLab, Blender's importer, and most online viewers read this)."""
    lines = [f"v {x:.5f} {y:.5f} {z:.5f} {r:.5f} {g:.5f} {b:.5f}" for (x, y, z), (r, g, b) in zip(vertices, colors)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def save_obj_mesh(path: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(mesh_to_obj_str(vertices, faces, colors))
