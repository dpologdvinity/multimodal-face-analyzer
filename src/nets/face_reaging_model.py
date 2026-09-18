# Vendored from timroelofs123/face_reaging (MIT license), a U-Net-based re-aging network
# reproducing Disney Research's FRAN paper: model/models.py's UNet/DownLayer/UpLayer,
# reproduced faithfully so the upstream pretrained checkpoint (best_unet_model.pth, loadable
# directly with torch.load, no training needed -- see
# https://huggingface.co/timroelofs123/face_re-aging) loads and runs the same way it would in
# the original repo.
#
# LICENSE CAVEAT (same treatment as insightface's gender/age model already in this repo):
#
#   1. BlurPool (below, used inside DownLayer/UpLayer) is vendored from Adobe's
#      antialiased-cnns (github.com/adobe/antialiased-cnns), which is licensed under
#      Creative Commons Attribution-NonCommercial-ShareAlike 4.0 -- NON-COMMERCIAL ONLY,
#      and share-alike (any redistribution/adaptation must carry the same license). Unlike
#      most of this repo's NC caveats, this one is NOT just about training-data provenance --
#      BlurPool is a required inference-time component of the network architecture itself.
#   2. The pretrained weights' training data (FFHQ, re-aged via SAM/StyleGAN2) traces back
#      through NVIDIA's non-commercial-research-licensed SAM, even though upstream's own
#      code/weights carry no explicit restriction.
#
# Net effect: this whole feature (src/nets/face_reaging_model.py, the age_progression_nets
# backend in src/inference.py, and the AGE PROGRESSION button in src/app.py) is
# non-commercial/research use only. See README.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FACE_REAGING_INPUT_SIZE = 512


class BlurPool(nn.Module):
    """Vendored from adobe/antialiased-cnns's blurpool.py (CC BY-NC-SA 4.0 -- see module
    docstring). Anti-aliased strided downsampling: blur with a fixed binomial filter, then
    subsample, instead of naive strided conv/pool."""

    def __init__(self, channels: int, pad_type: str = "reflect", filt_size: int = 4, stride: int = 2):
        super().__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        pad = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))] * 2
        self.pad_sizes = pad

        filters_by_size = {
            1: [1.], 2: [1., 1.], 3: [1., 2., 1.], 4: [1., 3., 3., 1.],
            5: [1., 4., 6., 4., 1.], 6: [1., 5., 10., 10., 5., 1.], 7: [1., 6., 15., 20., 15., 6., 1.],
        }
        a = np.array(filters_by_size[filt_size])
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((channels, 1, 1, 1)))
        self.pad = nn.ReflectionPad2d(self.pad_sizes) if pad_type in ("refl", "reflect") else nn.ZeroPad2d(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class DownLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            BlurPool(in_channels, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class UpLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.blur_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            BlurPool(out_channels, stride=1),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.blur_upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.layer(x)


class UNet(nn.Module):
    """5-channel input (RGB + source-age channel + target-age channel, each /100), 3-channel
    residual output added back onto the RGB input by the caller (see age_progress_face)."""

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024)
        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.init_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.final_conv(x)


def build_face_reaging_model(weight_path: str) -> UNet:
    net = UNet()
    state_dict = torch.load(weight_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    return net


def age_progress_face(net: UNet, face_rgb: np.ndarray, source_age: float, target_age: float) -> np.ndarray:
    """Re-age one face crop (RGB, any size, uint8 0-255). Resizes to the network's native
    512x512 training resolution, runs a single forward pass (no sliding-window tiling --
    unlike upstream's own full-photo pipeline, this app already hands in a tight face crop
    rather than a large image needing multiple 512x512 windows stitched together), adds the
    predicted residual back onto the resized input, then resizes the result back to the
    input's original size. Returns an RGB uint8 array the same size as face_rgb."""
    orig_h, orig_w = face_rgb.shape[:2]
    size = FACE_REAGING_INPUT_SIZE

    image = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    resized = F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]

    source_channel = torch.full((1, size, size), source_age / 100.0, dtype=torch.float32)
    target_channel = torch.full((1, size, size), target_age / 100.0, dtype=torch.float32)
    input_tensor = torch.cat([resized, source_channel, target_channel], dim=0).unsqueeze(0)

    with torch.no_grad():
        residual = net(input_tensor)[0]

    aged = torch.clamp(resized + residual, 0.0, 1.0)
    aged_resized_back = F.interpolate(aged.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False, antialias=True)[0]
    aged_uint8 = (aged_resized_back.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return aged_uint8
