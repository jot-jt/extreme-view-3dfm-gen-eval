import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import List, Tuple
import time

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import WorldMirror
import sys
sys.path.append('models/worldmirror')
from src.utils.inference_utils import load_and_preprocess_images
from src.models.utils.geometry import closed_form_inverse_se3


def load_and_resize14(filelist: List[str], resize_to: int, device: str):
    images = load_and_preprocess_images(
        filelist, output_size=resize_to
    ).to(device)  # [1,N,3,H,W], in [0,1]

    ori_h, ori_w = images.shape[-2:]
    patch_h, patch_w = max(1, ori_h // 14), max(1, ori_w // 14)
    
    # Reshape [1, N, 3, H, W] -> [1*N, 3, H, W] for interpolation
    batch, num_views, channels, h, w = images.shape
    images_reshaped = images.reshape(batch * num_views, channels, h, w)
    
    # Interpolate to nearest multiple of 14
    images_reshaped = F.interpolate(
        images_reshaped, 
        (patch_h * 14, patch_w * 14), 
        mode="bilinear", 
        align_corners=False, 
        antialias=True
    )
    
    # Reshape back to [1, N, 3, H_14, W_14]
    images = images_reshaped.reshape(batch, num_views, channels, patch_h * 14, patch_w * 14)
    return images



def infer_monodepth(file: str, model: WorldMirror, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14([file], resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)

    inputs = {'img': images}
    cond_flags = [0, 0, 0]

    prior_data = {
        'camera_pose': None,      # Camera pose tensor [1, N, 4, 4]
        'depthmap': None,         # Depth map tensor [1, N, H, W]
        'camera_intrinsics': None # Camera intrinsics tensor [1, N, 3, 3]
    }

    for idx, (key, data) in enumerate(prior_data.items()):
        if data is not None:
            cond_flags[idx] = 1
            inputs[key] = data
    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(inputs, cond_flags)

    depth_preds = predictions["depth"][0]    # Z-depth in camera frame: [S, H, W, 1]
    return depth_preds[0, ..., 0].detach()  # returns (h_14, w_14) torch tensor


def infer_videodepth(filelist: List[str], model: WorldMirror, hydra_cfg: DictConfig):
    raise NotImplementedError


def infer_cameras_w2c(filelist: List[str], model: WorldMirror, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)

    inputs = {'img': images}
    cond_flags = [0, 0, 0]

    prior_data = {
        'camera_pose': None,      # Camera pose tensor [1, N, 4, 4]
        'depthmap': None,         # Depth map tensor [1, N, H, W]
        'camera_intrinsics': None # Camera intrinsics tensor [1, N, 3, 3]
    }
    
    for idx, (key, data) in enumerate(prior_data.items()):
        if data is not None:
            cond_flags[idx] = 1
            inputs[key] = data

    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(inputs, cond_flags)

    camera_poses = predictions["camera_poses"][0]  # Camera-to-world poses (OpenCV convention): [S, 4, 4]
    extrinsics = closed_form_inverse_se3(camera_poses)       # World-to-camera extrinsics: [S, 4, 4]
    return extrinsics.cpu(), None


def infer_cameras_c2w(filelist: List[str], model: WorldMirror, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)

    inputs = {'img': images}
    cond_flags = [0, 0, 0]

    prior_data = {
        'camera_pose': None,      # Camera pose tensor [1, N, 4, 4]
        'depthmap': None,         # Depth map tensor [1, N, H, W]
        'camera_intrinsics': None # Camera intrinsics tensor [1, N, 3, 3]
    }
    
    for idx, (key, data) in enumerate(prior_data.items()):
        if data is not None:
            cond_flags[idx] = 1
            inputs[key] = data

    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(inputs, cond_flags)

    camera_poses = predictions["camera_poses"][0]  # Camera-to-world poses (OpenCV convention): [S, 4, 4]
    return camera_poses.cpu(), None


def infer_mv_pointclouds(filelist: List[str], model: WorldMirror, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)

    inputs = {'img': images}
    cond_flags = [0, 0, 0]

    prior_data = {
        'camera_pose': None,      # Camera pose tensor [1, N, 4, 4]
        'depthmap': None,         # Depth map tensor [1, N, H, W]
        'camera_intrinsics': None # Camera intrinsics tensor [1, N, 3, 3]
    }

    for idx, (key, data) in enumerate(prior_data.items()):
        if data is not None:
            cond_flags[idx] = 1
            inputs[key] = data

    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(inputs, cond_flags)

    # Geometry outputs
    pts3d_preds = predictions["pts3d"][0]  # 3D point cloud in world coordinate: [S, H, W, 3]
    return pts3d_preds.cpu().numpy()