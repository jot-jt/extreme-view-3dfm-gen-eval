import os
import os.path as osp
import json
import pathlib
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils.read_write_model import read_model_cameras_images, qvec2rotmat     # noqa: E402
from datasets.utils.cropping import resize_image_depth_and_intrinsic          # noqa: E402
from models.vggt.utils.geometry import unproject_depth_map_to_point_map       # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

to_tensor = tvf.ToTensor()


class MegaUnScene:
    """Dataset for MegaUnScene."""

    def __init__(
        self,
        megaunscene_root: str,
        scale_json_path: str,
        eval_images_json_path: str,
        load_img_size: int = 518,
        processed_cache_dir: str = "data/dataset_cache/megaunscene_processed",
    ):
        self.megaunscene_root = megaunscene_root
        self.scale_json_path = scale_json_path
        self.eval_images_json_path = eval_images_json_path
        self.load_img_size = load_img_size
        self.processed_cache_dir = processed_cache_dir
        pathlib.Path(processed_cache_dir).mkdir(parents=True, exist_ok=True)

        # Load scale measurements JSON
        with open(scale_json_path, "r") as f:
            self.consolidated_data = json.load(f)

        # Load evaluation JSON to get the sampled images
        with open(eval_images_json_path, "r") as f:
            eval_items = json.load(f)

        # Build mapping from (scene, recon_id) to sampled image paths
        eval_seq_to_relpaths: Dict[Tuple[str, str], List[str]] = {}
        for item in eval_items:
            scene = item.get("scene")
            recon_id = str(item.get("id"))
            samp = item.get("sampled_images", [])
            relpaths = []
            for s in samp:
                rel = s.get("image_name") or s.get("image_filename")
                if rel is None:
                    continue
                relpaths.append(rel)
            key = (scene, recon_id)
            eval_seq_to_relpaths[key] = relpaths

        # Filter consolidated data to only include sequences in eval set
        self.sequence_list: List[Tuple[str, str]] = []
        self.seq_to_relpaths: Dict[Tuple[str, str], List[str]] = {}
        self.seq_to_metadata: Dict[Tuple[str, str], dict] = {}

        for seq_key, metadata in self.consolidated_data.items():
            scene = metadata.get("scene")
            recon_id = str(metadata.get("recon_id"))
            key = (scene, recon_id)
            
            # Only include if in eval set
            if key in eval_seq_to_relpaths:
                self.sequence_list.append(key)
                self.seq_to_relpaths[key] = eval_seq_to_relpaths[key]
                self.seq_to_metadata[key] = metadata

        self._colmap_cache: Dict[Tuple[str, str], Tuple[dict, dict, Dict[str, int]]] = {}

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[Tuple[str, str]] = None) -> int:
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.seq_to_relpaths[sequence_name])

    def _load_colmap_for_seq(self, scene: str, recon_id: str):
        key = (scene, recon_id)
        if key in self._colmap_cache:
            return self._colmap_cache[key]
        sparse_recon_path = osp.join(self.megaunscene_root, "scenes", scene, recon_id, "sparse")
        cameras, images = read_model_cameras_images(sparse_recon_path)
        name_to_id = {img.name: img_id for img_id, img in images.items()}
        self._colmap_cache[key] = (cameras, images, name_to_id)
        return self._colmap_cache[key]

    @staticmethod
    def _resolve_depth_path(base_depth_dir: str, rel_image_path: str) -> Optional[str]:
        p = pathlib.Path(base_depth_dir) / rel_image_path
        candidates = [
            p.with_suffix(".npy"),
            pathlib.Path(str(p) + ".npy"),
            p.with_suffix(".npz"),
            pathlib.Path(str(p) + ".npz"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[Union[Tuple[str, str], str]] = None,
        ids: Union[List[int], np.ndarray, None] = None,
    ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        
        # Handle both string (from seq_id_map) and tuple sequence names
        if isinstance(sequence_name, str):
            # Convert "scene-recon_id" string to (scene, recon_id) tuple
            parts = sequence_name.rsplit("-", 1)
            if len(parts) == 2:
                scene_name, recon_id = parts
                sequence_name = (scene_name, recon_id)
            else:
                raise ValueError(f"Invalid sequence name format: {sequence_name}. Expected 'scene-recon_id'")
        else:
            scene_name, recon_id = sequence_name

        relpaths_all: List[str] = self.seq_to_relpaths[sequence_name]
        seq_len: int = len(relpaths_all)

        if ids is None:
            ids = list(range(seq_len))
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1
            ids = ids.tolist()

        model_cameras, model_images, name_to_id = self._load_colmap_for_seq(scene_name, recon_id)

        image_paths: List[str] = [""] * len(ids)
        images: List[torch.Tensor] = [None] * len(ids)
        depths: List[np.ndarray] = [None] * len(ids)
        extrinsics: np.ndarray = np.zeros((len(ids), 4, 4), dtype=np.float32)
        intrinsics: np.ndarray = np.zeros((len(ids), 3, 3), dtype=np.float32)

        image_root = osp.join(self.megaunscene_root, "scenes", scene_name, recon_id, "images")
        depth_root = osp.join(self.megaunscene_root, "scenes", scene_name, recon_id, "depth_maps")

        for k, local_id in enumerate(ids):
            rel = relpaths_all[local_id]
            img_path = osp.join(image_root, rel)
            depth_path = self._resolve_depth_path(depth_root, rel)
            if depth_path is None:
                raise FileNotFoundError(f"Depth not found for {rel} under {depth_root}")

            if rel not in name_to_id:
                basename = osp.basename(rel)
                matches = [iid for iid, im in model_images.items() if osp.basename(im.name) == basename]
                if len(matches) == 1:
                    colmap_image_id = matches[0]
                else:
                    raise KeyError(f"Image name '{rel}' not found in COLMAP model and basename match ambiguous: {matches}")
            else:
                colmap_image_id = name_to_id[rel]

            qvec = model_images[colmap_image_id].qvec
            tvec = model_images[colmap_image_id].tvec
            R = qvec2rotmat(qvec)
            extr = np.concatenate([R, tvec[:, None]], axis=1)  # 3x4

            cam_id = model_images[colmap_image_id].camera_id
            cam = model_cameras[cam_id]
            if cam.model != "PINHOLE":
                raise ValueError(f"Unsupported camera model: {cam.model}. Images should be undistorted.")
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            rgb_image = Image.open(img_path)
            width, height = rgb_image.size

            if depth_path.endswith('.npz'):
                arr = np.load(depth_path)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    for key in ('depth', 'arr_0'):
                        if key in arr:
                            depthmap = arr[key]
                            break
                    else:
                        first_key = list(arr.files)[0]
                        depthmap = arr[first_key]
                else:
                    depthmap = arr
            else:
                depthmap = np.load(depth_path)

            depthmap = depthmap.reshape(height, width)
            depthmap[~np.isfinite(depthmap)] = -1

            scene_cache_dir = osp.join(self.processed_cache_dir, scene_name, recon_id)
            os.makedirs(scene_cache_dir, exist_ok=True)
            cached_image_path = osp.join(scene_cache_dir, "images", rel)

            rgb_image, depthmap, K_resized = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=K,
                output_width=self.load_img_size,
                pad_to_square=True,
            )
            if not osp.exists(cached_image_path):
                os.makedirs(osp.dirname(cached_image_path), exist_ok=True)
                rgb_image.save(cached_image_path)

            image_paths[k] = cached_image_path
            images[k] = to_tensor(rgb_image)
            depths[k] = depthmap
            intrinsics[k] = K_resized
            extrinsics[k, :3] = extr

        depths_np = np.array(depths)
        
        # Get metadata including metric_scale_factor
        metadata = self.seq_to_metadata[sequence_name]
        metric_scale_factor = None
        if "measurement" in metadata and metadata["measurement"] is not None:
            metric_scale_factor = metadata["measurement"].get("metric_scale_factor")
        
        # Apply metric scale factor to depths and extrinsics
        if metric_scale_factor is not None and metric_scale_factor > 0:
            # Scale depths by metric_scale_factor
            depths_np = depths_np * metric_scale_factor
            # Scale translation component of extrinsics
            extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * metric_scale_factor
        
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths_np[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics[:, :3, :],
        )

        batch = {
            "seq_id": sequence_name,
            "scene_name": scene_name,
            "recon_id": recon_id,
            "seq_len": seq_len,
            "ind": torch.tensor(ids),
            "image_paths": image_paths,
            "images": torch.stack(images, dim=0),
            "pointclouds": pointclouds,
            "valid_mask": depths_np > 1e-4,
            "extrs": torch.tensor(extrinsics),
            "intrs": intrinsics,
            "metadata": metadata,
        }
        return batch