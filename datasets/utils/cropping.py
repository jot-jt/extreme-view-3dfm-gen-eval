import cv2
import numpy as np

from PIL import Image
from typing import Tuple

try:
    lanczos = Image.Resampling.LANCZOS
    bicubic = Image.Resampling.BICUBIC
except AttributeError:
    lanczos = Image.LANCZOS
    bicubic = Image.BICUBIC

def resize_image(image: Image.Image, output_resolution: Tuple[int, int]) -> Image.Image:
    max_resize_scale = max(output_resolution[0] / image.size[0], output_resolution[1] / image.size[1])
    return image.resize(output_resolution, resample=lanczos if max_resize_scale < 1 else bicubic)

def resize_image_depth_and_intrinsic(
    image: Image.Image,
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    output_width: int,
    pixel_center: bool = True,
    pad_to_square: bool = False,
) ->  Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Resize the image and depth map to the specified output width while maintaining the aspect ratio.
    If pad_to_square is True, the image and depth map will be resized with the longer side to the output width, then padded to a square shape.
    """
    if len(depth_map.shape) != 2:
        raise ValueError(f"Depth map must be a 2D array, but found depthmap.shape = {depth_map.shape}")
    input_resolution = np.array(depth_map.shape[::-1], dtype=np.float32)  # (H, W) -> (W, H)
    is_taller = input_resolution[1] > input_resolution[0]
    
    if pad_to_square and is_taller:
        output_height = output_width
        output_resolution = np.array([round(input_resolution[0] * (output_height / input_resolution[1]) / 14) * 14, output_height])
    else:
        output_resolution = np.array([output_width, round(input_resolution[1] * (output_width / input_resolution[0]) / 14) * 14])

    image = resize_image(image, tuple(output_resolution))

    depth_map = cv2.resize(
        depth_map,
        output_resolution,
        interpolation = cv2.INTER_NEAREST,
    )

    intrinsic = np.copy(intrinsic)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    resize_scale = np.max(output_resolution / input_resolution)
    intrinsic[:2, :] = intrinsic[:2, :] * resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5
    
    # Pad to make the output square if the flag is set
    if pad_to_square:
        target_size = output_width
        
        if output_resolution[0] < target_size or output_resolution[1] < target_size:
            # Calculate padding
            pad_width = max(0, target_size - output_resolution[0])
            pad_height = max(0, target_size - output_resolution[1])
            
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            
            # Pad the image with black background
            padded_image = Image.new(image.mode, (target_size, target_size), (0, 0, 0))
            padded_image.paste(image, (pad_left, pad_top))
            
            # Pad the depth map
            padded_depth = np.full((target_size, target_size), -1, dtype=depth_map.dtype)
            padded_depth[pad_top:pad_top+output_resolution[1], pad_left:pad_left+output_resolution[0]] = depth_map
            
            # Adjust intrinsic matrix for padding
            intrinsic[0, 2] = intrinsic[0, 2] + pad_left
            intrinsic[1, 2] = intrinsic[1, 2] + pad_top
            
            image = padded_image
            depth_map = padded_depth
            
            # Update output_resolution to reflect the new size
            output_resolution = np.array([target_size, target_size])
    
    assert image.size == depth_map.shape[::-1], f"Image size {image.size} does not match depth map shape {depth_map.shape[::-1]}"
    return image, depth_map, intrinsic