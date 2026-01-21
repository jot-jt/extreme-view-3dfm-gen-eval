import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from interfaces.vggt import *

def infer_mv_pointclouds(filelist: List[str], model: VGGT, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

    point_maps = predictions['world_points'].squeeze(0).cpu().numpy()
    return point_maps