import torch
import torch.nn as nn

from typing import Optional, Dict

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class Pi3(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            finetuned_exrot_ckpt_path: Optional[str] = None
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.pi3.models.pi3 import Pi3 as Pi3Model
            model = Pi3Model.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

        # Load/apply extreme rotation finetuned checkpoint if provided
        if finetuned_exrot_ckpt_path is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(finetuned_exrot_ckpt_path, map_location=device)
            state = ckpt.get("bias_state_dict", ckpt.get("model"))
            if state is not None:
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
        
        self.model = model

    def forward(self, images: torch.Tensor):
        return self.model(images)


class VGGT(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            finetuned_exrot_ckpt_path: Optional[str] = None
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.vggt.models.vggt import VGGT as VGGTModel
            model = VGGTModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        
        # Load/apply extreme rotation finetuned checkpoint if provided
        if finetuned_exrot_ckpt_path is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(finetuned_exrot_ckpt_path, map_location=device)
            state = ckpt.get("bias_state_dict", ckpt.get("model"))
            if state is not None:
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
        
        self.model = model

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images, query_points)


class WorldMirror(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            finetuned_exrot_ckpt_path: Optional[str] = None
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            import sys
            sys.path.append('models/worldmirror')

            try:  # for logging compatibility with WorldMirror's repo
                from training.utils.logger import rank_zero_only
                rank_zero_only.rank = 0
            except ImportError:
                print("Warning: Could not find training.utils.logger to set rank.")

            from src.models.models.worldmirror import WorldMirror as WorldMirrorModel
            model = WorldMirrorModel.from_pretrained(pretrained_model_name_or_path, enable_gs=False)
        else:
            raise NotImplementedError
        
        # Load/apply extreme rotation finetuned checkpoint if provided
        if finetuned_exrot_ckpt_path is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(finetuned_exrot_ckpt_path, map_location=device)
            state = ckpt.get("bias_state_dict", ckpt.get("model"))
            if state is not None:
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
        
        self.model = model

    def forward(self, inputs, cond_flags):
        return self.model(views=inputs, cond_flags=cond_flags)
    

class MoGe(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            ori_model: bool = True,
        ):
        super().__init__()

        if ori_model and pretrained_model_name_or_path is not None:
            from models.moge.model.v1 import MoGeModel
            self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        return self.model(image, num_tokens)
