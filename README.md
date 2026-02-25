# Extreme-View 3DFM Generalization Evaluations

<a href="https://arxiv.org/abs/2511.22686"><img src="https://img.shields.io/badge/arXiv-2511.22686-b31b1b" alt="arXiv"></a> &nbsp;
<a href="https://cornell-vailab.github.io/Ext-3DFMs/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a> &nbsp;
<a href="https://github.com/theREALevan/extreme-view-3dfm"><img src="https://img.shields.io/badge/Main_Code-black?logo=github" alt="Main Code Repository"></a> &nbsp;
<a href="https://github.com/jot-jt/extreme-view-3dfm-gen-eval"><img src="https://img.shields.io/badge/Generalization_Evaluations-black?logo=github" alt="Generalization Eval Repository"></a> &nbsp;
<a href="https://huggingface.co/datasets/cornell-vailab/megaunscene"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face"></a>

A unified evaluation framework for 3D reconstruction, modified for the paper ["Emergent Extreme-View Geometry in 3D Foundation Models"](https://cornell-vailab.github.io/Ext-3DFMs/) ("Ext-3DFMs" for short).

This is a fork of the evaluation framework used in [π³](https://arxiv.org/abs/2507.13347), specifically from [this GitHub repository](https://github.com/ZhouTimeMachine/recons_eval). This fork is  meant to help replicate evaluations on VGGT, π³, and WorldMirror, as well as the respective finetuned models used in the Ext-3DFMs paper. Other evaluations used in the original π³ code still are available in this repository, but are not the main focus.

Specifically, this code can be used to replicate Extreme-View 3DFM paper results for:
- Monocular Depth Estimation on Sintel, Bonn, KITTI, and NYU-v2
- Multiview Pose Estimation on RealEstate10K and ETH3D
- Dense Reconstruction on UnSceneRecon, ETH3D, DTU, 7Scenes, and NRGBD

For Extreme Relative Rotation Estimation evaluation, please see [the main GitHub repository](https://github.com/theREALevan/extreme-view-3dfm).

## Setup

Please clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/jot-jt/extreme-view-3dfm-gen-eval.git
```

### Configurations and General Documentation
Please refer to the [original π³ GitHub repository's README](https://github.com/ZhouTimeMachine/recons_eval/blob/main/README.md) for evaluation/config documentation and dataset preparation details. For this fork, we focus on commands that replicate Ext-3DFMs' results.

### Using MegaUnScene for UnSceneRecon Evaluation
A new addition is the [MegaUnScene dataset](https://huggingface.co/datasets/cornell-vailab/megaunscene) for the UnSceneRecon dense reconstruction evaluation. Please update the MegaUnScene root or link the dataset accordingly in [configs/data/mv_recon.yaml](configs/data/mv_recon.yaml). Refer to the dataset page for download instructions.

### Config Model Names
In the configs, the model names are `vggt`, `pi3`, and `worldmirror`. The finetuned versions for extreme rotation are `vggt_exrot`, `pi3_exrot`, and `worldmirror_exrot`.

For dense reconstruction, we evaluate the point head for VGGT (called `vggt_pointhead` and `vggt_exrot_pointhead`) in the paper, rather than the unprojected depth maps (`vggt` and `vggt_exrot`).

## 1. Monocular Depth Estimation

See [monodepth/README.md](monodepth/README.md) for more details.

To evaluate all of Pi3, VGGT, WorldMirror, and their finetuned versions on Sintel, Bonn, KITTI, and NYU-v2, run the commands below to use the preconfigured config at [configs/evaluation/monodepth.yaml](configs/evaluation/monodepth.yaml):

```bash
python monodepth/infer.py
# torchrun --nnodes=1 --nproc_per_node=8 monodepth/infer_mp.py  # accelerate with multi gpus
python monodepth/eval.py
```

To evaluate specific models and/or datasets, run:

```bash
python monodepth/infer.py eval_models=[MODEL_NAMES] eval_datasets=[EVAL_DATASETS]
python monodepth/eval.py eval_models=[MODEL_NAMES] eval_datasets=[EVAL_DATASETS]
```

where `MODEL_NAMES` is any of `pi3,pi3_exrot,vggt,vggt_exrot,worldmirror,worldmirror_exrot`

and `EVAL_DATASETS` is any of `sintel,bonn,kitti,nyu-v2`

## 2. Multiview Pose Estimation

configs in `configs/evaluation/relpose-angular.yaml`, see [relpose/README.md](relpose/README.md) for more details.

To evaluate all of Pi3, VGGT, WorldMirror, and their finetuned versions on RealEstate10K and ETH3D, run the commands below to use the preconfigured config at [configs/evaluation/relpose-angular.yaml](configs/evaluation/relpose-angular.yaml):

```bash
# python relpose/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python relpose/eval_angle.py
# torchrun --nnodes=1 --nproc_per_node=8 videodepth/eval_angle_mp.py   # accelerate with multi gpus
```

To evaluate specific models and/or datasets, run:

```bash
python relpose/eval_angle.py eval_models=[MODEL_NAMES] eval_datasets=[EVAL_DATASETS]
```

where `MODEL_NAMES` is any of `pi3,pi3_exrot,vggt,vggt_exrot,worldmirror,worldmirror_exrot`

and `EVAL_DATASETS` is any of `Re10K,ETH3D`


## 3. Dense Reconstruction (Point Map Estimation)

See [mv_recon/README.md](mv_recon/README.md) for more details.

To evaluate all of Pi3, VGGT, WorldMirror, and their finetuned versions on UnSceneRecon, ETH3D, DTU, 7Scenes, and NRGBD, run the commands below to use the preconfigured config at [configs/evaluation/mv_recon.yaml](configs/evaluation/mv_recon.yaml):

```bash
# python mv_recon/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python mv_recon/eval.py
# torchrun --nnodes=1 --nproc_per_node=8 mv_recon/eval_mp.py  # accelerate with multi gpus
```

To evaluate specific models and/or datasets, run:

```bash
python monodepth/eval.py eval_models=[MODEL_NAMES] eval_datasets=[EVAL_DATASETS]
```

where `MODEL_NAMES` is any of `pi3,pi3_exrot,vggt_pointhead,vggt_exrot_pointhead,worldmirror,worldmirror_exrot`

and `EVAL_DATASETS` is any of `7scenes-sparse,7scenes-dense,NRGBD-sparse,NRGBD-dense,DTU,ETH3D,UnSceneRecon`

Note that for this evaluation, we use the VGGT model output from the point head (`vggt_pointhead` and `vggt_exrot_pointhead`), not unprojected depth (`vggt` and `vggt_exrot`).

## Acknowledgements

This fork mainly builds upon:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [MoGe](https://github.com/microsoft/MoGe)
- [VGGT](https://github.com/facebookresearch/vggt)
- [π³](https://github.com/yyfz/Pi3) ([alt](https://github.com/ZhouTimeMachine/recons_eval))
- [WorldMirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)


## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{zhang2025emergentextremeviewgeometry3d,
      title={Emergent Extreme-View Geometry in 3D Foundation Models}, 
      author={Yiwen Zhang and Joseph Tung and Ruojin Cai and David Fouhey and Hadar Averbuch-Elor},
      year={2025},
      eprint={2511.22686},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.22686}, 
}
```

Please also consider citing the original [π³](https://arxiv.org/abd/2507.13347) authors, who provided the foundations for this evaluation framework:

```bibtex
@misc{wang2025pi3,
      title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning}, 
      author={Yifan Wang and Jianjun Zhou and Haoyi Zhu and Wenzheng Chang and Yang Zhou and Zizun Li and Junyi Chen and Jiangmiao Pang and Chunhua Shen and Tong He},
      year={2025},
      eprint={2507.13347},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13347}, 
}
```

## License

This project is licensed under CC BY-NC-SA 4.0 License. See the LICENSE file and https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.
