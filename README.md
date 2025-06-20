[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
LFTransMamba: A Hybrid Transformer-Mamba Network for Light Field Image Super-Resolution
</b>
</h1>
<h4>
<b>
Kai Jin, Zeqiang Wei, Angulia Yang, Mingzhi Gao, Xiuzhuang Zhou
</b>
</h4>
</div>


## News

✅ **Mar, 2025:** This repository contains official pytorch implementation of "LFTransMamba: A Hybrid Transformer-Mamba Network for Light Field Image Super-Resolution" in **? solutions 👑** in [NTIRE2025 Light-Field Super Resolution: Track 1 Classsic](https://codalab.lisn.upsaclay.fr/competitions/21276#results)

✅ **Mar, 2025:** This repository contains official pytorch implementation of "LFTransMamba: A Hybrid Transformer-Mamba Network for Light Field Image Super-Resolution" in **? solutions 👑** in [NTIRE2025 Light-Field Super Resolution: Track 2 Efficiency](https://codalab.lisn.upsaclay.fr/competitions/21277#results)

✅ **Mar, 2025:** This repository contains official pytorch implementation of "LFTransMamba: A Hybrid Transformer-Mamba Network for Light Field Image Super-Resolution" in **? solutions 👑** in [NTIRE2025 Light-Field Super Resolution: Track 3 Large Model](https://codalab.lisn.upsaclay.fr/competitions/21278#results)

## Citation

If you find this work helpful, please consider citing the following papers:

```bibtex
@ARTICLE{Wei_L2FMamba_TCI25,
    author   = {Wei, Zeqiang and Jin, Kai and Hou, Zeyi and Song, Kuan and Zhou, Xiuzhuang},
    journal  = {IEEE Transactions on Computational Imaging}, 
    title    = {$L^{2}$FMamba: Lightweight Light Field Image Super-Resolution With State Space Model}, 
    year     = {2025},
    volume   = {11},
    pages    = {816-826},
    doi      = {10.1109/TCI.2025.3577338}
}

@InProceedings{Jin_LFTransMamba_CVPR25,
    author    = {Jin, Kai and Wei, Zeqiang and Yang, Angulia and Wu, Di and Gao, Mingzhi and Zhou, Xiuzhuang},
    title     = {LFTransMamba: A Hybrid Mamba-Transformer Model for Light Field Image Super-Resolution},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {1195-1204}
}
```

## Instructions

### 1. Dependencies

It is recommended to use a **Python 3.10** or above version.

```
conda create -n lftransmamba python=3.10
pip install opencv-python numpy==1.26.4 scikit-image==0.19.0 h5py imageio mat73 scipy torch torchvision einops fvcore timm scikit-learn

python setup.py develop
```

### 2. Prepare Dataset

```bash
python Generate_Data_for_Training.py --angRes 5 --scale_factor 4
python Generate_Data_for_Test.py --angRes 5 --scale_factor 4
python Generate_Data_for_inference.py --angRes 5 --scale_factor 4
```

### 3. Inference

Pretrained Model could be downloaded from: [Google Drive](https://drive.google.com/drive/folders/1EYAvQ-aNjBJOyJhQSwhq-YiAZSYU51wC)

#### For Public Validation

```bash
# Track1 - Classic (ensemble 3 models)
# mean_psnr:33.7795, mean_ssim:0.9543, real_psnr:32.6072, synth_psnr:35.5381
python lfsr.py --name Track1.LF_DET_0214.P16S8.VAL --model LF_DET_0214 --device cuda:0 --task val_all --scale 4 --patch-size 16 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LF_DET_0214.pth --tta
# mean_psnr:33.8803, mean_ssim:0.9549, real_psnr:32.7436, synth_psnr:35.5854
python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2.P40S12.VAL --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task val_all --scale 4 --patch-size 40 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LFTransMamba_D16_R1_K18_C80_T2.pth --tta
# mean_psnr:33.8786, mean_ssim:0.9549, real_psnr:32.7512, synth_psnr:35.5698
python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2.P48S12.VAL --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task val_all --scale 4 --patch-size 48 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LFTransMamba_D16_R1_K18_C80_T2.pth --tta

# Track2 - Efficiency
# mean_psnr: 32.9525, mean_ssim: 0.9478, real_psnr: 31.6891, synth_psnr: 34.8476
python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0.VAL --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task val_all --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track2.L2FMamba_D16_R1_K6_C32_T0.pth

# Track3 - Large Model (ensemble track-1 models and track3 models)
# mean_psnr:33.9249, mean_ssim:0.9553, real_psnr:32.7735, synth_psnr:35.6521
python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P32S8.VAL --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task val_all --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta
# mean_psnr:33.9310, mean_ssim:0.9553, real_psnr:32.7864, synth_psnr:35.6480
python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P40S8.VAL --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task val_all --scale 4 --patch-size 40 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta
# mean_psnr:33.9192, mean_ssim:0.9553, real_psnr:32.7783, synth_psnr:35.6307
python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P48S12.VAL --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task val_all --scale 4 --patch-size 48 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta

```

#### For NTIRE-25 Validation

```bash
# Track1 - Classic (ensemble 3 models)
python lfsr.py --name Track1.LF_DET_0214.P16S8.VAL.NTIRE --model LF_DET_0214 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 16 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LF_DET_0214.pth --tta

python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2.P40S12.VAL.NTIRE --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 40 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LFTransMamba_D16_R1_K18_C80_T2.pth --tta

python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2.P48S12.VAL.NTIRE --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 48 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LFTransMamba_D16_R1_K18_C80_T2.pth --tta

# Track2 - Efficiency
python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0.VAL.NTIRE --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track2.L2FMamba_D16_R1_K6_C32_T0.pth

# Track3 - Large Model (ensemble track-1 models and track3 models)
python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P32S8.VAL.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta

python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P40S8.VAL.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 40 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta

python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P48S12.VAL.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 48 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta
```

#### For NTIRE-25 Test

```bash
# Track1 - Classic (ensemble 3 models)
python lfsr.py --name Track1.LF_DET_0214.P16S8.TEST.NTIRE --model LF_DET_0214 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 16 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LF_DET_0214.pth --tta

python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2.P40S12.TEST.NTIRE --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 40 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LFTransMamba_D16_R1_K18_C80_T2.pth --tta

python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2.P48S12.TEST.NTIRE --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 48 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track1.LFTransMamba_D16_R1_K18_C80_T2.pth --tta

# Track2 - Efficiency
python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0.TEST.NTIRE --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track2.L2FMamba_D16_R1_K6_C32_T0.pth

# Track3 - Large Model (ensemble track-1 models and track3 models)
python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P32S8.TEST.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta

python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P40S8.TEST.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 40 --stride 8 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta

python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.P48S12.TEST.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 48 --stride 12 --processor psw++ --angular 5 --model-source vanilla --model-path checkpoints/Track3.LFTransMamba_D16_R1_K24_C80_T2.pth --fp16 --tta
```

### 4. Training

```bash
# Track1 - Classic
python lfsr.py --name Track1.LF_DET_0214 --model LF_DET_0214 --device cuda:0 --task train --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --dataset LFSR.ALL4 --log 10 --log-val 1 --log-save 1 --train-lr-scheduler 2 --train-batchsize 3 --train-epoch 30 --train-lr 2e-4 --ema 0.999

python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2 --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task train --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --dataset LFSR.ALL4 --log 10 --log-val 1 --log-save 1 --train-lr-scheduler 2 --train-batchsize 3 --train-epoch 30 --train-lr 3e-4 --ema 0.999

# Track2 - Efficiency
python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0 --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task train --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --dataset LFSR.ALL4 --log 10 --log-val -1 --log-save 1 --train-lr-scheduler 3 --train-batchsize 3 --train-epoch 100 --train-lr 2e-4 --ema 0.999

# Track3 - Large Model
python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2 --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task train --scale 4 --patch-size 32 --stride 8 --processor psw++ --angular 5 --dataset LFSR.ALL4.EXTRA --log 10 --log-val -1 --log-save 1 --train-lr-scheduler 2 --train-batchsize 3 --train-epoch 30 --train-lr 3e-4 --ema 0.999 --fp16
```

## Contact

Welcome to send email to jinkai@bigo.sg, if you have any questions about this repository or other issues.
