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

## Introduction

```bibtex
@InProceedings{DistgEPIT,
    author    = {Jin, Kai and Wei, Zeqiang and Yang, Angulia and Gao, Mingzhi and Zhou, Xiuzhuang},
    title     = {LFTransMamba: A Hybrid Transformer-Mamba Network for Light Field Image Super-Resolution},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year      = {2025},
}
```

## Instructions

### 1. Dependencies

It is recommended to use a **Python 3.10** or above version.

```
conda create -n lftransmamba python=3.10
pip install opencv-python numpy scikit-image h5py imageio mat73 scipy torch torchvision einops fvcore timm scikit-learn

python setup.py develop
```

### 2. Prepare Dataset

```bash
python Generate_Data_for_Training.py --angRes 5 --scale_factor 4
python Generate_Data_for_Test.py --angRes 5 --scale_factor 4
python Generate_Data_for_inference.py --angRes 5 --scale_factor 4
```

### 3. Inference

#### For Public Validation

```bash
# Track1 - Classic
python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2.VAL --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task val_all --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track1.classic.pth --tta

# Track2 - Efficiency
python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0.VAL --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task val_all --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track2.efficiency.pth

# Track3 - Large Model
python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.VAL --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task val_all --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track3.llm.pth --fp16 --tta

```

#### For NTIRE-25 Validation

```bash
# Track1 - Classic
python lfsr.py --name  Track1.LFTransMamba_D16_R1_K18_C80_T2.VAL.NTIRE --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track1.classic.pth --tta

# Track2 - Efficiency
python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0.VAL.NTIRE --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track2.efficiency.pth

# Track3 - Large Model
python lfsr.py --name  Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.VAL.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.VAL --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track3.llm.pth --fp16 --tta
```

#### For NTIRE-25 Test

```bash
# Track1 - Classic
python lfsr.py --name  Track1.LFTransMamba_D16_R1_K18_C80_T2.TEST.NTIRE --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track1.classic.pth --tta

# Track2 - Efficiency
python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0.TEST.NTIRE --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track2.efficiency.pth

# Track3 - Large Model
python lfsr.py --name  Track3.LFTransMamba_D16_R1_K24_C80_T2.B3.FP16.TEST.NTIRE --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task test --dataset LFSR.NTIRE.TEST --scale 4 --patch-size 32 --stride 8 --processor sad --angular 5 --model-source vanilla --model-path checkpoints/track3.llm.pth --fp16 --tta
```

### 4. Training

```bash
# Track1 - Classic
CUDA_VISIBLE_DEVICES=1,2,3 python lfsr.py --name Track1.LFTransMamba_D16_R1_K18_C80_T2 --model LFTransMamba_D16_R1_K18_C80_T2 --device cuda:0 --task train --scale 4 --patch-size 32 --stride 8 --processor psw --angular 5 --dataset LFSR.ALL4 --log 10 --log-val 1 --log-save 1 --train-lr-scheduler 2 --train-batchsize 3 --train-epoch 30 --train-lr 3e-4 --ema 0.999

# Track2 - Efficiency
CUDA_VISIBLE_DEVICES=1,2,3 python lfsr.py --name Track2.LFTransMamba_D16_R1_K6_C32_T0 --model LFTransMamba_D16_R1_K6_C32_T0 --device cuda:0 --task train --scale 4 --patch-size 32 --stride 12 --processor psw --angular 5 --dataset LFSR.ALL4 --log 10 --log-val -1 --log-save 1 --train-lr-scheduler 3 --train-batchsize 3 --train-epoch 100 --train-lr 2e-4 --ema 0.999

# Track3 - Large Model
CUDA_VISIBLE_DEVICES=1,2,3 python lfsr.py --name Track3.LFTransMamba_D16_R1_K24_C80_T2 --model LFTransMamba_D16_R1_K24_C80_T2 --device cuda:0 --task train --scale 4 --patch-size 32 --stride 8 --processor psw --angular 5 --dataset LFSR.ALL4.EXTRA --log 10 --log-val -1 --log-save 1 --train-lr-scheduler 2 --train-batchsize 3 --train-epoch 30 --train-lr 3e-4 --ema 0.999 --fp16
```

## Citation

If you find this work helpful, please consider citing the following papers:

```bibtex
@InProceedings{BasicLFSR,
  author    = {Wang, Yingqian and Wang, Longguang and Liang, Zhengyu and Yang, Jungang and Timofte, Radu and Guo, Yulan and Jin, Kai and Wei, Zeqiang and Yang, Angulia and Guo, Sha and Gao, Mingzhi and Zhou, Xiuzhuang and Duong, Vinh Van and Huu, Thuc Nguyen and Yim, Jonghoon and Jeon, Byeungwoo and Liu, Yutong and Cheng, Zhen and Xiao, Zeyu and Xu, Ruikang and Xiong, Zhiwei and Liu, Gaosheng and Jin, Manchang and Yue, Huanjing and Yang, Jingyu and Gao, Chen and Zhang, Shuo and Chang, Song and Lin, Youfang and Chao, Wentao and Wang, Xuechun and Wang, Guanghui and Duan, Fuqing and Xia, Wang and Wang, Yan and Xia, Peiqi and Wang, Shunzhou and Lu, Yao and Cong, Ruixuan and Sheng, Hao and Yang, Da and Chen, Rongshan and Wang, Sizhe and Cui, Zhenglong and Chen, Yilei and Lu, Yongjie and Cai, Dongjun and An, Ping and Salem, Ahmed and Ibrahem, Hatem and Yagoub, Bilel and Kang, Hyun-Soo and Zeng, Zekai and Wu, Heng},
  title     = {NTIRE 2023 Challenge on Light Field Image Super-Resolution: Dataset, Methods and Results},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2023},
}
```

## Contact

Welcome to send email to jinkai@bigo.sg, if you have any questions about this repository or other issues.
