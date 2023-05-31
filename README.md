# PSFusion
This is official Pytorch implementation of "Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity"

## Motivation
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Demo.jpg" alt="Demo" width="800"  style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>

<p align="center">
    <em>Comparison of fusion and segmentation results. </em>
</p>
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/parm.jpg" alt="Parm" width="800"  style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>Comparison of the computational complexity for semantic segmentation.</em>
</p>
## Framework
<div>
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Framework.jpg" alt="Framework" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>The overall framework of the proposed PSFusion.</em>
</p>

## Network Architecture
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/SDFM.jpg" alt="SDFM" width="800" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>The architecture of the superficial detail fusion module (SDFM) based on the channel-spatial attention mechanism.</em>
</p>

<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/PSFM.jpg" alt="PSFM" width="800" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">The architecture of the profound semantic fusion module (PSFM) based on the cross-attention mechanism.</em>
</p>

## Recommended Environment
 - [ ] torch  1.10.0
 - [ ] cudatoolkit 11.3.1
 - [ ] torchvision 0.11.0
 - [ ] kornia 0.6.5
 - [ ] pillow  8.3.2
    
## To Test
1. Downloading the pre-trained checkpoint from [best_model.pth](https://pan.baidu.com/s/1N_dZvfiKwuwQf2DZPstJ0A?pwd=PSFu) and putting it in **./results/PSFusion/checkpoints**.
2. Downloading the MSRS dataset from [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS) and putting it in **./datasets**.
3. `python test.py --dataroot=./datasets/MSRS --dataset_name=MSRS --resume=./results/PSFusion/checkpoints/best_model.pth`

If you need to test other datasets, please put the dataset according to the dataloader and specify **--dataroot** and **--dataset-name**

## To Train 
Before training PSFusion, you need to download the pre-processed **MSRS** dataset [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS) and putting it in **./datasets**.

Then running `python train.py --dataroot=./datasets/MSRS --name=PSFusion`

## To Segmentation
### BANet
### Segmentation
### SegNeXt

## Experiments
### Fusion results on the MSRS dataset
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/MSRS_F.jpg" alt="MSRS" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Qualitative comparison of PSFusion with 9 state-of-the-art methods on the MSRS dataset.</em>
</p>
    
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/MSRS.jpg" alt="MSRS" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Quantitative comparisons of the six metrics on $361$ image pairs from the MSRS dataset. A point ($x$, $y$) on the curve denotes that there are ($100*x$)\% percent of image pairs which have metric values no more than $y$.</em>
</p>
