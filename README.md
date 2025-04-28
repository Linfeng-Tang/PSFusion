# PSFusion
This is official Pytorch implementation of "[Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity](https://www.sciencedirect.com/science/article/pii/S1566253523001860)"

## ✨News:

[2025-3-15] 我们的论文《[C2RF: Bridging Multi-modal Image Registration and Fusion via Commonality Mining and Contrastive Learning]([https://github.com/Linfeng-Tang/C2RF))》被**International Journal of Computer Vision (IJCV)** 正式接收！[[论文下载](https://link.springer.com/article/10.1007/s11263-025-02427-1)] [[Code](https://github.com/Linfeng-Tang/C2RF)]

[2025-02-11] 我们发布了一个用于红外和可见光视频融合的大规模数据集：[M2VD: Multi-modal Multi-scene Video Dataset](https://github.com/Linfeng-Tang/M2VD).


```
@article{TANG2023PSFusion,
  title={Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity},
  author={Tang, Linfeng and Zhang, Hao and Xu, Han and Ma, Jiayi},
  journal={Information Fusion},
  volume = {99},
  pages = {101870},
  year={2023},
}
```

## Image Fusion Example
![Schematic illustration of multi-modal image fusion and digital photography image fusion. ](https://github.com/Linfeng-Tang/SwinFusion/blob/master/SwinFusion/Schematic_illustration.jpg)
Schematic illustration of multi-modal image fusion and digital photography image fusion. First row: source image pairs, second row: fused results of U2Fusion and our SwinFusion.

 
## Framework
<div>
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Framework.jpg" alt="Framework" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>The overall framework of the proposed PSFusion.</em>
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
3. `python test_Fusion.py --dataroot=./datasets --dataset_name=MSRS --resume=./results/PSFusion/checkpoints/best_model.pth`

If you need to test other datasets, please put the dataset according to the dataloader and specify **--dataroot** and **--dataset-name**

## To Train 
Before training PSFusion, you need to download the pre-processed **MSRS** dataset [MSRS](https://pan.baidu.com/s/18q_3IEHKZ48YBy2PzsOtRQ?pwd=MSRS) and putting it in **./datasets**.

Then running `python train.py --dataroot=./datasets/MSRS --name=PSFusion`

## Motivation
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Demo.jpg" alt="Demo" width="800"  style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>

<p align="center">
    <em>Comparison of fusion and segmentation results between <a href="https://github.com/Linfeng-Tang/SeAFusion">SeAFusion</a> and our method under harsh conditions.</em>
</p>
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/parm.jpg" alt="Parm" width="800"  style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>Comparison of the computational complexity between feature-level fusion and image-level fusion for the semantic segmentation task.</em>
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

## To Segmentation
### [BANet](https://github.com/Linfeng-Tang/PSFusion/tree/BANet)
 
### [SegFormer](https://github.com/Linfeng-Tang/PSFusion/tree/SegFormer)
 
### [SegNeXt](https://github.com/Linfeng-Tang/PSFusion/tree/SegNeXt)
 

## Experiments
### Qualitative fusion results
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/MSRS_F.jpg" alt="MSRS" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Qualitative comparison of PSFusion with 9 state-of-the-art methods on the **MSRS** dataset.</em>
</p>
    
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/M3FD_F.jpg" alt="M3FD" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Qualitative comparison of PSFusion with 9 state-of-the-art methods on the **M3FD** dataset.</em>
</p>
    
### Quantitative fusion results    
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/MSRS.jpg" alt="MSRS" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Quantitative comparisons of the six metrics on 361 image pairs from the MSRS dataset. A point (x, y) on the curve denotes that there are (100*x)% percent of image pairs which have metric values no more than y.</em>
</p>
   
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/M3FD.jpg" alt="M3FD" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Quantitative comparisons of the six metrics on 300 image pairs from the M3FD dataset.</em>
</p>
   
### Segmentation comparison
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Seg_MSRS.jpg" alt="MSRS" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Segmentation results of various fusion algorithms on the MSRS dataset.</em>
</p>

<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Tabel_MSRS.jpg" alt="MSRS" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Per-class segmentation results on the MSRS dataset. </em>
</p>

### Potential of image-level fusion for high-level vision tasks
<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Seg_MFNet.jpg" alt="MFNet" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Segmentation results of feature-level fusion-based multi-modal segmentation algorithms and our image-level fusion-based solution on the MFNet dataset.</em>
</p>

<div align="center">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Tabel_MFNet.jpg" alt="MSRS" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em><span style="font-size: 50px;">Per-class segmentation results of image-level fusion and feature-level fusion on the MFNet dataset.</em>
</p>

## If this work is helpful to you, please cite it as：
```
@article{TANG2023PSFusion,
  title={Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity},
  author={Tang, Linfeng and Zhang, Hao and Xu, Han and Ma, Jiayi},
  journal={Information Fusion},
  volume={99},
  pages={101870},
  year={2023},
  publisher={Elsevier}
}
```
