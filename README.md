# PSFusion
This is official Pytorch implementation of "Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity"
## Comparsion
<div>
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Demo.jpg" alt="Demo" height="300" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/parm.jpg" alt="Parm" height="300" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>

<p align="center">
    <em>(a) Comparison of fusion and segmentation results. | (b) Comparison of the computational complexity for semantic segmentation.</em>
</p>

## Framework
<div>
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/Framework.jpg" alt="Framework" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>The overall framework of the proposed PSFusion.</em>
</p>

## Network Architecture
<div>
    <center>
        <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/SDFM.jpg" alt="SDFM" width="800" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
    </center>
</div>
<p align="center">
    <em>The architecture of the superficial detail fusion module (SDFM) based on the channel-spatial attention mechanism.</em>
</p>

<div style="text-align: center;">
    <img src="https://github.com/Linfeng-Tang/PSFusion/blob/main/Figure/PSFM.jpg" alt="PSFM" width="800" style="display:inline-block;margin-right:20px;margin-bottom:20px;">
</div>
<p align="center">
    <em>The architecture of the profound semantic fusion module (PSFM) based on the cross-attention mechanism.</em>
</p>
