# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision
import kornia.filters as KF
# from loss_ssim import ssim
shape = (256, 256)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from Fusion_losses import * 

"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    #print(mask.shape,ssim_map.shape)
    ssim_map = ssim_map*mask

    ssim_map = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def sigma_transformation(x):
    return torch.exp(2 * x) -1

def Contrast(img1, img2, window_size=11, channel=1, eps=1e-6):
    ## img1 is the infrared image, img2 is the visible image
    window = create_window(window_size, channel)    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq ## 写代码尝试显示一下图像的对比度图
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    
    sigma1 = sigma1_sq / (2 * (sigma1_sq + sigma2_sq) + eps)
    sigma2 = 1 - sigma1
    return sigma1, sigma2

    
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=1):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        mask = torch.logical_and(img1>0,img2>0).float()
        for i in range(self.window_size//2):
            mask = (F.conv2d(mask, window, padding=self.window_size//2, groups=channel)>0.8).float()
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask=mask)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1) # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2) # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3) # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4) # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 , 1.0]

    def forward(self, x, y):
        contentloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        contentloss += self.L1Loss(x_vgg[3], y_vgg[3].detach())
        return contentloss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
# def Fusion_loss(ir, vi, fu, mask, weights=[1, 1, 1]):
#     sigma_ir, sigma_vi = Contrast(ir, vi)
#     ssim_loss = SSIMLoss(window_size=11)
#     grad_ir =  KF.spatial_gradient(ir, order=2).abs().sum(dim=[1,2])
#     grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1,2])
#     grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1,2])
#     grad_joint = torch.max(grad_ir, grad_vi)
#     ## 梯度损失
#     loss_grad = F.l1_loss(grad_fus, grad_joint)
#     ## SSIM损失
#     loss_ssim = 0.5 * ssim_loss(ir, fu)+ 0.5 * ssim_loss(vi,fu)
#     ## 强度损失
#     loss_intensity_obj = F.l1_loss(mask * fu, mask * ir) #在目标区域迫使融合结果的强度与红外图像的强度保一致
#     loss_intensity_back = F.l1_loss((1 - mask) * fu, (1 - mask) * (sigma_ir * ir + sigma_vi * vi))
#     loss_intensity = loss_intensity_obj + loss_intensity_back
#     loss_total = weights[0] * loss_ssim + weights[1] * loss_grad + weights[2] * loss_intensity
#     return loss_total, loss_ssim, loss_grad, loss_intensity

def Fusion_loss(ir, vi, fu, mask, weights=[1, 10, 10], device=None):
    # grad_ir =  KF.spatial_gradient(ir, order=2).abs().sum(dim=[1,2])
    # grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1,2])
    # grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1,2])
    # grad_joint = torch.max(grad_ir, grad_vi)
    sobelconv=Sobelxy(device) 
    vi_grad_x, vi_grad_y = sobelconv(vi)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    
    loss_grad=F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
    ## 梯度损失
    # loss_grad = F.l1_loss(grad_fus, grad_joint)
    ## SSIM损失
    loss_ssim = corr_loss(ir, vi, fu)
    ## 强度损失
    loss_intensity = final_mse1(ir, vi, fu, mask) + 0 * F.l1_loss(fu, torch.max(ir, vi))
    loss_total = weights[0] * loss_ssim + weights[1] * loss_grad + weights[2] * loss_intensity
    return loss_total, loss_intensity, loss_grad, loss_ssim


class Fusionloss(nn.Module):
    def __init__(self, device):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy(device)      

    def forward(self, vi, ir, fu, mask=None):
        # 定义内容损失
        # 亮度
        # x_in_max=x_inf_origin
        # temp=(image_ir*x_y_origin)/(1.0-image_ir)+x_y_origin
        # white=torch.ones_like(image_ir)
        # temp=torch.min(white,temp)
        x_in_max=(torch.max(ir,vi)) 
        loss_in=F.l1_loss(x_in_max * (1-mask),fu*(1-mask)) + F.l1_loss(ir * mask, fu * mask)
        #梯度
        vi_grad_x, vi_grad_y = self.sobelconv(vi)
        ir_grad_x, ir_grad_y = self.sobelconv(ir)
        fu_grad_x, fu_grad_y = self.sobelconv(fu)
        grad_joint_x = torch.max(vi_grad_x, ir_grad_x)        
        grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
        loss_grad=F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
        # 内容损失
        loss_total=5*loss_in+loss_grad
        return loss_total, loss_in, loss_grad

class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)
    
    
def Re_loss(img1, img2, weights=[5, 5], mask=None, ir_flag=False):
    ## img1 is the reconstructed image, img2 is the reference image
    grad1 =  KF.spatial_gradient(img1, order=2).abs().sum(dim=[1,2])
    grad2 = KF.spatial_gradient(img2, order=2).abs().sum(dim=[1,2])
    loss_intensity = F.l1_loss(img1, img2)
    loss_grad = F.l1_loss(grad1, grad2)
    loss_total = weights[0] * loss_intensity + weights[1] * loss_grad
    return loss_total, loss_intensity, loss_grad

def Seg_loss(pred, label, device, criteria=None):
    '''
    利用预训练好的分割网络,计算在融合结果上的分割结果与真实标签之间的语义损失
    :param fused_image:
    :param label:
    :param seg_model: 分割模型在主函数中提前加载好,避免每次充分load分割模型
    :return seg_loss:
    fused_image 在输入Seg_loss函数之前需要由YCbCr色彩空间转换至RGB色彩空间
    '''
    # 计算语义损失         
    lb = torch.squeeze(label, 1)
    seg_loss = criteria(pred, lb)
    # lb = torch.squeeze(label, 1)
    # seg_results = torch.argmax(pred, dim=1, keepdim=True)
    # seg_loss = lovasz_softmax(pred, lb)
    return seg_loss

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, device, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).to(device)
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

def l1loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,[3,3],[1,1])*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],[1,1])*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).mean()

def l2loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,[3,3],[1,1])*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],[1,1])*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss,self).__init__()
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
        self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()
    def forward(self,img1,img2,mask=1,eps=1e-2):
        #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
        mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
        mean_ = mean_.detach()/2
        std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
        std_ = std_.detach()/2 
        img1 = (img1-mean_)/std_
        img2 = (img2-mean_)/std_
        grad1 = KF.spatial_gradient(img1,order=2)
        grad2 = KF.spatial_gradient(img2,order=2)
        mask = mask.unsqueeze(1)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        l = (((grad1-grad2)+(grad1-grad2).pow(2)*10)*mask).abs().clamp(min=eps).mean()
        #l = l[...,5:-5,10:-10].mean()
        return l
    
class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss,self).__init__()
        
    def forward(self, img1, img2, mask=1, eps=1e-3):
        mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
        mean_ = mean_.detach()/2
        std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
        std_ = std_.detach()/2 
        img1 = (img1-mean_)/std_
        img2 = (img2-mean_)/std_
        grad1 = KF.spatial_gradient(img1,order=2)
        grad2 = KF.spatial_gradient(img2,order=2)
        mask = mask.unsqueeze(1)
        l = (((grad1-grad2)+(grad1-grad2).pow(2)*10)*mask).abs().clamp(min=eps).mean()
        return l


def l2regularization(img):
    return img.pow(2).mean()
# def l1loss(img1,img2,mask=1,eps=1e-2):
#     img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
#     img2 = KF.gaussian_blur2d(img2,[7,7],[2,2])
#     return ((img1-img2)*mask).abs().clamp(min=eps).mean()

# def l2loss(img1,img2,mask=1,eps=1e-2):
#     img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
#     img2 = KF.gaussian_blur2d(img2,[7,7],[2,2])
#     return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

# class gradientloss(nn.Module):
#     def __init__(self):
#         super(gradientloss,self).__init__()
#         self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
#         self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()
#     def forward(self,img1,img2,mask=1,eps=1e-3):
#         #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
#         #img2 = KF.gaussian_blur2d(img2,[7,7],[2,2])
#         grad1 = KF.spatial_gradient(img1,order=2).abs().sum(dim=[1,2])
#         grad2 = KF.spatial_gradient(img2,order=2).abs().sum(dim=[1,2])
#         # grad1 = self.AP5(self.MP5(grad1))
#         # grad2 = self.AP5(self.MP5(grad2))
#         l = ((grad1-grad2)*mask).abs().clamp(min=eps).mean()
#         return l

# def smoothloss(img):
#     grad = KF.spatial_gradient(img,order=2).mean(dim=1).abs().sum(dim=1)
#     return grad.clamp(min=1e-2,max=0.5).mean()
# a = torch.rand(1,2,256,256)
# a[:,1]=0
# smoothloss(a)
def orthogonal_loss(t):
    # C=A'A, a positive semi-definite matrix
    # should be close to I. For this, we require C
    # has eigen values close to 1
    c = torch.matmul(t, t)
    k = torch.linalg.eigvals(c)  # Get eigenvalues of C
    ortho_loss = torch.mean((k[0][0] - 1.0) ** 2) + torch.mean((k[0][1] - 1.0) ** 2)
    ortho_loss = ortho_loss.float()
    return ortho_loss






def feat_loss(feat1,feat2,grid=16):
    b,c,h,w=feat1.shape[0],feat1.shape[1],feat1.shape[2],feat1.shape[3]
    shift_x = np.random.randint(1,w//grid)
    shift_y = np.random.randint(1,h//grid)
    x = tuple(np.arange(grid)*w//grid+shift_x)
    y = tuple(np.arange(grid)*w//grid+shift_y)
    feat1_sampled = feat1[:,:,y,:]
    feat1_sampled = F.normalize(feat1_sampled[:,:,:,x],dim=1).view(b,c,-1).permute(0,2,1).contiguous().view(-1,c)
    feat2_sampled = feat2[:,:,y,:]
    feat2_sampled = F.normalize(feat2_sampled[:,:,:,x],dim=1).view(b,c,-1).permute(0,2,1).contiguous().view(-1,c)
    # .view(b,c,-1).permute(0,2,1).view(-1,c)
    featset = torch.cat([feat1_sampled,feat2_sampled])
    perseed = torch.randperm(featset.shape[0])
    featset = featset[perseed][0:feat1_sampled.shape[0]]
    simi_pos = (feat1_sampled*feat2_sampled).sum(dim=-1)
    simi_neg = (feat1_sampled*featset).sum(dim=-1) if torch.rand(1)>0.5 else (feat2_sampled*featset).sum(dim=-1)
    loss = (simi_neg-simi_pos+0.5).clamp(min=0.0).mean()
    return loss

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes,weight=1)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
