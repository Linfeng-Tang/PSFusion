import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

def conv1(in_chsnnels, out_channels):
    "1x1 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=1, stride=1, bias=False)


def conv3(in_chsnnels, out_channels):
    "3x3 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x2 = torch.stack([x, x1], dim=0)
        out, _ = torch.max(x2, dim=0)
        return out

class Feature_extract(nn.Module):
    '''
    特征提取模块
    '''
    def __init__(self, in_channels, out_channels):
        super(Feature_extract, self).__init__()
        self.SFEB1 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels/2), kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(int(out_channels/2)),
            FReLU(int(out_channels/2)),
            nn.Conv2d(int(out_channels/2), int(out_channels/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_channels/2)),
            FReLU(int(out_channels/2)),
        )
        self.SFEB2= nn.Sequential(
            nn.Conv2d(int(out_channels/2), out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            FReLU(out_channels),
            nn.Conv2d(out_channels,  out_channels, kernel_size=3, stride=1, padding=1),)

    def forward(self, x):        
        high_x = self.SFEB1(x)
        x = self.SFEB2(high_x)
        return high_x, x

class S2M(nn.Module):
    '''
    Scene Specific Mask
    '''
    def __init__(self, channels, r=4):
        super(S2M, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels), 
            nn.ReLU(inplace=True),            
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels), 
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_block = nn.BatchNorm2d(channels)

    def forward(self, x):
        # spatial attention
        local_w = self.local_att(x) ## local attention
        ## channel attention
        global_w = self.global_att(x)
        mask = self.sigmoid(local_w * global_w)
        masked_feature = mask * x
        output = self.conv_block(masked_feature)
        return output


class Prediction_head(nn.Module):
    '''
    自适应特征连接模块, 用于跳变连接的自适应连接 Adaptive_Connection
    '''
    def __init__(self, channels, img=False):
        super(Prediction_head, self).__init__()

        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        return (self.conv_block(x) + 1) / 2

class SFP(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels, img=False):        
        super(SFP, self).__init__()
        self.mask = S2M(channels[0])
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        x = self.mask(x)
        return (self.conv_block(x) + 1) / 2

class IFP(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels):        
        super(IFP, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, x):
        return (self.conv_block(x) + 1) / 2

class SIM(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):

        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)
        
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        #actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = self.bn(normalized * (1 + gamma)) + beta

        return out

class PSF(nn.Module):

    def __init__(self, n_classes):
        super(PSF, self).__init__()
        self.num_resnet_layers = 34
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)

        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)

        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)

        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)

        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)

        self.dims = [32, 32, 64, 64, 64, 64]
        self.decoder_dim_rec = 32        
        self.decoder_dim_seg = 64
        
        
        ########  Thermal ENCODER  ########
        self.encoder_thermal_conv1 = Feature_extract(1, 64)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer3 = resnet_raw_model1.layer1
        self.encoder_thermal_layer4 = resnet_raw_model1.layer2
        self.encoder_thermal_layer5 = resnet_raw_model1.layer3
        self.encoder_thermal_layer6 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = Feature_extract(3, 64)
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer3 = resnet_raw_model2.layer1
        self.encoder_rgb_layer4 = resnet_raw_model2.layer2
        self.encoder_rgb_layer5 = resnet_raw_model2.layer3
        self.encoder_rgb_layer6 = resnet_raw_model2.layer4
        
        self.high_fuse6 = PSFM(512, 64, 128)
        self.high_fuse5 = PSFM(256, 64, 128)
        self.high_fuse4 = PSFM(128, 64, 128)        
        self.low_fuse3 = SDFM(64, 64)
        self.low_fuse2 = SDFM(64, 32)
        self.low_fuse1 = SDFM(32, 32)
        
        self.SIM1 = SIM(norm_nc=32, label_nc=64, nhidden=32)
        self.SIM2 = SIM(norm_nc=32, label_nc=32, nhidden=32)
        self.to_fused_seg = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, self.decoder_dim_seg, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor = 2 ** i, mode='bilinear', align_corners=True)
        ) for i, dim in enumerate(self.dims[2:])])
        
        self.SIM3 = SIM(norm_nc=32, label_nc=64, nhidden=32)
              
        self.seg_decoder = S2PM(4 * self.decoder_dim_seg, self.decoder_dim_seg)
        self.rec_decoder = DSRM(self.decoder_dim_rec, self.decoder_dim_rec)
        self.seg_rec_decoder = DSRM(self.decoder_dim_rec, self.decoder_dim_rec)        
        self.classfier = S2P2(feature=self.decoder_dim_seg, n_classes=n_classes)
        self.pred_fusion = IFP([self.decoder_dim_rec, 1])   
        self.pred_vi = SFP([self.decoder_dim_rec, 1])                
        self.pred_ir = SFP([self.decoder_dim_rec, 1])

    def forward(self, rgb, depth):
        rgb = rgb
        thermal = depth[:, :1, ...]

        vobose = False

        # encoder
        ######################################################################

        if vobose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if vobose: print("thermal.size() original: ", thermal.size())  # (480, 640)

        ######################################################################

        rgb1, rgb2 = self.encoder_rgb_conv1(rgb)  # (240, 320)
        rgb2 = self.encoder_rgb_bn1(rgb2)  # (240, 320)
        rgb2 = self.encoder_rgb_relu(rgb2) # (240, 320)

        thermal1, thermal2 = self.encoder_thermal_conv1(thermal)  # (240, 320)
        thermal2 = self.encoder_thermal_bn1(thermal2)  # (240, 320)
        thermal2 = self.encoder_thermal_relu(thermal2) # (240, 320)

        ######################################################################
        rgb3 = self.encoder_rgb_maxpool(rgb2)  # (120, 160)
        thermal3 = self.encoder_thermal_maxpool(thermal2) # (120, 160)
        rgb3 = self.encoder_rgb_layer3(rgb3) # (120, 160)
        thermal3 = self.encoder_thermal_layer3(thermal3)   # (120, 160)


        ######################################################################
        rgb4= self.encoder_rgb_layer4(rgb3)  # (60, 80)
        thermal4 = self.encoder_thermal_layer4(thermal3)  # (60, 80)

        ######################################################################
        rgb5 = self.encoder_rgb_layer5(rgb4)  # (30, 40)
        thermal5 = self.encoder_thermal_layer5(thermal4) # (30, 40)

        ######################################################################
        rgb6 = self.encoder_rgb_layer6(rgb5)  # (30, 40)
        thermal6 = self.encoder_thermal_layer6(thermal5) # (30, 40)

       ## fused featrue
       
        fused_f6 = self.high_fuse6(rgb6,thermal6)
        fused_f5 = self.high_fuse5(rgb5,thermal5)
        fused_f4 = self.high_fuse4(rgb4, thermal4)
        fused_f3 = self.low_fuse3(rgb3, thermal3)
        fused_f2 = self.low_fuse2(rgb2, thermal2)
        fused_f1 = self.low_fuse1(rgb1, thermal1)
        
        encoded_featrues_seg = [fused_f3, fused_f4, fused_f5, fused_f6]
        rec_f1 = self.SIM1(fused_f2, fused_f3)
        rec_f = self.SIM2(fused_f1, rec_f1)  
        seg_fused_f = [to_fused(output) for output, to_fused in zip(encoded_featrues_seg, self.to_fused_seg)] 
        seg_f = torch.cat(seg_fused_f, dim=1)
        
        ## sparse scene understanding 
        seg_f = self.seg_decoder(seg_f)
        semantic_out, binary_out, boundary_out = self.classfier(seg_f)
        
        ## image reconstruction
        ## visible image        
        rec_f = self.rec_decoder(rec_f)
        features_seg_rec = self.SIM3(rec_f, seg_f)
        rec_f = self.seg_rec_decoder(features_seg_rec)
                
        vi_img =  self.pred_vi(rec_f)
        ## infrared image
        ir_img =  self.pred_ir(rec_f)
        ## fused image 
        fused_img =  self.pred_fusion(rec_f)
        return semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img



class SDFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(SDFM, self).__init__()
        self.RGBobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.RGBobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)        
        self.RGBspr = BBasicConv2d(out_C, out_C, 3, 1, 1)        

        self.Infobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.Infobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)        
        self.Infspr = BBasicConv2d(out_C, out_C, 3, 1, 1) 
        self.obj_fuse = Fusion_module(channels=out_C)  
        

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj1_2(self.RGBobj1_1(rgb))
        rgb_obj = self.RGBspr(rgb_sum)
        Inf_sum = self.Infobj1_2(self.Infobj1_1(depth))
        Inf_obj = self.Infspr(Inf_sum)
        out = self.obj_fuse(rgb_obj, Inf_obj)
        return out

class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input) ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w) ## 计算特征x1的权重 
        xo = w * x1 + (1 - w) * x2 ## fusion results ## 特征聚合
        return xo

class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K= BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Q = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.INF_K= BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)       
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y):
        Q = self.Q(torch.cat([x,y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width*height)
        RGB_K = RGB_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width*height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height,width)
        RGB_refine = self.gamma1*RGB_refine+y
        
        
        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width*height)
        INF_K = INF_K.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width*height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height,width) 
        INF_refine = self.gamma2 * INF_refine + x
        
        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))        
        return out  

class PSFM(nn.Module):
    def __init__(self, in_C, out_C, cat_C):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(in_C, out_C)
        self.Infobj = DenseLayer(in_C, out_C)           
        self.obj_fuse = GEFM(cat_C, out_C)
        

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj(rgb)        
        Inf_sum = self.Infobj(depth)        
        out = self.obj_fuse(rgb_sum,Inf_sum)
        return out


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        # print(down_feats.shape)
        # print(self.denseblock)
        out_feats = []
        for i in self.denseblock:
            # print(self.denseblock)
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            # print(feats.shape)
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

#########################################################################################################    Inception


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU6(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x





#########################################################################################################     decoder


class decoder(nn.Module):
    def __init__(self, channel=64):
        super(decoder, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = x3 + x
        out = self.up2(out)
        return out



class S2PM(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(S2PM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        out = self.block3(x2)
        return out

class DSRM(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(DSRM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(3 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
        self.block4 = nn.Sequential(
            BasicConv2d(4 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(torch.cat([x, x1], dim=1))
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))
        return out

class S2P2(nn.Module):
    '''This path plays the role of a classifier and is responsible for predicting the results of semantic segmentation, binary segmentation and edge segmentation'''
    def __init__(self, feature=64, n_classes=9):
        super(S2P2, self).__init__()
        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(feature // 4, 2, kernel_size=3, padding=1)

        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(feature, n_classes, kernel_size=3, padding=1)

        self.boundary_conv1 = ConvBNReLU(feature * 2, feature, kernel_size=1)
        self.boundary_conv2 = nn.Conv2d(feature, 2, kernel_size=3, padding=1)

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(feature * 2, feature, kernel_size=1),
            nn.BatchNorm2d(feature),
            nn.ReLU6(inplace=True),
            nn.Conv2d(feature, 2, kernel_size=3, padding=1),
        )

        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat):
        binary = self.binary_conv2(self.binary_conv1(feat))
        binary_out = self.up4x(binary)

        weight = torch.exp(binary)
        weight = weight[:, 1:2, :, :] / torch.sum(weight, dim=1, keepdim=True)

        feat_sematic = self.up2x(feat * weight)
        feat_sematic = self.semantic_conv1(feat_sematic)

        semantic_out = self.semantic_conv2(feat_sematic)
        semantic_out = self.up2x(semantic_out)

        feat_boundary = torch.cat([feat_sematic, self.up2x(feat)], dim=1)
        boundary_out = self.boundary_conv(feat_boundary)
        boundary_out = self.up2x(boundary_out)
        return semantic_out, binary_out, boundary_out

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
