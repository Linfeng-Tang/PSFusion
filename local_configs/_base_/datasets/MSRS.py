# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = './datasets/MSRS'
input_type = 'PSFusion'
if input_type == 'RFN-Nest': 
    img_norm_cfg = dict(mean=[37.8679, 46.3265, 37.4203], std=[29.9532, 29.6859, 29.8964], to_rgb=True)
elif input_type == 'DIDFuse': 
    img_norm_cfg = dict(mean=[15.7897, 21.9346, 17.2476], std=[30.2497, 30.2504, 29.2938], to_rgb=True)
elif input_type == 'SDNet': 
    img_norm_cfg = dict(mean=[117.37, 121.76, 121.31], std=[44.31, 42.24, 41.06], to_rgb=True)
elif input_type == 'Thermal': 
    img_norm_cfg = dict(mean=[23.3697, 23.3697, 23.3697], std=[17.2976, 17.2976, 17.2976], to_rgb=True)
elif input_type == 'SeAFusion': 
    img_norm_cfg = dict(mean=[61.8184, 70.3214, 61.1608], std=[42.1158, 42.4317, 42.6605], to_rgb=True)
elif input_type == 'GTF': 
    img_norm_cfg = dict(mean=[24.326, 24.326, 24.326], std=[19.812, 19.812, 19.812], to_rgb=True)
elif input_type == 'RGB': 
    img_norm_cfg = dict(mean=[57.4817, 65.9765, 56.8101], std=[42.6123, 43.3069, 43.0446], to_rgb=True)
elif input_type == 'FusionGAN': 
    img_norm_cfg = dict(mean=[37.5174, 46.0014, 37.2713], std=[19.3892, 17.3636, 19.8513], to_rgb=True)
elif input_type == 'U2Fusion': 
    img_norm_cfg = dict(mean=[29.359, 38.7819, 29.0997], std=[28.7129, 28.3591, 28.5458], to_rgb=True)
elif input_type == 'MST-SR': 
    img_norm_cfg = dict(mean=[57.1535, 65.6079, 56.5315], std=[42.4231, 42.9403, 42.8978], to_rgb=True)
elif input_type == 'UMF-CMGR': 
    img_norm_cfg = dict(mean=[29.5385, 37.7764, 29.7331], std=[25.7347, 24.3688, 25.6476], to_rgb=True)
elif input_type == 'TarDAL': 
    img_norm_cfg = dict(mean=[37.618, 44.992, 37.797], std=[42.199, 42.798, 40.982], to_rgb=True)
elif input_type == 'SwinFusion': 
    img_norm_cfg = dict(mean=[54.1082, 61.3798, 53.7709], std=[48.0233, 48.8413, 48.2514], to_rgb=True)
elif input_type == 'MPF': 
    img_norm_cfg = dict(mean=[57.569, 66.04, 56.949], std=[44.437, 44.728, 44.164], to_rgb=True)    
elif input_type == 'PSFusion': 
    img_norm_cfg = dict(mean=[57.34, 65.78, 56.76], std=[46.75, 46.9, 46.38], to_rgb=True)
elif input_type == 'PSFusion_S2P2': 
    img_norm_cfg = dict(mean=[51.79, 60.23, 51.23], std=[43.32, 43.48, 42.99], to_rgb=True)
elif input_type == 'PSFusion_PSFM': 
    img_norm_cfg = dict(mean=[55.57, 64.06, 54.92], std=[40.22, 40.31, 40.12], to_rgb=True)
crop_size = (640, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 480), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=input_type,
        ann_dir='Label',
        split='split/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=input_type,
        ann_dir='Label',
        split='split/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=input_type,
        ann_dir='Label',
        split='split/test.txt',
        pipeline=test_pipeline))