# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/data/timer/Segmentation/SegNext/datasets/TNO'
input_type = 'U2Fusion'
if input_type == 'SeAFusion': 
    img_norm_cfg = dict(mean=[107.04, 107.04, 107.04], std=[42.17, 42.17, 42.17], to_rgb=True)
elif input_type == 'RGB': 
    img_norm_cfg = dict(mean=[70.6, 70.6, 70.6], std=[38.49, 38.49, 38.49], to_rgb=True)
elif input_type == 'Thermal': 
    img_norm_cfg = dict(mean=[102.74, 102.74, 102.74], std=[53.34, 53.34, 53.34], to_rgb=True)
elif input_type == 'U2Fusion': 
    img_norm_cfg = dict(mean=[133.39, 133.39, 133.39], std=[38.8, 38.8, 38.8], to_rgb=True)
elif input_type == 'PSFusion': 
    img_norm_cfg = dict(mean=[119.07, 119.07, 119.07], std=[49.05, 49.05, 49.05], to_rgb=True)
elif input_type == 'TarDAL': 
    img_norm_cfg = dict(mean=[114.2, 114.2, 114.2], std=[46.93, 46.93, 46.93], to_rgb=True)
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
        split='split/show.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=input_type,
        ann_dir='Label',
        split='split/show.txt',
        pipeline=test_pipeline))