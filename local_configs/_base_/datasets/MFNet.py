# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = './datasets/MFNet'
input_type = 'PSFusion'
if input_type == 'PSFusion': 
    img_norm_cfg = dict(mean=[58.2, 67.06, 59.0], std=[47.01, 47.21, 46.45], to_rgb=True)
elif input_type == 'RGB': 
    img_norm_cfg = dict(mean=[58.51, 67.42, 59.23], std=[42.62, 43.36, 43.05], to_rgb=True)
elif input_type == 'Thermal': 
    img_norm_cfg = dict(mean=[23.45, 23.45, 23.45], std=[17.25, 17.25, 17.25], to_rgb=True)
elif input_type == 'SeAFusion': 
    img_norm_cfg = dict(mean=[62.71, 71.62, 63.44], std=[42.1, 42.47, 42.57], to_rgb=True)
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