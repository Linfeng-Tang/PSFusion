# dataset settings

dataset_type = 'PascalVOCDataset'
data_root = '/data/timer/Segmentation/SegNext/datasets/MSRS'  ## Here you need to modify according to the actual path
input_type = 'PSFusion'
if input_type == 'SeAFusion':
    img_norm_cfg = dict(
        mean=[65.03, 73.54, 64.52], std=[43.26, 43.44, 43.79], to_rgb=True)
elif input_type == 'PSFusion': 
    img_norm_cfg = dict(mean=[57.34, 65.78, 56.76], std=[46.75, 46.9, 46.38], to_rgb=True)
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
 