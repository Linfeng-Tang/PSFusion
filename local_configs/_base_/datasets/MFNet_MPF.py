# dataset settings

dataset_type = 'PascalVOCDataset'
data_root = '/data/timer/Segmentation/SegFormer/tools/data/VOCdevkit/VOC2012/ImageSets'
img_norm_cfg = dict(
    mean=[57.57541018, 66.0416688,  56.85973653], std=[41.69555427, 41.7460047,  41.87265185], to_rgb=True)
crop_size = (480, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(480, 640), ratio_range=(0.5, 2.0)),
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
        img_scale=(480, 640),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='MPF',
        ann_dir='SegmentationClass',
        split='Segmentation/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='MPF',
        ann_dir='SegmentationClass',
        split='Segmentation/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='MPF',
        ann_dir='SegmentationClass',
        split='Segmentation/test.txt',
        pipeline=test_pipeline))
 