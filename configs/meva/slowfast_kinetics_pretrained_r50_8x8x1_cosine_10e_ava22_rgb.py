model = dict(
    type='FasterRCNN',
    # backbone=dict(
    #     type='ResNet3dSlowFast',
    #     pretrained=None,
    #     resample_rate=4,
    #     speed_ratio=4,
    #     channel_ratio=8,
    #     slow_pathway=dict(
    #         type='resnet3d',
    #         depth=50,
    #         pretrained=None,
    #         lateral=True,
    #         fusion_kernel=7,
    #         conv1_kernel=(1, 7, 7),
    #         dilations=(1, 1, 1, 1),
    #         conv1_stride_t=1,
    #         pool1_stride_t=1,
    #         inflate=(0, 0, 1, 1),
    #         spatial_strides=(1, 2, 2, 1)),
    #     fast_pathway=dict(
    #         type='resnet3d',
    #         depth=50,
    #         pretrained=None,
    #         lateral=False,
    #         base_channels=8,
    #         conv1_kernel=(5, 7, 7),
    #         conv1_stride_t=1,
    #         pool1_stride_t=1,
    #         spatial_strides=(1, 2, 2, 1))),
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True),
    rpn_head=dict(
        type='AVARPNHead',
        in_channels=768,
        feat_channels=768,
        anchor_generator=dict(
            type='AnchorGenerator',
            # scales=[2, 4, 8, 16, 32],
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            # strides=[16]),
            strides=[2]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            dropout_ratio=0.5,
            # in_channels=2304,
            in_channels=768,
            num_classes=3,
            multilabel=True,
            topk=(1,))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=8,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms=dict(type='nms', iou_threshold=0.7),
            nms_pre=6000,
            max_per_img=1000,
            min_bbox_size=0),
        rcnn=dict(action_thr=0.002)))

dataset_type = 'MEVADataset'
data_root = '/data/meva_data/meva_frames'
data_root_val = '/data1/tzh/MEVA_frame'
ann_file_train = '/data/meva_data/clip_json/detectobject.json'
ann_file_val = '/data/meva_data/clip_json/detectobject.json'
ann_file_test = '/data/meva_data/clip_json/meva_cascade_persononly_val_allframes_det.bbox.json'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleMEVAFrames', clip_len=8, frame_interval=4),
    dict(type='RawFrameDecode'),
    # dict(type='RandomRescale', scale_range=(256, 320)),
    # dict(type='RandomCrop', size=256),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=['scores', 'pad_shape', 'img_shape', 'scale_factor'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='SampleMEVAFrames', clip_len=8, frame_interval=4),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape'],
        nested=True)
]

data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        person_det_score_thr=0.9,
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        person_det_score_thr=0.9,
        data_prefix=data_root))
data['test'] = data['val']
# optimizer
optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=0.1)
total_epochs = 10
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1)
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb'  # noqa: E501
load_from = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
resume_from = None
find_unused_parameters = False
