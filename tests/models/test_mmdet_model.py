# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch


@pytest.fixture
def mmdet_detection_batch():
    batch_size = 2
    num_labels_per_image = 20
    image_size = 224
    return {
        'img_metas': [{
            'filename': '../../data/coco/train2017/fake_img.jpg',
            'ori_filename': 'fake_image.jpg',
            'img_shape': (image_size, image_size, 3),
            'ori_shape': (image_size, image_size, 3),
            'pad_shape': (image_size, image_size, 3),
            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32)
        }] * batch_size,
        'img':
            torch.zeros(batch_size, 3, image_size, image_size, dtype=torch.float32),
        'gt_bboxes': [torch.zeros(num_labels_per_image, 4, dtype=torch.float32)] * batch_size,
        'gt_labels': [torch.zeros(num_labels_per_image, dtype=torch.int64)] * batch_size
    }


@pytest.fixture
def mmdet_detection_eval_batch():
    # Eval settings for mmdetection datasets have an extra list around inputs.
    batch_size = 2
    num_labels_per_image = 20
    image_size = 224
    return {
        'img_metas': [[{
            'filename': '../../data/coco/train2017/fake_img.jpg',
            'ori_filename': 'fake_image.jpg',
            'img_shape': (image_size, image_size, 3),
            'ori_shape': (image_size, image_size, 3),
            'pad_shape': (image_size, image_size, 3),
            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
        }] * batch_size],
        'img': [torch.zeros(batch_size, 3, image_size, image_size, dtype=torch.float32)],
        'gt_bboxes': [[torch.zeros(num_labels_per_image, 4, dtype=torch.float32)] * batch_size],
        'gt_labels': [[torch.zeros(num_labels_per_image, dtype=torch.int64)] * batch_size]
    }


@pytest.fixture
def yolox_config():
    # from https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/yolox_s_8x8_300e_coco.py
    return dict(
        type='YOLOX',
        input_size=(640, 640),
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
        neck=dict(type='YOLOXPAFPN', in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
        bbox_head=dict(type='YOLOXHead', num_classes=80, in_channels=128, feat_channels=128),
        train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        # In order to align the source code, the threshold of the val phase is
        # 0.01, and the threshold of the test phase is 0.001.
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))


@pytest.fixture
def faster_rcnn_config():
    # modified from https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/models/faster_rcnn_r50_fpn.py
    return dict(
        type='FasterRCNN',
        backbone=dict(type='ResNet',
                      depth=50,
                      num_stages=4,
                      out_indices=(0, 1, 2, 3),
                      frozen_stages=1,
                      norm_cfg=dict(type='BN', requires_grad=True),
                      norm_eval=True,
                      style='pytorch'),
        neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
        rpn_head=dict(type='RPNHead',
                      in_channels=256,
                      feat_channels=256,
                      anchor_generator=dict(type='AnchorGenerator',
                                            scales=[8],
                                            ratios=[0.5, 1.0, 2.0],
                                            strides=[4, 8, 16, 32, 64]),
                      bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                      target_means=[.0, .0, .0, .0],
                                      target_stds=[1.0, 1.0, 1.0, 1.0]),
                      loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                      loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(type='StandardRoIHead',
                      bbox_roi_extractor=dict(type='SingleRoIExtractor',
                                              roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                              out_channels=256,
                                              featmap_strides=[4, 8, 16, 32]),
                      bbox_head=dict(type='Shared2FCBBoxHead',
                                     in_channels=256,
                                     fc_out_channels=1024,
                                     roi_feat_size=7,
                                     num_classes=80,
                                     bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                                     target_means=[0., 0., 0., 0.],
                                                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                                     reg_class_agnostic=False,
                                     loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                                     loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(rpn=dict(assigner=dict(type='MaxIoUAssigner',
                                              pos_iou_thr=0.7,
                                              neg_iou_thr=0.3,
                                              min_pos_iou=0.3,
                                              match_low_quality=True,
                                              ignore_iof_thr=-1),
                                sampler=dict(type='RandomSampler',
                                             num=256,
                                             pos_fraction=0.5,
                                             neg_pos_ub=-1,
                                             add_gt_as_proposals=False),
                                allowed_border=-1,
                                pos_weight=-1,
                                debug=False),
                       rpn_proposal=dict(nms_pre=2000,
                                         max_per_img=1000,
                                         nms=dict(type='nms', iou_threshold=0.7),
                                         min_bbox_size=0),
                       rcnn=dict(assigner=dict(type='MaxIoUAssigner',
                                               pos_iou_thr=0.5,
                                               neg_iou_thr=0.5,
                                               min_pos_iou=0.5,
                                               match_low_quality=False,
                                               ignore_iof_thr=-1),
                                 sampler=dict(type='RandomSampler',
                                              num=512,
                                              pos_fraction=0.25,
                                              neg_pos_ub=-1,
                                              add_gt_as_proposals=True),
                                 pos_weight=-1,
                                 debug=False)),
        test_cfg=dict(
            rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)
            # soft-nms is also supported for rcnn testing
            # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
        ))


def test_mmdet_model_forward_yolox(mmdet_detection_batch, yolox_config):
    pytest.importorskip('mmdet')

    from mmcv import ConfigDict
    from mmdet.models import build_detector

    from composer.models import MMDetModel

    config = ConfigDict(yolox_config)
    # non pretrained model to avoid a slow test that downloads the weights.
    model = build_detector(config)
    model.init_weights()
    model = MMDetModel(model=model)
    out = model(mmdet_detection_batch)
    assert list(out.keys()) == ['loss_cls', 'loss_bbox', 'loss_obj']


def test_mmdet_model_eval_forward_yolox(mmdet_detection_eval_batch, yolox_config):
    pytest.importorskip('mmdet')

    from mmcv import ConfigDict
    from mmdet.models import build_detector

    from composer.models import MMDetModel

    config = ConfigDict(yolox_config)
    # non pretrained model to avoid a slow test that downloads the weights.
    model = build_detector(config)
    model.init_weights()
    model = MMDetModel(model=model)
    out = model.eval_forward(mmdet_detection_eval_batch)
    assert len(out) == mmdet_detection_eval_batch['img'][0].shape[0]  # batch size
    assert list(out[0].keys()) == ['labels', 'boxes', 'scores']


def test_mmdet_model_forward_faster_rcnn(mmdet_detection_batch, faster_rcnn_config):
    pytest.importorskip('mmdet')

    from mmcv import ConfigDict
    from mmdet.models import build_detector

    from composer.models import MMDetModel

    config = ConfigDict(faster_rcnn_config)

    # non pretrained model to avoid a slow test that downloads the weights.
    model = build_detector(config)
    model.init_weights()
    model = MMDetModel(model=model)
    out = model(mmdet_detection_batch)
    assert list(out.keys()) == ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox']
