# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/Lightning-AI/metrics/blob/1c42f6643f9241089e55a4d899f14da480021e16/torchmetrics/detection/map.py
# Current versions of MAP in torchmetrics are incorrect or slow as of 9/21/22.
# Relevant issues:
# https://github.com/Lightning-AI/metrics/issues/1024
# https://github.com/Lightning-AI/metrics/issues/1164

"""MAP torchmetric for object detection."""
#type: ignore
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchvision.ops import box_convert

from composer.utils import MissingConditionalImportError

__all__ = ['MAP']

log = logging.getLogger(__name__)


@dataclass
class MAPMetricResults:
    """Dataclass to wrap the final mAP results."""
    map: Tensor
    map_50: Tensor
    map_75: Tensor
    map_small: Tensor
    map_medium: Tensor
    map_large: Tensor
    mar_1: Tensor
    mar_10: Tensor
    mar_100: Tensor
    mar_small: Tensor
    mar_medium: Tensor
    mar_large: Tensor
    map_per_class: Tensor
    mar_100_per_class: Tensor

    def __getitem__(self, key: str) -> Union[Tensor, List[Tensor]]:
        return getattr(self, key)


# noinspection PyMethodMayBeStatic
class WriteToLog:
    """Logging class to move logs to log.debug()."""

    def write(self, buf: str) -> None:  # skipcq: PY-D0003, PYL-R0201
        for line in buf.rstrip().splitlines():
            log.debug(line.rstrip())

    def flush(self) -> None:  # skipcq: PY-D0003, PYL-R0201
        for handler in log.handlers:
            handler.flush()

    def close(self) -> None:  # skipcq: PY-D0003, PYL-R0201
        for handler in log.handlers:
            handler.close()


class _hide_prints:
    """Internal helper context to suppress the default output of the pycocotools package."""

    def __init__(self) -> None:
        self._original_stdout = None

    def __enter__(self) -> None:
        self._original_stdout = sys.stdout  # type: ignore
        sys.stdout = WriteToLog()  # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        sys.stdout.close()
        sys.stdout = self._original_stdout  # type: ignore


def _input_validator(preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
    """Ensure the correct input format of `preds` and `targets`."""
    if not isinstance(preds, Sequence):
        raise ValueError('Expected argument `preds` to be of type List')
    if not isinstance(targets, Sequence):
        raise ValueError('Expected argument `target` to be of type List')
    if len(preds) != len(targets):
        raise ValueError('Expected argument `preds` and `target` to have the same length')

    for k in ['boxes', 'scores', 'labels']:
        if any(k not in p for p in preds):
            raise ValueError(f'Expected all dicts in `preds` to contain the `{k}` key')

    for k in ['boxes', 'labels']:
        if any(k not in p for p in targets):
            raise ValueError(f'Expected all dicts in `target` to contain the `{k}` key')

    if any(type(pred['boxes']) is not torch.Tensor for pred in preds):
        raise ValueError('Expected all boxes in `preds` to be of type torch.Tensor')
    if any(type(pred['scores']) is not torch.Tensor for pred in preds):
        raise ValueError('Expected all scores in `preds` to be of type torch.Tensor')
    if any(type(pred['labels']) is not torch.Tensor for pred in preds):
        raise ValueError('Expected all labels in `preds` to be of type torch.Tensor')
    if any(type(target['boxes']) is not torch.Tensor for target in targets):
        raise ValueError('Expected all boxes in `target` to be of type torch.Tensor')
    if any(type(target['labels']) is not torch.Tensor for target in targets):
        raise ValueError('Expected all labels in `target` to be of type torch.Tensor')

    for i, item in enumerate(targets):
        if item['boxes'].size(0) != item['labels'].size(0):
            raise ValueError(
                f'Input boxes and labels of sample {i} in targets have a'
                f" different length (expected {item['boxes'].size(0)} labels, got {item['labels'].size(0)})")
    for i, item in enumerate(preds):
        if item['boxes'].size(0) != item['labels'].size(0) != item['scores'].size(0):
            raise ValueError(f'Input boxes, labels and scores of sample {i} in preds have a'
                             f" different length (expected {item['boxes'].size(0)} labels and scores,"
                             f" got {item['labels'].size(0)} labels and {item['scores'].size(0)})")


class MAP(Metric):
    """Computes the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) for object detection predictions.

    Optionally, the mAP and mAR values can be calculated per class.
    Predicted boxes and targets have to be in Pascal VOC format \
    (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
    See the :meth:`update` method for more information about the input format to this metric.
    See `this blog <https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173>`_ for more details on (mAP)
    and (mAR).

    .. warning:: This metric is a wrapper for the `pycocotools <https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools>`_,
        which is a standard implementation for the mAP metric for object detection. Using this metric
        therefore requires you to have `pycocotools` installed. Please install with ``pip install pycocotools``

    .. warning:: As the pycocotools library cannot deal with tensors directly, all results have to be transfered
        to the CPU, this may have an performance impact on your training.

    Args:
        class_metrics (bool, optional): Option to enable per-class metrics for mAP and mAR_100. Has a performance impact. Default: ``False``.
        compute_on_step (bool, optional): Forward only calls ``update()`` and return ``None`` if this is set to ``False``. Default: ``False``.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at each ``forward()`` before returning the value at the step. Default: ``False``.
        process_group (any, optional): Specify the process group on which synchronization is called. Default: ``None`` (which selects the entire world).
        dist_sync_fn (callable, optional): Callback that performs the allgather operation on the metric state. When ``None``, DDP will be used to perform the all_gather. Default: ``None``.

    Raises:
        ValueError: If ``class_metrics`` is not a boolean.
    """

    # Have default behavior for this complex metric
    full_state_update = True

    def __init__(
            self,
            class_metrics: bool = False,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,  # type: ignore
    ) -> None:  # type: ignore
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='coco', conda_package='pycocotools') from e

        self.COCO = COCO
        self.COCOeval = COCOeval

        if not isinstance(class_metrics, bool):
            raise ValueError('Expected argument `class_metrics` to be a boolean')
        self.class_metrics = class_metrics

        self.add_state('detection_boxes', default=[], dist_reduce_fx=None)
        self.add_state('detection_scores', default=[], dist_reduce_fx=None)
        self.add_state('detection_labels', default=[], dist_reduce_fx=None)
        self.add_state('groundtruth_boxes', default=[], dist_reduce_fx=None)
        self.add_state('groundtruth_labels', default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and groundtruth to the metric.

        Args:
            preds (list[Dict[str, ~torch.Tensor]]): A list of dictionaries containing the key-values:

                    ``boxes`` (torch.FloatTensor): [num_boxes, 4] predicted boxes of the format [xmin, ymin, xmax, ymax] in absolute image coordinates.

                    ``scores`` (torch.FloatTensor): of shape [num_boxes] containing detection scores for the boxes.

                    ``labels`` (torch.IntTensor): of shape [num_boxes] containing 0-indexed detection classes for the boxes.

            target (list[Dict[str, ~torch.Tensor]]): A list of dictionaries containing the key-values:

                    ``boxes`` (torch.FloatTensor): [num_boxes, 4] ground truth boxes of the format [xmin, ymin, xmax, ymax] in absolute image coordinates.

                    ``labels`` (torch.IntTensor): of shape [num_boxes] containing 1-indexed groundtruth classes for the boxes.

        Raises:
            ValueError: If ``preds`` and ``target`` are not of the same length.
            ValueError: If any of ``preds.boxes``, ``preds.scores`` and ``preds.labels`` are not of the same length.
            ValueError: If any of ``target.boxes`` and ``target.labels`` are not of the same length.
            ValueError: If any box is not type float and of length 4.
            ValueError: If any class is not type int and of length 1.
            ValueError: If any score is not type float and of length 1.
        """
        _input_validator(preds, target)

        for item in preds:
            self.detection_boxes.append(item['boxes'])  # type: ignore
            self.detection_scores.append(item['scores'])  # type: ignore
            self.detection_labels.append(item['labels'])  # type: ignore

        for item in target:
            self.groundtruth_boxes.append(item['boxes'])  # type: ignore
            self.groundtruth_labels.append(item['labels'])  # type: ignore

    def compute(self) -> dict:
        """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) scores.

        All detections added in the ``update()`` method are included.

        Note:
            Main `map` score is calculated with @[ IoU=0.50:0.95 | area=all | maxDets=100 ]

        Returns:
            MAPMetricResults (dict): containing:

                ``map`` (torch.Tensor): map at 95 iou.

                ``map_50`` (torch.Tensor): map at 50 iou.

                ``map_75`` (torch.Tensor): map at 75 iou.

                ``map_small`` (torch.Tensor): map at 95 iou for small objects.

                ``map_medium`` (torch.Tensor): map at 95 iou for medium objects.

                ``map_large`` (torch.Tensor): map at 95 iou for large objects.

                ``mar_1`` (torch.Tensor): mar at 1 max detection.

                ``mar_10`` (torch.Tensor): mar at 10 max detections.

                ``mar_100`` (torch.Tensor): mar at 100 max detections.

                ``mar_small`` (torch.Tensor): mar at 100 max detections for small objects.

                ``mar_medium`` (torch.Tensor): mar at 100 max detections for medium objects.

                ``mar_large`` (torch.Tensor): mar at 100 max detections for large objects.

                ``map_per_class`` (torch.Tensor) (-1 if class metrics are disabled): map value for each class.

                ``mar_100_per_class`` (torch.Tensor) (-1 if class metrics are disabled): mar at 100 detections for each class.
        """
        coco_target, coco_preds = self.COCO(), self.COCO()  # type: ignore
        coco_target.dataset = self._get_coco_format(self.groundtruth_boxes, self.groundtruth_labels)  # type: ignore
        coco_preds.dataset = self._get_coco_format(  # type: ignore
            self.detection_boxes,  # type: ignore
            self.detection_labels,  # type: ignore
            self.detection_scores)  # type: ignore

        with _hide_prints():
            coco_target.createIndex()
            coco_preds.createIndex()
            coco_eval = self.COCOeval(coco_target, coco_preds, 'bbox')  # type: ignore
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats

        map_per_class_values: Tensor = torch.Tensor([-1])
        mar_100_per_class_values: Tensor = torch.Tensor([-1])
        # if class mode is enabled, evaluate metrics per class
        if self.class_metrics:
            map_per_class_list = []
            mar_100_per_class_list = []
            for class_id in torch.cat(self.detection_labels +
                                      self.groundtruth_labels).unique().cpu().tolist():  # type: ignore
                coco_eval.params.catIds = [class_id]
                with _hide_prints():
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    class_stats = coco_eval.stats

                map_per_class_list.append(torch.Tensor([class_stats[0]]))
                mar_100_per_class_list.append(torch.Tensor([class_stats[8]]))
            map_per_class_values = torch.Tensor(map_per_class_list)
            mar_100_per_class_values = torch.Tensor(mar_100_per_class_list)

        metrics = MAPMetricResults(
            map=torch.Tensor([stats[0]]),
            map_50=torch.Tensor([stats[1]]),
            map_75=torch.Tensor([stats[2]]),
            map_small=torch.Tensor([stats[3]]),
            map_medium=torch.Tensor([stats[4]]),
            map_large=torch.Tensor([stats[5]]),
            mar_1=torch.Tensor([stats[6]]),
            mar_10=torch.Tensor([stats[7]]),
            mar_100=torch.Tensor([stats[8]]),
            mar_small=torch.Tensor([stats[9]]),
            mar_medium=torch.Tensor([stats[10]]),
            mar_large=torch.Tensor([stats[11]]),
            map_per_class=map_per_class_values,
            mar_100_per_class=mar_100_per_class_values,
        )
        return metrics.__dict__

    def _get_coco_format(self,
                         boxes: List[torch.Tensor],
                         labels: List[torch.Tensor],
                         scores: Optional[List[torch.Tensor]] = None) -> Dict:
        """Transforms and returns all cached targets or predictions in COCO format.

        Format is defined at https://cocodataset.org/#format-data.
        """
        images = []
        annotations = []
        annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

        boxes = [box_convert(box, in_fmt='xyxy', out_fmt='xywh') for box in boxes]  # type: ignore
        for image_id, (image_boxes, image_labels) in enumerate(zip(boxes, labels)):
            image_boxes = image_boxes.cpu().tolist()
            image_labels = image_labels.cpu().tolist()

            images.append({'id': image_id})
            for k, (image_box, image_label) in enumerate(zip(image_boxes, image_labels)):
                if len(image_box) != 4:
                    raise ValueError(
                        f'Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})'
                    )

                if type(image_label) != int:
                    raise ValueError(f'Invalid input class of sample {image_id}, element {k}'
                                     f' (expected value of type integer, got type {type(image_label)})')

                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'bbox': image_box,
                    'category_id': image_label,
                    'area': image_box[2] * image_box[3],
                    'iscrowd': 0,
                }
                if scores is not None:  # type: ignore
                    score = scores[image_id][k].cpu().tolist()  # type: ignore
                    if type(score) != float:  # type: ignore
                        raise ValueError(f'Invalid input score of sample {image_id}, element {k}'
                                         f' (expected value of type float, got type {type(score)})')
                    annotation['score'] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{
            'id': i,
            'name': str(i)
        } for i in torch.cat(self.detection_labels + self.groundtruth_labels).unique().cpu().tolist()]  # type: ignore
        return {'images': images, 'annotations': annotations, 'categories': classes}
