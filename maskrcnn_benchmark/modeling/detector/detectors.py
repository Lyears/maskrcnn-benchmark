# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)


def build_detection_model_with_filter(cfg, category_num):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    result = meta_arch(cfg)
    for i in range(len(result)):
        selected_ids = result[i].extra_fields['labels'] == category_num
        result[i].bbox[selected_ids] = 0
        result[i].extra_fields['scores'][selected_ids] = 0
        result[i].extra_fields['mask'][selected_ids] = 0
    return result
