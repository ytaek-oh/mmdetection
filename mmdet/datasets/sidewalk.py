import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class SideWalkDataset(CocoDataset):

    CLASSES = ('barricade', 'bench', 'bicycle', 'bollard', 'bus', 'car',
               'carrier', 'cat', 'chair', 'dog', 'fire_hydrant', 'kiosk',
               'motorcycle', 'movable_signage', 'parking_meter', 'person',
               'pole', 'potted_plant', 'power_controller', 'scooter', 'stop',
               'stroller', 'table', 'traffic_light',
               'traffic_light_controller', 'traffic_sign', 'tree_trunk',
               'truck', 'wheelchair')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = osp.join(info['img_prefix'], info['file_name'])
            img_infos.append(info)
        return img_infos

    def show_annotations(self, img_id, show=False, out_file=None, **kwargs):
        img_info = self.img_infos[img_id]
        img_path = osp.join(self.img_prefix, img_info['filename'])
        img = mmcv.imread(img_path)
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))

        bboxes = []
        labels = []
        class_names = ['bg'] + list(self.CLASSES)
        for ann in annotations:
            if len(ann['segmentation']) > 0:
                rle = maskUtils.frPyObjects(ann['segmentation'],
                                            img_info['height'],
                                            img_info['width'])
                ann_mask = np.sum(
                    maskUtils.decode(rle), axis=2).astype(np.bool)
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                img[ann_mask] = img[ann_mask] * 0.5 + color_mask * 0.5
            bbox = ann['bbox']
            x, y, w, h = bbox
            bboxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=class_names,
            show=show,
            out_file=out_file,
            **kwargs)
        if not (show or out_file):
            return img
