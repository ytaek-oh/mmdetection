import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmcv.utils.path import check_file_exist

from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class SideWalkBBoxDataset(XMLDataset):
    CLASSES = ('barricade', 'bench', 'bicycle', 'bollard', 'bus', 'car',
               'carrier', 'cat', 'chair', 'dog', 'fire_hydrant', 'kiosk',
               'motorcycle', 'movable_signage', 'parking_meter', 'person',
               'pole', 'potted_plant', 'power_controller', 'scooter', 'stop',
               'stroller', 'table', 'traffic_light',
               'traffic_light_controller', 'traffic_sign', 'tree_trunk',
               'truck', 'wheelchair')

    def __init__(self, **kwargs):
        super(SideWalkBBoxDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        img_infos = []
        tree = ET.parse(ann_file)
        root = tree.getroot()
        images = root.findall('image')
        for image in images:
            attrib = image.attrib

            img_id = int(attrib['id'])
            filename = attrib['name']
            check_file_exist(osp.join(self.img_prefix, filename))

            width = int(attrib['width'])
            height = int(attrib['height'])
            img_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann_file=ann_file))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_file = self.img_infos[idx]['ann_file']

        tree = ET.parse(ann_file)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        image = root.find('image[@id="{}"]'.format(img_id))
        assert image is not None

        for box in image.findall('box'):
            attrib = box.attrib
            name = attrib['label']
            label = self.cat2label[name]
            bbox = [
                float(attrib['xtl']),
                float(attrib['ytl']),
                float(attrib['xbr']),
                float(attrib['ybr'])
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
