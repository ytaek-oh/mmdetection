from __future__ import division
import argparse
import glob
import os.path as osp
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mmcv
import numpy as np
from pycocotools import mask as maskUtils

random.seed(100)
np.random.seed(100)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset Splitter for NIA Sidewalk Dataset')
    parser.add_argument(
        'imagefolder_location',
        help='Path where downloaded images are located')
    parser.add_argument(
        '--dataset_type',
        choices=['bbox', 'polygon'],
        help='target dataset to convert')
    args = parser.parse_args()
    return args


class XMLProcessor():

    def __init__(self, xml_path, dataset_type):
        self.xml_path = xml_path
        self.dataset_type = dataset_type

        self.CLASSES = ('barricade', 'bench', 'bicycle', 'bollard', 'bus',
                        'car', 'carrier', 'cat', 'chair', 'dog',
                        'fire_hydrant', 'kiosk', 'motorcycle',
                        'movable_signage', 'parking_meter', 'person', 'pole',
                        'potted_plant', 'power_controller', 'scooter', 'stop',
                        'stroller', 'table', 'traffic_light',
                        'traffic_light_controller', 'traffic_sign',
                        'tree_trunk', 'truck', 'wheelchair')
        self.img_prefix = osp.dirname(xml_path)
        self.img_infos = self._load_img_infos()
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = 0

    def _load_img_infos(self):
        # gather image info from xml annotation
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        images = root.findall('image')
        img_infos = []
        img_ids = []
        filenames = []
        for image in images:
            attrib = image.attrib
            img_id_xml = int(attrib['id'])
            width = int(attrib['width'])
            height = int(attrib['height'])
            filename = attrib['name']

            image_dict = dict(
                id_local=img_id_xml,
                img_prefix=osp.basename(self.img_prefix),
                file_name=filename,
                width=width,
                height=height)
            img_infos.append(image_dict)

            img_ids.append(img_id_xml)
            filenames.append(filename)
        assert len(img_infos) == len(set(img_ids)) == len(set(filenames))

        return img_infos

    def parse_annotations(self, img_id, dataset_type, img_size=None):
        labels = []
        annotations = []
        if dataset_type == 'bbox':
            dataset_type = 'box'
            bboxes = [] # noqa
        else:
            points = []
            groups = []

        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        image = root.find('image[@id="{}"]'.format(img_id))
        for ann_xml in image.findall(dataset_type):
            attrib = ann_xml.attrib
            label = self.cat2label[attrib['label']]

            if dataset_type == 'polygon':
                _point = attrib['points'].replace(';', ',').split(',')
                group_id = int(attrib.get('group_id', '-1'))
                point = []
                for pt in _point:
                    pt = float('{:.2f}'.format(float(pt)))
                    point.append(pt)
                assert len(point) == len(_point)

                points.append(point)
                groups.append(group_id)
                labels.append(label)
            else:
                pass

        if dataset_type == 'polygon':
            segmentations, labels = self.group_polygon(points, groups, labels)
            for segm, label in zip(segmentations, labels):
                rle = self.poly_to_RLE(segm, img_size)
                bbox = self.mask_to_bbox(rle)
                area = self.mask_to_area(rle)

                anno_dict = dict(
                    category_id=label,
                    segmentation=segm,
                    area=area,
                    bbox=bbox,
                    iscrowd=0)
                annotations.append(anno_dict)
        else:
            pass

        return annotations

    def group_polygon(self, points, groups, labels):
        grouping_dict = OrderedDict()
        for group in groups:
            if group != -1:
                grouping_dict[group] = dict(segmentation=[], categories=[])

        for poly, group, label in zip(points, groups, labels):
            if group != -1:
                grouping_dict[group]['segmentation'].append(poly)
                grouping_dict[group]['categories'].append(label)

        # final lists(ordered, then grouped) that should be returned
        segmentations = []
        labels_seg = []
        done_group = []
        for poly, group, label in zip(points, groups, labels):
            if group != -1:
                if group in done_group:
                    assert group not in grouping_dict.keys()
                    continue

                # first in, first out
                key, dict_item = grouping_dict.popitem(last=False)
                assert key == group

                segment = dict_item['segmentation']
                categories = dict_item['categories']
                category_set = set(categories)

                assert len(category_set) == 1
                category_id = category_set.pop()
                done_group.append(group)
            else:
                segment = [poly]
                category_id = label

            segmentations.append(segment)
            labels_seg.append(category_id)
        assert len(grouping_dict) == 0
        return segmentations, labels_seg

    def poly_to_RLE(self, segm, img_size):
        h, w = img_size
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
        return rle

    def mask_to_area(self, rle):
        return int(maskUtils.area(rle))

    def mask_to_bbox(self, rle):
        return maskUtils.toBbox(rle).tolist()


def symlink_images_to_target(image_list, target_path):
    for source_image_path in image_list:
        img_filename = osp.basename(source_image_path)
        target_img_path = osp.join(target_path, img_filename)
        mmcv.symlink(source_image_path, target_img_path)
    print('images and annotations are copied to: {}'.format(target_path))


def convert_annotations(img_location, ann_location, split_type, dataset_type):
    print('Generating annotations of images located in: {}'.format(
        img_location))  # noqa
    image_folder_list = glob.glob(osp.join(img_location, '*'))

    image_id = 0
    ann_id = 0

    images = []
    annotations = []
    categories = []
    prog_bar = mmcv.ProgressBar(len(image_folder_list))
    for image_folder in image_folder_list:
        xml_list = glob.glob(osp.join(image_folder, '*.xml'))
        assert len(xml_list) == 1
        xml_ann = xml_list[0]
        xp = XMLProcessor(xml_ann, dataset_type)
        for img_info in xp.img_infos:
            img_info['id'] = image_id
            img_size = (img_info['height'], img_info['width'])

            img_id_local = img_info['id_local']
            anns = xp.parse_annotations(
                img_id_local, dataset_type, img_size=img_size)
            for ann_dict in anns:
                ann_dict['id'] = ann_id
                ann_dict['image_id'] = image_id
                ann_id += 1
            image_id += 1
            images.append(img_info)
            annotations.extend(anns)
        prog_bar.update()

    for name in xp.CLASSES:
        label = xp.cat2label[name]
        cat_dict = dict(id=label, name=name)
        categories.append(cat_dict)

    json_anno = dict(
        images=images, annotations=annotations, categories=categories)
    ann_savepath = osp.join(ann_location,
                            dataset_type + '_' + split_type + '.json')
    mmcv.dump(json_anno, ann_savepath)
    print('\ntotal number of images from {} dataset: {}'.format(
        split_type, len(images)))
    print('Converted annotation is saved to: {}\n'.format(ann_savepath))


def main():
    args = parse_args()
    assert args.dataset_type == 'bbox' or args.dataset_type == 'polygon'

    # target directory construction for dataset
    target_location = 'data/sidewalk_dataset'
    train_location = osp.join(target_location,
                              'images_' + args.dataset_type + '/train')
    val_location = osp.join(target_location,
                            'images_' + args.dataset_type + '/val')
    test_location = osp.join(target_location,
                             'images_' + args.dataset_type + '/test')
    ann_location = osp.join(target_location, 'annotations')
    mmcv.mkdir_or_exist(train_location)
    mmcv.mkdir_or_exist(val_location)
    mmcv.mkdir_or_exist(test_location)
    mmcv.mkdir_or_exist(ann_location)

    image_location = args.imagefolder_location
    mmcv.mkdir_or_exist(image_location)

    image_folder_list = glob.glob(osp.join(image_location, '*'))
    image_folder_list.sort(reverse=False)
    assert len(image_folder_list) > 0

    random.shuffle(image_folder_list)

    num_valid = int(len(image_folder_list) * 0.05)
    num_test = int(len(image_folder_list) * 0.15)
    num_train = len(image_folder_list) - num_valid - num_test

    train_folders = image_folder_list[:num_train]
    val_folders = image_folder_list[num_train:num_train + num_valid]
    test_folders = image_folder_list[num_train + num_valid:]

    # symlink image and annotations to train, val, test split
    symlink_images_to_target(train_folders, train_location)
    symlink_images_to_target(val_folders, val_location)
    symlink_images_to_target(test_folders, test_location)

    # convert annotation file for each split
    print('')
    convert_annotations(train_location, ann_location, 'train',
                        args.dataset_type)
    convert_annotations(val_location, ann_location, 'val', args.dataset_type)
    convert_annotations(test_location, ann_location, 'test', args.dataset_type)


if __name__ == '__main__':
    main()
