import copy
import os
import os.path as osp

from mmdet.utils import build_from_cfg
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .registry import DATASETS


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefixes', None)
    proposal_files = cfg.get('proposal_file', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_sidewalk_dataset(cfg, mode='train', custom_postfix=''):
    assert mode in ['train', 'val', 'test']
    assert cfg['type'] == 'SideWalkBBoxDataset' \
        or cfg['type'] == 'SideWalkPolygonDataset'
    # assign ann_files and img_prefixes
    data_cfg = cfg.copy()
    if cfg['type'] == 'SideWalkBBoxDataset':
        data_root = osp.join(data_cfg.pop('data_root'), mode + custom_postfix)
        dir_list = [
            path for path in os.listdir(data_root)
            if osp.isdir(osp.join(data_root, path))
        ]
        dir_list.sort()
        ann_file_list = [
            osp.join(data_root, folder, folder + '.xml') for folder in dir_list
        ]
        img_prefix_list = [
            osp.join(data_root, folder) + '/' for folder in dir_list
        ]

        data_cfg['ann_file'] = ann_file_list
        data_cfg['img_prefix'] = img_prefix_list
    else:
        # SideWalkPolygonDataset
        data_cfg['img_prefix'] = data_cfg['img_prefix'] + custom_postfix

    return build_dataset(data_cfg)


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg['ann_file'], (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
