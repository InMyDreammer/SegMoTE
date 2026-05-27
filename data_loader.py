
import math
import os
import json
import ast
import random
import itertools
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from monai import data, transforms
from scipy import sparse
from scipy.ndimage import binary_opening, binary_closing
from scipy.ndimage import label as label_structure
from scipy.ndimage import sum as sum_structure
from PIL import Image

from dataloaders.data_utils import (
    Resize,
    PermuteTransform,
    LongestSidePadding,
    Normalization,
    get_points_from_mask,
    get_bboxes_from_mask
)


class UniversalDataset(Dataset):


    def __init__(self, args, datalist, classes_list, transform, data_dir, mask_num, test_mode, max_retry=10):
        self.data_dir = data_dir
        self.datalist = datalist
        self.test_mode = test_mode
        self.image_size = args.image_size
        self.mask_num = mask_num
        self.transform = transform
        self.max_retry = max_retry


        classes = list(classes_list)
        if 'background' in classes:
            classes.remove('background')
        self.target_list = classes

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):


        for _ in range(self.max_retry):
            item_dict = self.datalist[idx]
            image_path = os.path.join(self.data_dir, item_dict['image'])
            label_path = os.path.join(self.data_dir, item_dict['label'])


            try:
                image_array = np.array(Image.open(image_path).convert('RGB'))
            except Exception as e:

                idx = np.random.randint(self.__len__())
                continue


            try:
                gt_shape = ast.literal_eval(label_path.split('.')[-2])
                allmatrix_sp = sparse.load_npz(label_path)
                label_array = allmatrix_sp.toarray().reshape(gt_shape)
            except Exception as e:
                idx = np.random.randint(self.__len__())
                continue

            if self.test_mode:
                item_ori = {'image': image_array, 'label': label_array}
                item = self.transform(item_ori)
                _, H, W = item['image'].shape


                label_ids = torch.sum(item['label'], dim=(1, 2))
                label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()
                if len(label_ids) == 0:
                    idx = np.random.randint(self.__len__())
                    continue

                nonzero_labels = torch.zeros(len(label_ids), 1, H, W)
                nonzero_category, nonzero_ori_labels = [], []
                point_coords, point_labels, bboxes = [], [], []

                for j, region_id in enumerate(label_ids):
                    nonzero_labels[j][0] = item['label'][region_id]

                    nonzero_ori_labels.append(torch.tensor(np.moveaxis(label_array[region_id], -1, 0)))

                    pts = get_points_from_mask(nonzero_labels[j], top_num=0.5)
                    point_coords.append(torch.as_tensor(pts[0]))
                    point_labels.append(torch.as_tensor(pts[1]))

                    bboxes.append(torch.as_tensor(get_bboxes_from_mask(nonzero_labels[j], offset=0)))
                    nonzero_category.append(self.target_list[region_id])

                item['gt'] = nonzero_labels
                item['ori_gt'] = torch.stack(nonzero_ori_labels, dim=0)
                item['gt_target'] = nonzero_category
                item['gt_point_coords'] = torch.stack(point_coords)
                item['gt_point_labels'] = torch.stack(point_labels)
                item['gt_bboxes'] = torch.stack(bboxes)
                item['image_root'] = [image_path]

            else:

                pseudo_key = 'imask' if 'imask' in self.datalist[0] else 'pseudo'
                pseudo_path = os.path.join(self.data_dir, item_dict[pseudo_key])

                try:
                    pseudo_array = np.load(pseudo_path).astype(np.float32)
                except Exception as e:
                    idx = np.random.randint(self.__len__())
                    continue

                item_ori = {'image': image_array, 'label': label_array, 'pseudo': pseudo_array}
                item = self.transform(item_ori)
                item['pseudo'] = self.cleanse_pseudo_label(item['pseudo'])

                pseudo_ids = torch.unique(item['pseudo'])
                pseudo_ids = pseudo_ids[pseudo_ids != -1]
                if len(pseudo_ids) == 0:
                    idx = np.random.randint(self.__len__())
                    continue

                _, H, W = item['image'].shape
                select_pseudo = torch.zeros(self.mask_num, 1, H, W)

                (select_pseudo,
                 point_coords_pseudo,
                 point_labels_pseudo,
                 bboxes_pseudo) = self.preprocess_pseudo(item['pseudo'], pseudo_ids, select_pseudo)

                label_ids = torch.sum(item['label'], dim=(1, 2))
                label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()
                if len(label_ids) == 0:
                    idx = np.random.randint(self.__len__())
                    continue

                select_labels = torch.zeros(self.mask_num, 1, H, W)
                (select_labels,
                 point_coords,
                 point_labels,
                 bboxes,
                 nonzero_category) = self.preprocess_label(item['label'], label_ids, select_labels)


                item['gt'] = select_labels
                item['pseudo'] = select_pseudo
                item['gt_point_coords'] = point_coords
                item['gt_point_labels'] = point_labels
                item['gt_bboxes'] = bboxes
                item['gt_target'] = nonzero_category
                item['pseudo_point_coords'] = point_coords_pseudo
                item['pseudo_point_labels'] = point_labels_pseudo
                item['pseudo_bboxes'] = bboxes_pseudo
                item['npromt'] = False
                if 'isic' in image_path:
                    item['npromt'] = True


            post_item = self.std_keys(item)
            post_item['dataset_code'] = getattr(self, 'dataset_code', 'unknown')
            return post_item


        raise RuntimeError("Max retries exceeded when sampling a valid item.")


    def preprocess_pseudo(self, pseudo_label, pseudo_ids, select_pseudo):
        point_coords, point_labels, bboxes = [], [], []
        choose = random.sample(list(pseudo_ids), k=self.mask_num) if len(pseudo_ids) >= self.mask_num\
                 else random.choices(list(pseudo_ids), k=self.mask_num)
        for idx, region_id in enumerate(choose):
            select_pseudo[idx][pseudo_label == region_id.item()] = 1
            pts = get_points_from_mask(select_pseudo[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(pts[0]))
            point_labels.append(torch.as_tensor(pts[1]))
            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_pseudo[idx], offset=5)))
        return select_pseudo, torch.stack(point_coords), torch.stack(point_labels), torch.stack(bboxes)

    def preprocess_label(self, gt_label, label_ids, select_labels):
        point_coords, point_labels, bboxes, categories = [], [], [], []
        choose = random.sample(list(label_ids), k=self.mask_num) if len(label_ids) >= self.mask_num\
                 else random.choices(list(label_ids), k=self.mask_num)
        for idx, region_id in enumerate(choose):
            select_labels[idx][0] = gt_label[region_id]
            pts = get_points_from_mask(select_labels[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(pts[0]))
            point_labels.append(torch.as_tensor(pts[1]))
            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_labels[idx], offset=5)))


            categories.append(self.target_list[region_id])

        return (select_labels,
                torch.stack(point_coords),
                torch.stack(point_labels),
                torch.stack(bboxes),
                categories)

    def std_keys(self, post_item):
        keys_to_remain = ['image', 'gt', 'ori_gt', 'image_root',
                          'gt_point_coords', 'gt_point_labels', 'gt_bboxes', 'gt_target',
                          'pseudo', 'pseudo_point_coords', 'pseudo_point_labels', 'pseudo_bboxes','npromt']

        for k in list(post_item.keys()):
            if k not in keys_to_remain:
                del post_item[k]
        return post_item

    def cleanse_pseudo_label(self, pseudo_seg):
        total_voxels = pseudo_seg.numel()
        threshold = total_voxels * 0.0005
        unique_values = torch.unique(pseudo_seg)

        for value in unique_values:
            voxel_count = (pseudo_seg == value).sum()
            if voxel_count < threshold:
                pseudo_seg[pseudo_seg == value] = -1

        for label in torch.unique(pseudo_seg):
            if label == -1:
                continue
            binary_mask = pseudo_seg == label
            open_mask = binary_opening(binary_mask.squeeze())
            close_mask = binary_closing(open_mask)
            processed = torch.tensor(close_mask)

            labeled_mask, num_labels = label_structure(processed)
            label_sizes = sum_structure(processed, labeled_mask, range(num_labels + 1))
            small_labels = np.where(label_sizes < threshold)[0]
            for label_del in small_labels:
                processed[labeled_mask == label_del] = False

            pseudo_seg[binary_mask] = -1
            pseudo_seg[processed.unsqueeze(0)] = label
        return pseudo_seg


class UnionDataset(Dataset):

    def __init__(self, concat_dataset, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = torch.cumsum(torch.tensor([0] + self.lengths), dim=0)
        self.concat_dataset = concat_dataset

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

from bisect import bisect_right
from torch.utils.data import Sampler
from bisect import bisect_right
from torch.utils.data import Sampler

class GroupedBatchedDistributedSampler(DistributedSampler):


    def __init__(self, dataset, batch_size, group_a_codes,
                 shuffle=True, drop_last=True, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)
        self.batch_size = batch_size

        self.group_a = set([c.lower() for c in group_a_codes])


        self._dataset_codes = [ds.dataset_code for ds in dataset.datasets]
        self._offsets = dataset.offsets.tolist()
        self._lengths = dataset.lengths

    def _index_to_dataset_id(self, idx: int) -> int:

        return bisect_right(self._offsets, idx) - 1

    def _split_indices_by_group(self):
        idxs_a, idxs_b = [], []
        for idx in range(len(self.dataset)):
            did = self._index_to_dataset_id(idx)
            code = self._dataset_codes[did]
            if code in self.group_a:
                idxs_a.append(idx)
            else:
                idxs_b.append(idx)
        return idxs_a, idxs_b

    @staticmethod
    def _shuffle_inplace(xs):
        random.shuffle(xs)

    @staticmethod
    def _truncate_to_mul(xs, k):
        n = (len(xs) // k) * k
        return xs[:n]

    def __iter__(self):

        idxs_a, idxs_b = self._split_indices_by_group()


        if self.shuffle:
            self._shuffle_inplace(idxs_a)
            self._shuffle_inplace(idxs_b)


        idxs_a = self._truncate_to_mul(idxs_a, self.batch_size)
        idxs_b = self._truncate_to_mul(idxs_b, self.batch_size)

        batches_a = [idxs_a[i:i+self.batch_size] for i in range(0, len(idxs_a), self.batch_size)]
        batches_b = [idxs_b[i:i+self.batch_size] for i in range(0, len(idxs_b), self.batch_size)]


        all_batches = batches_a + batches_b
        if self.shuffle:
            self._shuffle_inplace(all_batches)


        total_batches = (len(all_batches) // self.num_replicas) * self.num_replicas
        all_batches = all_batches[:total_batches]


        per = len(all_batches) // self.num_replicas
        start = self.rank * per
        end = start + per
        my_batches = all_batches[start:end]


        self.total_size = sum(len(b) for b in all_batches)
        self.num_samples = sum(len(b) for b in my_batches)


        flat = list(itertools.chain.from_iterable(my_batches))
        return iter(flat)


def test_collate_fn(batch):
    assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    gt_prompt['point_coords'] = batch[0]['gt_point_coords']
    gt_prompt['point_labels'] = batch[0]['gt_point_labels']
    gt_prompt['bboxes'] = batch[0]['gt_bboxes']
    image_root = batch[0].get('image_root', None)
    target_list = batch[0]['gt_target']

    dataset_code = batch[0].get('dataset_code', 'unknown')
    is_group_a = dataset_code in {'isic2016_task1','isic2017_task1','isic2018_task1','ISLES_SISS'}

    return {
        'image': batch[0]['image'].unsqueeze(0),
        'label': batch[0]['gt'],
        'ori_label': batch[0].get('ori_gt', None),
        'gt_prompt': gt_prompt,
        'target_list': target_list,
        'image_root': image_root,
        'is_group_a': is_group_a,
    }


def train_collate_fn(batch):
    images, labels, pseudos, target_list, npromt = [], [], [], [], []
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    pseudo_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}

    dataset_codes = []

    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['gt'])
        gt_prompt['point_coords'].append(sample['gt_point_coords'])
        gt_prompt['point_labels'].append(sample['gt_point_labels'])
        gt_prompt['bboxes'].append(sample['gt_bboxes'])
        target_list += sample['gt_target']
        npromt.append(sample['npromt'])

        if 'dataset_code' in sample:
            dataset_codes.append(sample['dataset_code'])

        if 'pseudo' in sample:
            pseudos.append(sample['pseudo'])
            pseudo_prompt['point_coords'].append(sample['pseudo_point_coords'])
            pseudo_prompt['point_labels'].append(sample['pseudo_point_labels'])
            pseudo_prompt['bboxes'].append(sample['pseudo_bboxes'])

    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    pseudos = torch.cat(pseudos, dim=0) if len(pseudos) > 0 else None

    gt_prompt = {k: torch.cat(v, dim=0) if len(v) != 0 else None for k, v in gt_prompt.items()}
    pseudo_prompt = {k: torch.cat(v, dim=0) if len(v) != 0 else None for k, v in pseudo_prompt.items()}


    is_group_a = all([c in {'isic2016_task1','isic2017_task1','isic2018_task1',} for c in dataset_codes])

    batch_out = {
        'image': images,
        'label': labels,
        'target_list': target_list,
        'gt_prompt': gt_prompt,
        'pseudo_prompt': pseudo_prompt,
        'npromt': npromt,
        'is_group_a': is_group_a,
        'dataset_codes': dataset_codes
    }
    if pseudos is not None:
        batch_out['pseudo'] = pseudos
    return batch_out


def build_concat_dataset(args, root_path, dataset_list, test_mode, mask_num):


    concat = []
    total_len = 0


    if dataset_list == 'all' or dataset_list == None:
        file_path = os.path.join(root_path, 'class_mapping.json')
        dataset_json = json.load(open(file_path, 'r', encoding='utf-8'))
        if test_mode:
            dataset_list = dataset_json['screening_dataset_list']
        else:
            dataset_list = dataset_json['dataset_list']

    for code in dataset_list:
        data_dir = os.path.join(root_path, code)
        dataset_json = os.path.join(data_dir, 'dataset.json')
        if not os.path.isfile(dataset_json):
            print(f'[WARN] skip dataset {code}: {dataset_json} not found.')
            continue

        ds = json.load(open(dataset_json, 'r'))
        classes_list = list(ds['labels'].values())

        target_size = (args.image_size, args.image_size)
        if test_mode:
            datalist = ds['test']
            transform = transforms.Compose([
                Resize(keys=["image", "label"], target_size=target_size),
                PermuteTransform(keys=["image"], dims=(2, 0, 1)),
                transforms.ToTensord(keys=["image", "label"]),
                Normalization(keys=["image"]),
            ])
        else:
            datalist = ds['training']
            transform = transforms.Compose([
                Resize(keys=["image", "label", "pseudo"], target_size=target_size),
                PermuteTransform(keys=["image"], dims=(2, 0, 1)),
                transforms.ToTensord(keys=["image", "label", "pseudo"]),
                Normalization(keys=["image"]),
                transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
                transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
            ])


        uni = UniversalDataset(
            args=args,
            datalist=datalist,
            classes_list=classes_list,
            transform=transform,
            data_dir=data_dir,
            mask_num=mask_num,
            test_mode=test_mode
        )
        uni.dataset_code = code.lower()
        concat.append(uni)
        total_len += len(uni)

    print(f'[INFO] Loaded {len(concat)} datasets, total size: {total_len}')
    if len(concat) == 0:
        raise RuntimeError('No valid datasets found to build.')

    return UnionDataset(ConcatDataset(concat), concat)


def get_multi_loader(args):


    test_mode = args.test_mode
    mask_num = args.mask_num
    root_path = args.data_dir


    union_ds = build_concat_dataset(
        args=args,
        root_path=root_path,
        dataset_list=args.dataset_list,
        test_mode=test_mode,
        mask_num=mask_num
    )


    group_a = set([c.lower() for c in getattr(args, 'group_a', ['isic'])])


    if getattr(args, 'dist', False):
        sampler = GroupedBatchedDistributedSampler(
            dataset=union_ds,
            batch_size=args.batch_size,
            group_a_codes=group_a,
            shuffle=not args.test_mode,
            drop_last=True,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
        )
        shuffle = False
    else:

        sampler = GroupedBatchedDistributedSampler(
            dataset=union_ds,
            batch_size=args.batch_size,
            group_a_codes=group_a,
            shuffle=not args.test_mode,
            drop_last=True,
            num_replicas=1,
            rank=0,
        )
        shuffle = False


    collate_fn = test_collate_fn if args.test_mode else train_collate_fn
    persistent = bool(args.num_workers and args.num_workers > 0)
    loader = data.DataLoader(
        union_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate_fn,
    )
    return loader


if __name__ == "__main__":
    import argparse


    def set_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, default="/data/medical")
        parser.add_argument("--dataset_list", type=str, nargs='*', default=['BTCV', 'LiTS'])
        parser.add_argument('--image_size', type=int, default=256)
        parser.add_argument('--test_mode', action='store_true')
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--dist', dest='dist', type=bool, default=False, help='use distributed training or not')
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--mask_num', type=int, default=5)
        args = parser.parse_args()
        return args

    args = set_parse()
    loader = get_multi_loader(args)
    for i, batch in enumerate(loader):
        imgs = batch['image']
        gts = batch['label']
        print(f'iter {i}: image {imgs.shape}, label {gts.shape}')
        if 'pseudo' in batch:
            print('pseudo:', batch['pseudo'].shape)
        print('targets:', batch['target_list'][:5], '...')
        if i >= 2:
            break
