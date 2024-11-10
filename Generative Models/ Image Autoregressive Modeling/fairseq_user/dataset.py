# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import logging
import math
import random
import time
import glob
import numpy as np
import os
from typing import List, Optional, Tuple
from pathlib import Path
import pandas as pd

import torch
from torchvision import datasets, transforms
from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)


class ImageDataset(FairseqDataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        shuffle=True,
        augmentation="noaug",
        num_shards: int = -1,
        shard_id: int = -1,
        src_dict = None,
        tgt_dict = None,
        linear_probe = False,
    ):
        FairseqDataset.__init__(self)
        
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.linear_probe = linear_probe

        self.shuffle = shuffle
        
        self.build_transfom(augmentation, input_size, split)
    
        self.dataset = datasets.ImageFolder(os.path.join(root, split))
        self.src_sizes = np.array([1] * len(self.dataset))

        logger.info(
            f"{split} transform: {self.transform}"
        )
        logger.info(
            f"{self.num_shards}, {self.shard_id}"
        )
        logger.info(f"loaded {len(self.dataset)} examples")
        
    def build_transfom(self, augmentation, input_size, split):
        if augmentation == "noaug":
            self.transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=3),
                transforms.CenterCrop(input_size),
                transforms.ToTensor()])
        elif augmentation == "centercrop":
            self.transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=3),
                transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        elif augmentation == "randcrop":
            self.transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=3),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        elif augmentation == "randresizedcrop":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            raise NotImplementedError

        if "val" in split:
            self.transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=3),
                transforms.CenterCrop(224 if self.linear_probe else input_size),  # 224 for linear probe
                transforms.ToTensor()])

    def __getitem__(self, index):
        if self.num_shards > 0:
            index = index * self.num_shards + self.shard_id
        
        img, cls_id = self.dataset[index]
        img = self.transform(img)
        
        return {"id": index, "imgs": img, "img_size": [img.size()[1], img.size()[2]], "cls_id": cls_id}

    def __len__(self):
        if self.num_shards > 0:
            self.src_sizes = self.src_sizes[self.shard_id::self.num_shards]
            return len(self.dataset) // self.num_shards + int(self.shard_id < (len(self.dataset) % self.num_shards))
        else:
            return len(self.dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        
        collated_img = torch.stack([s["imgs"] for s in samples], dim = 0)        
        img_sizes = torch.LongTensor([s["img_size"] for s in samples])
        cls_ids = torch.LongTensor([s["cls_id"] for s in samples])

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "src_tokens": torch.LongTensor([[0]] * len(samples)),
                "imgs": collated_img,
                "img_sizes": img_sizes,
                "src_lengths": img_sizes[:, 0] * img_sizes[:, 1] // 16 // 16,
                "cls_ids": cls_ids # [B]
            },
            "target": None,
            "ntokens": (img_sizes[:,0] * img_sizes[:,1]).sum(),
            "nsentences": len(samples),
        }
            
        return res

    def num_tokens(self, index):
        return self.src_sizes[index]

    def size(self, index):
        return self.src_sizes[index]

    @property
    def sizes(self):
        return self.src_sizes

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = np.random.permutation(len(self))
        else:
            order = np.arange(len(self))

        return order

class GenDataset(FairseqDataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        shuffle=False,
        augmentation="noaug",
        num_shards: int = -1,
        shard_id: int = -1,
        dataset_type: str = "imagefolder",
        src_dict = None,
        tgt_dict = None,
    ):
        FairseqDataset.__init__(self)

        self.dataset_type = dataset_type
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.shuffle = shuffle
        
        with open((Path(root) / split).with_suffix(".codecs")) as fo:
            self.dataset = [self.src_dict.encode_line(line, add_if_not_exist=False, append_eos=False) for line in fo.readlines()]
        
    def __getitem__(self, index):

        if self.num_shards > 0:
            index = index * self.num_shards + self.shard_id

        img = self.dataset[index].long()
        
        if len(img) != 16 * 16:
            logger.info(f"id: {index}, length: {len(img)}")
            img = img[:16 * 16]

        v = {"id": index, "imgs": img, "img_size": [16, 16]} # "pixel": pixel}
        return v

    def __len__(self):
        if self.num_shards > 0:
            return len(self.dataset) // self.num_shards + int(self.shard_id < (len(self.dataset) % self.num_shards))
        else:
            return len(self.dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s["imgs"] for s in samples], dim=0) 
        img_sizes = torch.LongTensor([s["img_size"] for s in samples])

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "src_tokens": collated_img,
                "imgs": None,
                "img_sizes": img_sizes,
                "src_lengths": img_sizes[:, 0] * img_sizes[:, 1],
            },
            "target": None,
            "ntokens": (img_sizes[:,0] * img_sizes[:,1]).sum(),
            "nsentences": len(samples),
        }
        
        return res

    def num_tokens(self, index):
        return 256

    def size(self, index):
        return 256

    @property
    def sizes(self):
        return np.full((len(self),), 256)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = np.random.permutation(len(self))
        else:
            order = np.arange(len(self))

        return order


