# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING, II

from fairseq.data import SubsampleDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import Dictionary

from fairseq_user.dataset import ImageDataset
from .generator import PrefixSequenceGenerator

logger = logging.getLogger(__name__)


@dataclass
class ImagePretrainingStage1Config(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    multi_data: Optional[List[str]] = None
    input_size: int = 256
    augmentation: str = "randcrop"
    subsample: float = 1
    seed: int = II("common.seed")
    dataset_type: str = "imagefolder"
    source_vocab_size: int = -1
    target_vocab_size: int = 8192
    max_target_positions: int = 1024
    
    add_bos_token: bool = False
    tokens_per_sample: int = 1024

    num_shards: int = -1
    shard_id: int = -1

    linear_probe: bool = False

@register_task("image_generation_stage1", dataclass=ImagePretrainingStage1Config)
class ImagePretrainingStage1Task(FairseqTask):
    """ """

    cfg: ImagePretrainingStage1Config
    
    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: ImagePretrainingStage1Config, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """
        src_dict = None
        tgt_dict = Dictionary(extra_special_symbols=["<n>"])
        for i in range(cfg.target_vocab_size):
            tgt_dict.add_symbol(str(i))
        logger.info(f"tgt dictionary size: " f"{len(tgt_dict):,}")

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        cfg = self.cfg

        if cfg.num_shards > 0:
            assert not split.startswith("train")
        
        self.datasets[split] = ImageDataset(
            root=data_path if cfg.multi_data is None else cfg.multi_data,
            split=split,
            input_size=cfg.input_size,
            augmentation=cfg.augmentation,
            num_shards=cfg.num_shards,
            shard_id=cfg.shard_id,
            tgt_dict=self.tgt_dict,
            linear_probe=cfg.linear_probe,
        )

        if cfg.subsample < 1:
            self.datasets[split] = SubsampleDataset(
                self.datasets[split],
                cfg.subsample,
                shuffle=True,
                seed=cfg.seed,
            )
    
    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
        ):
        seq_gen_cls = PrefixSequenceGenerator
        generator = super().build_generator(
                models,
                args,
                seq_gen_cls,
                extra_gen_cls_kwargs,
            )
        return generator

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize
