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

from fairseq_user.dataset import ImageDataset, GenDataset
from .generator import PrefixSequenceGenerator, LMIterativeRefinementGenerator

logger = logging.getLogger(__name__)


@dataclass
class ImagePretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    multi_data: Optional[List[str]] = None
    input_size: int = 256
    local_cache_path: Optional[str] = None
    augmentation: str = "randcrop"
    subsample: float = 1
    seed: int = II("common.seed")
    dataset_type: str = "imagefolder"
    source_vocab_size: int = 1024
    target_vocab_size: int = 1024
    max_target_positions: int = 1024
    
    add_bos_token: bool = False
    tokens_per_sample: int = 1024
    
    num_shards: int = -1
    shard_id: int = -1

@register_task("image_generation_stage2", dataclass=ImagePretrainingConfig)
class ImagePretrainingTask(FairseqTask):
    """ """

    cfg: ImagePretrainingConfig
    
    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: ImagePretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """
        src_dict = Dictionary(extra_special_symbols=["<n>"])
        for i in range(cfg.source_vocab_size):
            src_dict.add_symbol(str(i))
        logger.info(f"src dictionary size: " f"{len(src_dict):,}")
        
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
        
        if split.startswith("gen"):
            self.datasets[split] = GenDataset(
                root=data_path if cfg.multi_data is None else cfg.multi_data,
                split=split,
                input_size=cfg.input_size,
                augmentation=cfg.augmentation,
                dataset_type=cfg.dataset_type,
                num_shards=cfg.num_shards,
                shard_id=cfg.shard_id,
                src_dict=self.src_dict,
                tgt_dict=self.tgt_dict,
            )
        else:
            self.datasets[split] = ImageDataset(
                root=data_path if cfg.multi_data is None else cfg.multi_data,
                split=split,
                input_size=cfg.input_size,
                augmentation=cfg.augmentation,
                num_shards=cfg.num_shards,
                shard_id=cfg.shard_id,
                src_dict=self.src_dict,
                tgt_dict=self.tgt_dict,
            )

        if cfg.subsample < 1:
            self.datasets[split] = SubsampleDataset(
                self.datasets[split],
                cfg.subsample,
                shuffle=True,
                seed=cfg.seed,
            )

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.cfg.max_target_positions
    
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        import os
        if "_nar_" in os.environ['IMG_SAVE_DIR']:
            
            generator = LMIterativeRefinementGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
                max_iter=getattr(args, "iter_decode_max_iter", 10),
                beam_size=getattr(args, "iter_decode_with_beam", 1),
                reranking=getattr(args, "iter_decode_with_external_reranker", False),
                decoding_format=getattr(args, "decoding_format", None),
                adaptive=not getattr(args, "iter_decode_force_max_iter", False),
                retain_history=getattr(args, "retain_iter_history", False),
            )
            generator.tgt_dict = self.tgt_dict
            generator.src_dict = self.src_dict
            generator.dataset_type = self.cfg.dataset_type
            return generator
        else:
            seq_gen_cls = PrefixSequenceGenerator
            generator = super().build_generator(
                    models,
                    args,
                    seq_gen_cls,
                    extra_gen_cls_kwargs,
                )
            generator.src_dict = self.src_dict
            generator.dataset_type = self.cfg.dataset_type
            return generator


   