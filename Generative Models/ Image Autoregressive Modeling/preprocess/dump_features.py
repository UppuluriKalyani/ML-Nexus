# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from PIL import Image
import numpy as np
from npy_append_array import NpyAppendArray
import random
import os
import math
from tqdm import tqdm
import torch
import torch.distributed as distr

from data_handler import ManifestDataset
from distributed import init_distributed_context

logger = logging.getLogger(__name__)

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", required=True, help="Path to the dataset manifest file"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    parser.add_argument(
        "--rank",
        type=int,
    )
    parser.add_argument(
        "--world_size",
        type=int,
    )
    parser.add_argument(
        "--single",
        action="store_true"
    )
    parser.add_argument("--distributed_port", type=int, default=58554)
    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

    return args

def worker_shard_path(fname, suffix, worker_id) -> Path:
    return Path(fname) / f"{suffix}_partial_{worker_id}.npy"

def transcribe(args, rank, world_size):
    from dinov2_encoder import Encoder
    
    dataset = ManifestDataset(args.manifest, "train")
    
    os.makedirs(args.output, exist_ok=True)
    output_files = NpyAppendArray(worker_shard_path(args.output, "features", rank), delete_if_exists=True)
    encoder = Encoder("models/dinov2_vitb14_reg4_pretrain.pth", layer=args.layer).cuda()
    
    for i in tqdm(range(rank, len(dataset), world_size)):
        if random.random() > 0.1: continue
        image = dataset[i].cuda().unsqueeze(0)
        feat = encoder(image)
            
        feat = feat.flatten(0,1).cpu().detach().numpy()
        output_files.append(feat)
        if output_files.fortran_order:
            output_files.fortran_order = False
    output_files.close()
    
def main(args):
    context = init_distributed_context(args.distributed_port)
    logger.info(f"Distributed context {context}")

    n_gpus = torch.cuda.device_count()
    with torch.cuda.device(context.local_rank % n_gpus):
        transcribe(args, context.rank, context.world_size)

    if context.world_size > 1:
        distr.barrier()

if __name__ == "__main__":
    args = get_args()
    main(args)
