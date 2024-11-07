import logging
import glob
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
import torch_fidelity
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


def main(args):

    print("Evaluating FID, IS...")
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=args.results_path,
        input2=args.target_path,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']
    logger.info(f"FID: {fid}, Inception Score: {inception_score}")
    #shutil.rmtree(args.results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--target-path", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")
    
    main(args)
