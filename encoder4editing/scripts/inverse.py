import argparse

import torch
import numpy as np
import sys
import os
import dlib

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--image_path", type=str, default=None,
                        help="The path to the images to be inverted")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)