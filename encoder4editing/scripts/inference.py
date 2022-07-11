import argparse

import torch
import numpy as np
import sys
import os
import dlib
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image
import torch.nn.functional as F

def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    if os.path.exists(latents_file_path):
        latent_codes = torch.load(latents_file_path).to(device)
    else:
        latent_codes = get_all_latents(args, net, data_loader, args.n_sample, is_cars=is_cars)
        torch.save(latent_codes, latents_file_path)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes, is_cars=is_cars)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transform_image,
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader

def init(args):
    from models.stylegan2.model import Discriminator
    ckpt = torch.load(args.ganpath)
    global gandisc
    gandisc = Discriminator(1024, channel_multiplier=2, num_class=args.modes).cuda()
    gandisc.load_state_dict(ckpt['d'], strict=True)
    gandisc.eval()

    global all_modes_disc
    all_modes_disc = []
    from torchvision import transforms
    from PIL import Image
    global transform_image
    transform_image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
def get_latents(args, net, x, modes, is_cars=False):
    x_256 = F.interpolate(x, size=(256, 256), mode='bilinear')
    codes = net.encoder(x_256)

    with torch.no_grad():
        _, df_modes = gandisc(x, need_class=True)
        df_modes = df_modes.detach()
        soft_max = torch.nn.Softmax()
        df_modes[:,0] = 1
        for attri_id in range(args.modes // 3):
            df_modes[:,attri_id * 3 + 1 : attri_id * 3 + 4] = soft_max(df_modes[:,attri_id * 3 + 1: attri_id * 3 + 4])
            df_modes[:,attri_id * 3 + 2] = 0

        all_modes_disc.append(df_modes)

    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(args, net, data_loader, n_images=None, is_cars=False):
    modes = args.modes
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if n_images is not None and i > n_images:
                break
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(args, net, inputs, modes, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    modes = args.modes

    all_modes = torch.cat(all_modes_disc)

    for i in tqdm(range(args.n_sample)):
        df_mode = all_modes[i:i+1]
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True, default_mode=df_mode)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)
    torch.save(all_modes, args.save_dir+'/labels.pt')


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    parser.add_argument('--modes', type=int, default=10, help="number of modes")
    parser.add_argument('--ganpath',type=str, help="path to finetuned stylegan2(We need discriminator for modes)")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    init(args)
    main(args)
