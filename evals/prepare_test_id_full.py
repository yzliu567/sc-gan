import sys
sys.path.append('./simi')
sys.path.append('..')
from id_feature import IDFeatureNet
from model import Generator_MMD, Generator, Generator_Cond
import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from torchvision import utils
from train_mmd import getsoftlabel, ensure_dir

def test(args, g_ema, device, mean_latent):
    #init
    num = 5
    all_simi = torch.empty([args.num_test, num - 1]).cuda()
    id_feat = IDFeatureNet().to(device).eval()

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.num_test)):
            sample_z = torch.randn(1, args.latent, device=device).repeat(num, 1)
            df_modes = torch.zeros([num, args.modes], dtype=torch.float32).to(device)
            df_modes[:,0] = 1.0

            vals = np.random.uniform(0,1,[num])
            vals[0] = 0.5
            df_modes[:,7:10] = getsoftlabel(vals, 3, 0, 1)
            df_modes[:,8] = 0

            vals = np.random.uniform(0,1,[1]).repeat(num)
            df_modes[:,4:7] = getsoftlabel(vals, 3, 0, 1)
            df_modes[:,5] = 0

            vals = np.random.uniform(0, 1, [1]).repeat(num)
            df_modes[:, 1:4] = getsoftlabel(vals, 3, 0, 1)
            df_modes[:, 2] = 0

            if args.mmd:
                sample, _ = g_ema(
                    [sample_z], truncation=args.truncation, truncation_latent=mean_latent, default_mode=df_modes
                )
            else:
                sample, _ = g_ema(
                    [sample_z], truncation=args.truncation, truncation_latent=mean_latent
                )
            res = id_feat(sample)
            #compute similarity vectors
            simi = torch.sum(torch.mul(res[0], res[1:]), dim=1)
            all_simi[i] = simi
            for j in range(num):
                utils.save_image(
                    sample[j],
                    args.img_dir + f"/{str(i).zfill(6)}_{str(j)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            if args.debug:
                print(simi)
                exit(0)
    np.save(args.out_dir + '/simi.npy', all_simi.detach().cpu().numpy())

if __name__ == '__main__':
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="batch size"
    )
    parser.add_argument(
        "--num_test", type=int, default=1000, help="number of images to be generated"
    )
    parser.add_argument(
        "--exp_name", type=str, help="experiment folder name"
    )
    parser.add_argument(
        "--iters", type=str, help="ckpt iters"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        '--mmd',
        action="store_true"
    )
    parser.add_argument(
        '--modes', type=int, default=10, help="lambda for generator mode loss"
    )
    parser.add_argument(
        '--debug', action="store_true"
    )

    args = parser.parse_args()

    args.exp_dir = '../exps/' + args.exp_name
    args.ckpt = os.path.join(args.exp_dir, 'checkpoint', args.iters + '.pt')
    args.out_dir = args.exp_dir + '/images_for_id'
    ensure_dir(args.out_dir)
    args.img_dir = args.exp_dir + '/images_for_id/images'
    ensure_dir(args.img_dir)

    args.latent = 512
    args.n_mlp = 8
    if args.mmd:
        g_ema = Generator_MMD(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, num_class=args.modes
        ).to(device)
        print('using mmd')
    else:
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        print('not using mmd')
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    test(args, g_ema, device, mean_latent)
