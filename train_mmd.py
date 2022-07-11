import argparse
import math
import random
import os
import cv2

from model import Discriminator, Generator_MMD
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


from dataset_ffhq import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    #with conv2d_gradfix.no_weight_gradients():
    if True:
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device, return_if_mixed=False):
    if prob > 0 and random.random() < prob:
        if return_if_mixed:
            return make_noise(batch, latent_dim, 2, device), True
        else:
            return make_noise(batch, latent_dim, 2, device)

    else:
        if return_if_mixed:
            return [make_noise(batch, latent_dim, 1, device)], False
        else:
            return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

import json

def save_log(args):
    json_str = json.dumps(args.__dict__, indent=4)
    with open(os.path.join(args.output_dir, args.exp_name, 'log_file.json'), 'w') as json_file:
        json_file.write(json_str)

# todo: simplify
def val2label(val, modes, min_scale, max_scale, device="cuda"):
    label = torch.zeros([modes],dtype=torch.float32).to(device)
    bounds = torch.linspace(min_scale, max_scale, modes).numpy()

    if val < bounds[0]:
        label[0] = 1
    elif val >= bounds[modes - 1]:
        label[modes - 1] = 1
    else:
        for j in range(modes - 1):
            if val < bounds[j + 1]:
                label[j + 1] = (val - bounds[j]) / (bounds[j+1] - bounds[j])
                label[j] = 1 - label[j+1]
                break
    return label

def getsoftlabel(val, modes, min_scale, max_scale, device="cuda"):
    batch = val.shape[0]
    label = torch.empty([batch, modes], dtype=torch.float32).to(device)
    for j in range(batch):
        label[j] = val2label(val[j], modes, min_scale, max_scale, device)
    return label

def sample_mode_soft(batch, device, modes):
    df_mode = torch.zeros((batch, modes)).to(device).to(torch.float32)
    df_mode[:,0] = 1.0
    df_mode[:,1:4] = getsoftlabel(torch.rand(batch), 3, 0, 1)
    df_mode[:,4:7] = getsoftlabel(torch.rand(batch), 3, 0, 1)
    df_mode[:,7:10] = getsoftlabel(torch.rand(batch), 3, 0, 1)
    idxs = df_mode.detach().clone()
    df_mode[:,2] = 0
    df_mode[:,5] = 0
    df_mode[:,8] = 0
    return idxs, df_mode

def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, extra=None, e_optim=None):

    loader = sample_data(loader)
    base_dir = './distances'
    minmax_path = os.path.join(base_dir, args.exp_setting + '_min_max.txt')
    min_v1, max_v1, min_v2, max_v2, min_v3, max_v3 = np.loadtxt(minmax_path)

    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    ensure_dir(exp_dir)

    for subdir_name in ["sample", "checkpoint", "bound", "editresults"]:
        ensure_dir(os.path.join(exp_dir, subdir_name))

    save_log(args)
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    log_soft_max = torch.nn.LogSoftmax()
    loss_cls = torch.nn.KLDivLoss()
    sigmoid_ce = torch.nn.BCEWithLogitsLoss()
    args.attrs = args.modes // 3

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        else:
            real_img, real_indexes, real_v1, real_v2, real_v3 = next(loader)
            real_img = real_img.to(device)
            real_label = torch.zeros([args.batch, args.modes]).to(device).to(torch.float32)
            real_label[:,0] = 1.0
            real_label[:,1:4] = getsoftlabel(real_v1, 3, min_v1, max_v1)
            real_label[:,4:7] = getsoftlabel(real_v2, 3, min_v2, max_v2)
            real_label[:,7:10] = getsoftlabel(real_v3, 3, min_v3, max_v3)
            '''
            print(real_label)
            print(real_indexes)
            utils.save_image(
                real_img,
                "tmp.png",
                normalize=True,
                range=(-1, 1),
            )
            exit(0)
            '''

        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        idxs, df_mode = sample_mode_soft(args.batch, device, args.modes)
        fake_img, _ = generator(noise, default_mode=df_mode)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred, fake_class = discriminator(fake_img, need_class=True, class_weights=df_mode)
        real_pred, real_class = discriminator(real_img_aug, need_class=True, class_weights=real_label)

        d_sup_loss_part1 = loss_cls(log_soft_max(real_class[:,1:4]), real_label[:,1:4])
        d_sup_loss_part2 = loss_cls(log_soft_max(real_class[:,4:7]), real_label[:,4:7])
        d_sup_loss_part3 = loss_cls(log_soft_max(real_class[:, 7:10]), real_label[:, 7:10])
        d_sup_loss = args.md_d * (d_sup_loss_part1 + d_sup_loss_part2 + d_sup_loss_part3)
        d_loss = d_logistic_loss(real_pred, fake_pred) + d_sup_loss
        loss_dict["d_sup"] = d_sup_loss

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug, class_weights=real_label)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise, flag = mixing_noise(args.batch, args.latent, args.mixing, device, return_if_mixed=True)
        idxs, df_mode = sample_mode_soft(args.batch, device, args.modes)
        fake_img, latent = generator(noise, return_latents=True, default_mode=df_mode)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, fake_class = discriminator(fake_img, need_class=True, class_weights=df_mode)
        g_loss = g_nonsaturating_loss(fake_pred)

        g_sup_loss_part1 = loss_cls(log_soft_max(fake_class[:,1:4]), idxs[:,1:4])
        g_sup_loss_part2 = loss_cls(log_soft_max(fake_class[:,4:7]), idxs[:,4:7])
        g_sup_loss_part3 = loss_cls(log_soft_max(fake_class[:, 7:10]), idxs[:, 7:10])
        g_sup_loss = args.md_g * (g_sup_loss_part1 + g_sup_loss_part2 + g_sup_loss_part3)
        loss_dict["g_sup"] = g_sup_loss
        g_loss += g_sup_loss

        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            idxs, df_mode = sample_mode_soft(path_batch_size, device, args.modes)
            fake_img, latents = generator(noise, return_latents=True, default_mode=df_mode)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        d_loss_sup = loss_reduced["d_sup"].mean().item()
        g_loss_sup = loss_reduced["g_sup"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}; "
                    f"d_sup: {d_loss_sup: .4f}; g_sup:{g_loss_sup: .4f};"
                )
            )
            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    sample_z = torch.randn(args.n_sample, args.latent, device=device)
                    g_ema.eval()
                    default_modes = torch.zeros([args.n_sample, args.modes], dtype=torch.float32).cuda()
                    default_modes[:,0] = 1.0
                    default_modes[:,1:4] = getsoftlabel(torch.linspace(0.0, 1.0, steps=args.n_sample), 3, 0, 1)
                    default_modes[:,4:7] = getsoftlabel(torch.linspace(0.0, 1.0, steps=args.n_sample), 3, 0, 1)
                    default_modes[:,7:10] = getsoftlabel(torch.linspace(0.0, 1.0, steps=args.n_sample), 3, 0, 1)
                    default_modes[:,2] = 0
                    default_modes[:,5] = 0
                    default_modes[:,8] = 0
                    sample, _ = g_ema([sample_z], default_mode=default_modes)
                    utils.save_image(
                        sample,
                        os.path.join(exp_dir, "sample", f"{str(i).zfill(6)}_default.png"),
                        nrow=6,
                        normalize=True,
                        range=(-1, 1),
                    )
                    sample_z = sample_z[0].unsqueeze(0).repeat(args.n_sample, 1)
                    sample, _ = g_ema([sample_z], default_mode=default_modes)
                    utils.save_image(
                        sample,
                        os.path.join(exp_dir, "sample", f"{str(i).zfill(6)}_same.png"),
                        nrow=6,
                        normalize=True,
                        range=(-1, 1),
                    )


            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    os.path.join(exp_dir, "checkpoint", f"{str(i).zfill(6)}.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=590001, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=6,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        '--modes', type=int, default=10, help="lambda for generator mode loss"
    )
    parser.add_argument(
        '--md_g', type=float, default=1.0, help="lambda for generator mode loss"
    )
    parser.add_argument(
        '--md_d', type=float, default=1.0, help="lambda for discriminator mode loss"
    )
    parser.add_argument(
        '--cont', action="store_true", help="continue fine-tuning"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='glassesagepose',
        help='define training attributes setting'
    )
    parser.add_argument('--exp_name', type=str, help="exp name")

    parser.add_argument('--output_dir', type=str, default="./exps", help="Path to output directory")

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0
    args.r1 = 0.0002 * args.size * args.size / args.batch

    #multimodal GANs
    generator = Generator_MMD(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, num_class=args.modes
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, num_class=args.modes
    ).to(device)
    g_ema = Generator_MMD(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, num_class=args.modes
    ).to(device)
    g_ema.eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    accumulate(g_ema, generator, 0)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )


    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        if args.cont:
            generator.load_state_dict(ckpt["g"], strict=True)
            discriminator.load_state_dict(ckpt["d"], strict=True)
            g_ema.load_state_dict(ckpt["g_ema"], strict=True)
            g_optim = optim.Adam(
                filter(lambda p: p.requires_grad, generator.parameters()),
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
            )
            d_optim = optim.Adam(
                filter(lambda p: p.requires_grad, discriminator.parameters()),
                lr=args.lr * d_reg_ratio,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
            )

        else:
            generator.load_state_dict(ckpt["g"], strict=False)
            # copy base constants as starting
            generator.input.multiinput = torch.nn.Parameter(ckpt["g"]["input.input"].unsqueeze(4).repeat(1,1,1,1,args.modes).to(device))
            discriminator.load_state_dict(ckpt["d"], strict=False)
            g_ema.load_state_dict(ckpt["g_ema"], strict=False)
            g_ema.input.multiinput = torch.nn.Parameter(ckpt["g_ema"]["input.input"].unsqueeze(4).repeat(1,1,1,1,args.modes).to(device))

            g_optim = optim.Adam(
                filter(lambda p: p.requires_grad, generator.parameters()),
                lr=args.lr * g_reg_ratio,
                betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
            )
            d_optim = optim.Adam(
                filter(lambda p: p.requires_grad, discriminator.parameters()),
                lr=args.lr * d_reg_ratio,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
            )

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size, exp_setting=args.exp_setting)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=8,
        #pin_memory=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
