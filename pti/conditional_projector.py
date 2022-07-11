# Adapted from PTI (https://github.com/danielroich/PTI)

# MIT License
#
# Copyright (c) 2021 Daniel Roich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import math

import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

import lpips

class SGProjector(object):
    def __init__(self, g_ema) -> None:
        super().__init__()

        # Default PTI parameters
        self.generator = g_ema
        n_mean_latent = 10000
        self.size = 1024
        self.lr = 0.1
        self.lr_rampup = 0.05
        self.lr_rampdown = 0.25
        self.steps = 1000
        self.noise_scale = 0.05
        self.noise_ramp = 0.75
        self.resize = 256
        self.latent_dim = 512

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, self.latent_dim, device='cuda:0')
            latent_out = self.generator.style(noise_sample)

            self.latent_mean = latent_out.mean(0)
            self.latent_std = ((latent_out - self.latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        self.lpips = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)

    def project_image(self, img, modes, w_plus=False):
        img = self.transform(img).unsqueeze(0).cuda()

        noises_single = self.generator.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(img.shape[0], 1, 1, 1).normal_())

        latent_in = self.latent_mean.detach().clone().unsqueeze(0).repeat(img.shape[0], 1)

        if w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, self.generator.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = torch.optim.Adam([latent_in] + noises, lr=self.lr)

        pbar = tqdm(range(self.steps))
        latent_path = []

        for i in pbar:
            t = i / self.steps
            lr = get_lr(t, self.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = self.latent_std * self.noise_scale * max(0, 1 - t / self.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = self.generator([latent_n], input_is_latent=True, noise=noises, default_mode=modes)

            batch, channel, height, width = img_gen.shape

            if height > self.resize:
                factor = height // self.resize

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = self.lpips(img_gen, img).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, img)

            loss = p_loss + 1e5 * n_loss + 0 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        img_gen, _ = self.generator([latent_path[-1]], input_is_latent=True, noise=noises, default_mode=modes)

        return img_gen, (latent_in, noises)

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )