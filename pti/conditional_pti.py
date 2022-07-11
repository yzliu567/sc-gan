import sys

from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))

import argparse
from conditional_projector import SGProjector, make_image
from model import Generator, Generator_MMD
from criteria.lpips.lpips import LPIPS

from PIL import Image

import torch
from torchvision import transforms, utils

from tqdm import tqdm
from train_mmd import getsoftlabel


class PTIOptimizer(object):

    INIT_IMG_DEFAULT_NAME  = 'pti_cond_init.jpg'
    FINAL_IMG_DEFAULT_NAME = 'pti_cond_inverted.jpg'
    PTI_MODEL_DEFAULT_NAME = 'pti_cond_model.pt'
    PTI_LATENT_DEFAULT_NAME = "pti_cond_latent.pt"

    def __init__(self, 
        base_model_path,        # Path to pre-trained self-conditioned model
        output_path,            # Path to directory where outputs should be saved
        projection_steps=800,   # Number of inversion optimization steps. Not used if loading pre-computed (e.g. e4e) latents
        optimization_steps=350, # Number of PTI generator optimization steps.
        target_resolution=1024, # Full image resolution
        latent_dim=512,         # Size of latent code
        mapping_layers=8,       # Number of mapping network layers
        self_cond_classes=10,   # Number of self-conditional classes. Typically 1 + (3 * number of editing directions).
        ) -> None:

        super().__init__()

        self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
        self.l2_loss = torch.nn.MSELoss(reduction='mean')

        self.g_ema = Generator_MMD(target_resolution, latent_dim, mapping_layers, num_class=self_cond_classes)
        self.g_ema.load_state_dict(torch.load(base_model_path)["g_ema"], strict=False)
        self.g_ema.eval()
        self.g_ema = self.g_ema.cuda()

        self.projector = SGProjector(self.g_ema)
        self.projector.steps = projection_steps

        self.optimizer = torch.optim.Adam(self.g_ema.parameters(), lr=3e-4)
        self.optimization_steps = optimization_steps

        self.transform = transforms.Compose(
            [
                transforms.Resize(target_resolution),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.output_dir = output_path
        self.num_classes = self_cond_classes

    def invert_img(self, img, modes):
        print("Finding pivot code...")

        inverted, (latents, noises) = self.projector.project_image(img, modes)
        return latents
    
    def calc_loss(self, generated_image, real_image):
        loss = 0.0

        l2_loss_val = self.l2_loss(generated_image, real_image)
        loss += l2_loss_val

        loss_lpips = self.lpips_loss(generated_image, real_image)
        loss_lpips = torch.squeeze(loss_lpips)
        loss += loss_lpips

        return loss, l2_loss_val, loss_lpips
    
    def train(self, img_path, modes, latent=None):

        img = Image.open(img_path).convert("RGB")

        modes[:, 2] = 0
        modes[:, 5] = 0
        modes[:, 8] = 0

        if latent is None:
            w_pivot = self.invert_img(img, modes)
        else:
            w_pivot = latent

        torch.save(w_pivot, f"{self.output_dir}/{PTIOptimizer.PTI_LATENT_DEFAULT_NAME}")

        real_image = self.transform(img).unsqueeze(0).cuda()

        pbar = tqdm(range(self.optimization_steps))

        print("Optimizing generator weights...")
        for i in pbar:

            generated_image, _ = self.g_ema([w_pivot], input_is_latent=True, default_mode=modes)

            if i == 0:
                img_ar = make_image(generated_image)
        
                img_name = f"{self.output_dir}/{PTIOptimizer.INIT_IMG_DEFAULT_NAME}"
                pil_img = Image.fromarray(img_ar[0])
                pil_img.save(img_name)

            loss, l2_loss_val, loss_lpips = self.calc_loss(generated_image, real_image)

            self.optimizer.zero_grad()

            if loss_lpips <= 0.06:
                break

            loss.backward()
            self.optimizer.step()

            pbar.set_description(
                (
                    f"l2: {l2_loss_val.item():.4f}; lpips: {loss_lpips.item():.4f};"
                    f" total: {loss.item():.4f};"
                )
            )
        
        final_image, _ = self.g_ema([w_pivot], input_is_latent=True, default_mode=modes)

        img_ar = make_image(final_image)
        
        img_name = f"{self.output_dir}/{PTIOptimizer.FINAL_IMG_DEFAULT_NAME}"
        pil_img = Image.fromarray(img_ar[0])
        pil_img.save(img_name)

        torch.save(self.g_ema.state_dict(),
                    f'{self.output_dir}/{PTIOptimizer.PTI_MODEL_DEFAULT_NAME}')

        return w_pivot

    def edit_single_mode(self, modes, latent, mode_idx=1):

        num_samples = 7

        default_modes = torch.zeros([num_samples, self.num_classes]).cuda()
        default_modes[:, 0] = 1

        default_modes[:, mode_idx * 3 + 1: mode_idx * 3 + 4] = getsoftlabel(torch.linspace(0.0, 1.0, steps=num_samples), 3, 0, 1)
        default_modes[:, mode_idx * 3 + 2] = 0

        w_pivot = latent.repeat(num_samples, 1, 1)
        modes = modes.repeat(num_samples, 1)

        if mode_idx == 0:
            modes[:, 5] = 0
            modes[:, 1:4] = default_modes[:, 1:4]
            modes[:, 8] = 0

        if mode_idx == 1:
            modes[:, 4:7] = default_modes[:, 4:7]
            modes[:, 2] = 0
            modes[:, 8] = 0
        
        if mode_idx == 2:
            modes[:, 5] = 0
            modes[:, 2] = 0
            modes[:, 7:10] = default_modes[:, 7:10]

        with torch.no_grad():
            gen, _ = self.g_ema([w_pivot], input_is_latent=True, default_mode=modes)

        for j in range(num_samples):
            utils.save_image(
                gen[j:j+1],
                f"{self.output_dir}/{str(mode_idx)}_{str(j).zfill(2)}.jpg",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

    def edit_multi(self, modes, latent, step_sizes):

        num_samples = 1

        default_modes = torch.zeros([num_samples, self.num_classes]).cuda()
        default_modes[:, 0] = 1

        w_pivot = latent.repeat(num_samples, 1, 1)
        modes = modes.repeat(num_samples, 1)

        modes[:, 2] = 0
        modes[:, 5] = 0
        modes[:, 8] = 0

        gen, _ = self.g_ema([w_pivot], input_is_latent=True, default_mode=modes)
        utils.save_image(
            gen[0:1],
            f"{self.output_dir}/multi_base_{str(0).zfill(2)}.jpg",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        # Sequential editing
        for iter_idx, mode_idx in enumerate([0, 1, 2]):

            # adjust linspace values to control editing step size
            if mode_idx == 0:
                modes[:, 1:4] = getsoftlabel(torch.linspace(step_sizes[0], step_sizes[0], steps=1), 3, 0, 1)

            if mode_idx == 1:
                modes[:, 4:7] = getsoftlabel(torch.linspace(step_sizes[1], step_sizes[1], steps=1), 3, 0, 1)
            
            if mode_idx == 2:
                modes[:, 7:10] = getsoftlabel(torch.linspace(step_sizes[2], step_sizes[2], steps=1), 3, 0, 1)

            with torch.no_grad():
                gen, _ = self.g_ema([w_pivot], input_is_latent=True, default_mode=modes)

            for j in range(num_samples):
                utils.save_image(
                    gen[j:j+1],
                    f"{self.output_dir}/multi_{str(iter_idx)}_{str(mode_idx)}_{str(j).zfill(2)}.jpg",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )

    # Input / output args
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the base model checkpoint"
    )

    parser.add_argument(
        '--output_dir', type=str, required=True, help="Path to output directory"
    )

    parser.add_argument(
        '--img_path', type=str, required=True, help="Path to image to be inverted"
    )

    parser.add_argument(
        '--mode_file', type=str, required=True, help="Path to .pt file with saved mode scores for the image to be inverted (see get_mode_code_part.py)"
    )

    parser.add_argument(
        '--latent_file', type=str, help="Path to .pt file with saved inversion latent codes (e.g. from e4e). Will be used instead of optimization (highly recommended!)"
    )

    parser.add_argument(
        '--mode_idx', type=int, help='If using a multi-image modes file, provide the index of the entry you want to invert'
    )

    parser.add_argument(
        '--latent_idx', type=int, help='If using a multi-image latents file, provide the index of the entry you want to invert'
    )
    
    # Editing args
    parser.add_argument(
        '--skip_pti', action='store_true', help="Skip PTI step and go straight for editing. Assumes the optimization outputs are in the output_dir"
    )

    parser.add_argument(
        '--edit_mode_idx', type=int, default=1, help="Index of mode (edit direction) to modify along."
    )

    parser.add_argument(
        '--step_sizes', nargs='+', type=float, default=[0.8, 0.2, 1.0], help="Step sizes to use for when editing all modes"
    )

    args = parser.parse_args()

    opt = PTIOptimizer(args.ckpt, args.output_dir)

    modes = torch.load(args.mode_file).cuda()
    if args.mode_idx is not None:
        modes = modes[args.mode_idx:args.mode_idx + 1]

    latent = None

    if args.latent_file:
        latent = torch.load(args.latent_file).cuda()

        if args.latent_idx is not None:
            latent = latent[args.latent_idx:args.latent_idx + 1]

    if args.skip_pti:
        print("Skipping PTI...")

        opt.g_ema.load_state_dict(torch.load(f'{args.output_dir}/{PTIOptimizer.PTI_MODEL_DEFAULT_NAME}'), strict=True)
        latent = torch.load(f"{args.output_dir}/{PTIOptimizer.PTI_LATENT_DEFAULT_NAME}")
    else:
        print("Inverting with PTI...")

        latent = opt.train(args.img_path, modes, latent)

    print("Editing image...")
    opt.edit_single_mode(modes, latent, args.edit_mode_idx)
    opt.edit_multi(modes, latent, args.step_sizes)

    print(f"Done! Outputs have been placed in {args.output_dir}")

