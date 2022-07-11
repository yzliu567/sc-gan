from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		if opts.mmd:
			if 'afhq' in opts.dataset_type:
				self.is_hats = np.load('../distances/catage_LARGE.npy')
				self.pose_val = np.load('../distances/catyaw_LARGE.npy')
				self.is_glasses = np.load('../distances/catyaw_LARGE.npy')
			else:
				self.is_glasses = np.loadtxt('../distances/glasses_LARGE.txt')
				self.is_hats = np.loadtxt('../distances/age_LARGE.txt')
				self.pose_val = np.loadtxt('../distances/pose_LARGE.txt')

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im
		if not self.opts.mmd:
			return from_im, to_im
		else:
			return from_im, to_im, self.is_glasses[index], self.is_hats[index], self.pose_val[index]
