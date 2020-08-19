import random
import torch
from PIL import Image
from glob import glob
import os
import torchvision.transforms.functional as TF
import numpy as np
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
	def __init__(self, img_dir = 'output', transform = None):
		super(Dataset, self).__init__()
		self.paths = glob(f'{img_dir}/*.jpg', recursive=True)
		self.transform = transform

	def __getitem__(self, index):
		img = Image.open(self.paths[index])
		if self.transform is not None:
			img = self.transform(img)
		return img

	def __len__(self):
		return len(self.paths)

