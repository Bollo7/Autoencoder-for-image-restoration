import random
import torch
from PIL import Image
from glob import glob
import os
import torchvision.transforms.functional as TF
import numpy as np
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
	def __init__(self, img_dir = 'output'):
		super(Dataset, self).__init__()

		self.real_paths = glob(f'{img_dir}/real_imgs/*.jpg', recursive=True)
		self.cropped_paths = glob(f'{img_dir}/cropped_imgs/*.jpg', recursive=True)
		#self.mask = glob(f'{img_dir}/crops/*.jpg', recursive=True)
		#self.target = glob(f'{img_dir}/targets/*.jpg', recursive=True)

	def transform(self, image, label):

		image = TF.to_tensor(image)
		#mask = TF.to_tensor(mask)
		label = TF.to_tensor(label)
		#target = TF.to_tensor(target)

		return image, label

	def __getitem__(self, index):

		if torch.is_tensor(index):
			index = index.tolist()

		img_real = Image.open(self.real_paths[index])
		img_cropped = Image.open(self.cropped_paths[index])
		#mask = Image.open(self.mask[index])
		#target = Image.open(self.target[index])
		img_real, img_cropped = self.transform(img_real, img_cropped)

		return (img_real, img_cropped)

	def __len__(self):

		return len(self.real_paths)



class Test_data(torch.utils.data.Dataset):
	def __init__(self, imgs):
		super(Test_data, self).__init__()

		self.imgs = imgs

	def __getitem__(self, index):

		if torch.is_tensor(index):
			index = index.tolist()

		imgs = Image.fromarray(self.imgs[index])
		imgs = self.transform(imgs)

		return imgs

	def __len__(self):

		return len(self.imgs)

