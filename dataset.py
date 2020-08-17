import random
import torch
from PIL import Image
from glob import glob
import os
import torchvision.transforms.functional as TF
import numpy as np
import torchvision.transforms as transforms

class DataSet(torch.utils.data.Dataset):
	"""

	Takes the images from local folder 'data' and performs rotation

	"""
	def __init__(self):
		#super(DataSet, self).__init__()
		self.paths = glob('data/**/*.jpg', recursive=True)

	def __getitem__(self, index):
		img = Image.open(self.paths[index])
		return img

	def __len__(self):
		return len(self.paths)


class Rotation(DataSet):

	def __init__(self, dataset: DataSet, angle: float = 45.,
	             transform_chain: transforms.Compose = None):
		self.dataset = dataset
		self.angle = angle
		self.transform_chain = transform_chain

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		image, idx = self.dataset.__getitem__(item)
		image = transforms.functional.to_pil_image(image)

		if self.transform_chain is not None:
			image = self.transform_chain

		rotated_data = TF.rotate(image, angle=self.angle)
		image = TF.resized_crop(image, i=8, j=8, h=16, w=16, size=32)
		rotated_data = TF.resized_crop(rotated_data, i=8, j=8, h=16, w=16, size=32)
		image = np.asarray(image, dtype=np.float32)
		rotated_data = np.asarray(rotated_data, dtype=np.float32)
		mean = image.mean()
		std = image.std()
		image[:] -= mean
		image[:] /= std
		rotated_data[:] -= mean
		rotated_data[:] /= std

		full_inp = np.zeros(shape=(*image.shape, 1), dtype = image.dtype)
		full_inp[..., 0] = image
		full_inp[np.arange(full_inp.shape[0]), :, 1] = np.linspace(start=-1, stop=1, num=full_inp.shape[1])
		full_inp[:, :, 2] = np.transpose(full_inp[:, :, 1])

		full_inp = TF.to_tensor(full_inp)
		rotated_data = TF.to_tensor(rotated_data)

		return full_inp, rotated_data, idx


dd = DataSet()
rot = Rotation(dd)


