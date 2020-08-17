import random
import torch.utils.data.dataset as ds
import torch
from PIL import Image
from glob import glob


class DataSet(ds):
	"""

	Takes the images from local folders 'data' and 'test'

	"""
	def __init__(self, split='train'):
		super(DataSet, self).__init__()

		if split == 'train':
			self.paths = glob('/data/**/*.jpg', recursive=True)

		else:
			self.paths = glob('/{:s}/**/*.jpg'.format(split))

	def __getitem__(self, index):
		img = Image.open(self.paths[index])
		return img

	def __len__(self):
		return len(self.paths)


dd = DataSet()