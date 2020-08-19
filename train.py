import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision
import torchvision.utils as vutils
from torch.autograd import Variable
from architecture import Generator, Discriminator
from dataset import DataSet
from PIL import Image
from matplotlib import pyplot as plt


def get_data_loader(mode = 'train', root = 'data'):

	transform = {
		'train': transforms.Compose([
			transforms.Resize([256, 256]),  # Resize the images
			transforms.RandomHorizontalFlip(),  # Flip the data horizontally
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		]),

		'test': transforms.Compose([
			transforms.Resize([256, 256]),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
	}
	data = torchvision.datasets.ImageFolder(root=root, transform=transform['train'] if mode == 'train' else transform['test'])
	data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=32, shuffle=True, num_workers=4)

	return data_loader

dloader = get_data_loader()
trans = transforms.ToPILImage()


