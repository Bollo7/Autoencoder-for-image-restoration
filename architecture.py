import torch
import numpy as np
import torch.nn as nn

class Generator(torch.nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.t1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),

			#bottleneck
			nn.Conv2d(512, 4000, kernel_size=(4, 4)),
			nn.BatchNorm2d(4000),
			nn.ReLU(),

			nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(4, 4), stride=2, padding=1),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.t1(x)
		return x


class Discriminator(torch.nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.t1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),

			nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=1, padding=0),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.t1(x)
		return x