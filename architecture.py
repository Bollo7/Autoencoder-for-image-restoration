import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()

		# encoder
		self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)

		# decoder
		self.dec1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2)
		self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
		self.dec3 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
		self.dec4 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2)
		self.out = nn.Conv2d(256, 1, kernel_size=10, padding=0)
		#self.lin = nn.Linear(108, 1)

	def forward(self, x):
		# encoder
		x = F.relu(self.enc1(x))
		x = self.pool(x)
		x = F.relu(self.enc2(x))
		x = self.pool(x)
		x = F.relu(self.enc3(x))
		x = self.pool(x)
		x = F.relu(self.enc4(x))
		x = self.pool(x)

		# decoder
		x = F.relu(self.dec1(x))
		x = F.relu(self.dec2(x))
		x = F.relu(self.dec3(x))
		x = F.relu(self.dec4(x))
		x = F.sigmoid(self.out(x))
		#x = self.lin(x)
		#x = x.view(-1, 100, 100)
		#x = x.flatten()

		return x


class small_ae(nn.Module):
	def __init__(self):
		super(small_ae, self).__init__()

		# encoder
		self.enc1 = nn.Conv2d(1, 32, kernel_size=3)
		self.enc2 = nn.Conv2d(32, 64, kernel_size=3)

		# decoder
		self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3)
		self.dec2 = nn.ConvTranspose2d(32, 1, kernel_size=3)

	def forward(self, x):
		# encoder
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))

		# decoder
		x = F.relu(self.dec1(x))
		x = F.relu(self.dec2(x))

		return x


class mid_ae(nn.Module):
	def __init__(self):
		super(mid_ae, self).__init__()

		# encoder
		self.enc1 = nn.Conv2d(1, 32, kernel_size=3)
		self.enc2 = nn.Conv2d(32, 64, kernel_size=3)
		self.enc3 = nn.Conv2d(64, 128, kernel_size=3)

		# decoder
		self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
		self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=3)
		self.dec3 = nn.ConvTranspose2d(32, 1, kernel_size=3)

	def forward(self, x):
		# encoder
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))

		# decoder
		x = F.relu(self.dec1(x))
		x = F.relu(self.dec2(x))
		x = F.relu(self.dec3(x))

		return x

class large_ae(nn.Module):
	def __init__(self):
		super(large_ae, self).__init__()

		# encoder
		self.enc1 = nn.Conv2d(1, 16, kernel_size=3)
		self.enc2 = nn.Conv2d(16, 32, kernel_size=3)
		self.enc3 = nn.Conv2d(32, 64, kernel_size=3)
		self.enc4 = nn.Conv2d(64, 128, 3)

		# decoder
		self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
		self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=3)
		self.dec3 = nn.ConvTranspose2d(32, 16, kernel_size=3)
		self.dec4 = nn.ConvTranspose2d(16, 1, 3)

	def forward(self, x):
		# encoder
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))

		# decoder
		x = F.relu(self.dec1(x))
		x = F.relu(self.dec2(x))
		x = F.relu(self.dec3(x))
		x = F.relu(self.dec4(x))

		return x

class larger_ae(nn.Module):
	def __init__(self):
		super(larger_ae, self).__init__()

		# encoder
		self.enc1 = nn.Conv2d(1, 16, kernel_size=3)
		self.enc2 = nn.Conv2d(16, 32, kernel_size=3)
		self.enc3 = nn.Conv2d(32, 64, kernel_size=3)
		self.enc4 = nn.Conv2d(64, 128, 3)
		self.enc5 = nn.Conv2d(128, 256, 3)

		# decoder
		self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3)
		self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3)
		self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=3)
		self.dec4 = nn.ConvTranspose2d(32, 16, 3)
		self.dec5 = nn.ConvTranspose2d(16, 1, 3)

	def forward(self, x):
		# encoder
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = F.relu(self.enc5(x))

		# decoder
		x = F.relu(self.dec1(x))
		x = F.relu(self.dec2(x))
		x = F.relu(self.dec3(x))
		x = F.relu(self.dec4(x))
		x = F.relu(self.dec5(x))


		return x

class larger_ae_mp(nn.Module):
	def __init__(self):
		super(larger_ae_mp, self).__init__()

		# encoder
		self.enc1 = nn.Conv2d(1, 16, kernel_size=4)
		self.enc2 = nn.Conv2d(16, 32, kernel_size=4)
		self.enc3 = nn.Conv2d(32, 64, kernel_size=4)
		self.enc4 = nn.Conv2d(64, 128, 4)
		self.enc5 = nn.Conv2d(128, 256, 4)

		# decoder
		self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4)
		self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4)
		self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=4)
		self.dec4 = nn.ConvTranspose2d(32, 16, 4)
		self.dec5 = nn.ConvTranspose2d(16, 1, 4)

	def forward(self, x):
		# encoder
		x = F.relu(self.enc1(x))
		x = F.relu(self.enc2(x))
		x = F.relu(self.enc3(x))
		x = F.relu(self.enc4(x))
		x = F.relu(self.enc5(x))

		# decoder
		x = F.relu(self.dec1(x))
		x = F.relu(self.dec2(x))
		x = F.relu(self.dec3(x))
		x = F.relu(self.dec4(x))
		x = F.relu(self.dec5(x))

		return x


# Source: https://github.com/yjn870/REDNet-pytorch/blob/master/model.py

class REDNet10(nn.Module):
	def __init__(self, num_layers=5, num_features=64):
		super(REDNet10, self).__init__()
		conv_layers = []
		deconv_layers = []

		conv_layers.append(nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)))
		for i in range(num_layers - 1):
			conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

		for i in range(num_layers - 1):
			deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
		deconv_layers.append(nn.ConvTranspose2d(num_features, 1, kernel_size=3, stride=1, padding=1))

		self.conv_layers = nn.Sequential(*conv_layers)
		self.deconv_layers = nn.Sequential(*deconv_layers)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		residual = x
		out = self.conv_layers(x)
		out = self.deconv_layers(out)
		out += residual
		out = self.relu(out)
		return out


class REDNet20(nn.Module):
	def __init__(self, num_layers=10, num_features=64):
		super(REDNet20, self).__init__()
		self.num_layers = num_layers

		conv_layers = []
		deconv_layers = []

		conv_layers.append(nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)))
		for i in range(num_layers - 1):
			conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

		for i in range(num_layers - 1):
			deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))

		deconv_layers.append(nn.ConvTranspose2d(num_features, 1, kernel_size=3, stride=1, padding=1))

		self.conv_layers = nn.Sequential(*conv_layers)
		self.deconv_layers = nn.Sequential(*deconv_layers)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):

		residual = x

		conv_feats = []
		for i in range(self.num_layers):
			x = self.conv_layers[i](x)
			if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
				conv_feats.append(x)

		conv_feats_idx = 0
		for i in range(self.num_layers):
			x = self.deconv_layers[i](x)
			if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
				conv_feat = conv_feats[-(conv_feats_idx + 1)]
				conv_feats_idx += 1
				x = x + conv_feat
				x = self.relu(x)

		x += residual
		x = self.relu(x)

		return x