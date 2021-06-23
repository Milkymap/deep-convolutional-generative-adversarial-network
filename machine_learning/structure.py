import numpy as np 
import operator as op
import itertools as it, functools as ft 


import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import torch.optim as optim 

class Generator(nn.Module):
	def __init__(self, z_dim, fmap_dim, out_channel):
		super(Generator, self).__init__()
		self.depth = np.log2(fmap_dim) - 1
		if fmap_dim > 1 and self.depth.is_integer():
			self.conv_layers = nn.ModuleList([])
			self.activations = []
			self.normalizers = []
			
			p_val = z_dim 
			q_val = fmap_dim
			for idx in range(int(self.depth)):
				stride, padding = 2, 1
				if idx < self.depth - 1:
					if idx < 1:
						stride, padding = 1, 0
					o_val = q_val * 2 ** idx
					non_linear = nn.ReLU(inplace=True)
					norm_layer = nn.BatchNorm2d(o_val)
				else:
					o_val = out_channel 
					non_linear = nn.Tanh()
					norm_layer = nn.Identity()

				self.conv_layers.append(nn.ConvTranspose2d(p_val, o_val, 4, stride, padding))
				self.activations.append(non_linear)
				self.normalizers.append(norm_layer)
				
				p_val = o_val  
		else:
			raise ValueError('fmap_dim should be a power of 2 and greater than 1')

	def forward(self, X):
		reducer = lambda acc, crr: ft.reduce(lambda E, layer : layer(E), crr, acc)
		iterable = list(zip(self.conv_layers, self.normalizers, self.activations))
		return ft.reduce(reducer, iterable, X)


class Descriminator(nn.Module):
	def __init__(self, in_channel, fmap_dim):
		super(Descriminator, self).__init__()
		self.depth = np.log2(fmap_dim) - 1
		if fmap_dim > 1 and self.depth.is_integer():
			self.conv_layers = nn.ModuleList([])
			self.activations = []
			self.normalizers = []
			
			p_val = in_channel
			q_val = fmap_dim
			for idx in range(int(self.depth)):
				if idx < self.depth - 1:
					o_val = q_val * 2 ** idx
					stride, padding = 2, 1
					non_linear = nn.LeakyReLU(0.2, inplace=True)
					norm_layer = nn.BatchNorm2d(o_val)
				else:
					o_val = 1 
					stride, padding = 1, 0
					non_linear = nn.Sigmoid()
					norm_layer = nn.Identity()

				self.conv_layers.append(nn.Conv2d(p_val, o_val, 4, stride, padding))
				self.activations.append(non_linear)
				self.normalizers.append(norm_layer)
				
				p_val = o_val  
		else:
			raise ValueError('fmap_dim should be a power of 2 and greater than 1')

	def forward(self, X):
		reducer = lambda acc, crr: ft.reduce(lambda E, layer : layer(E), crr, acc)
		iterable = list(zip(self.conv_layers, self.normalizers, self.activations))
		return ft.reduce(reducer, iterable, X)


if __name__ == '__main__':
	desc = Descriminator(in_channel=3, fmap_dim=64)
	genr = Generator(z_dim=100, fmap_dim=64, out_channel=3)
	print(desc)
	print(genr)

	X = th.randn((10, 3, 64, 64))
	Y = desc(X)
	print(Y.shape)
	print(Y.view(-1))

