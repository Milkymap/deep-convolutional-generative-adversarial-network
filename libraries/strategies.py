import cv2  

import pickle 
import numpy as np 
import operator as op 
import itertools as it, functools as ft 


import torch as th
import torchvision as tv 

from os import path 
from glob import glob 
from torchvision import transforms as T 

from .log import logger 

def pull_files(target, rule):
	return glob(path.join(target, rule))

def read_image(image_path, by='cv'):
	if by == 'cv':
		return cv2.imread(image_path, cv2.IMREAD_COLOR)
	if by == 'th':
		return tv.io.read_image(image_path)
	raise ValueError(by)

def th2cv(tensor_3d):
	red, green, blue = tensor_3d.numpy()
	return cv2.merge((blue, green, red))

def cv2th(bgr_image):
	blue, green, red = cv2.split(bgr_image)
	return th.from_numpy(np.stack([red, green, blue]))

def to_grid(batch_images, nb_rows=8, padding=10, normalize=False):
	grid_images = tv.utils.make_grid(batch_images, nrow=nb_rows, padding=padding, normalize=normalize)
	return grid_images

def create_image_mapper(size):
	return T.Compose([
		T.ToPILImage(),
		T.Resize(size), 
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

def make_video(source, title, from_pickle=True):
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	writer = cv2.VideoWriter(title, fourcc, 10, (640, 480), True)
	if from_pickle:
		images = pickle.load(open(source, 'rb'))
	else:
		image_paths = glob(path.join(source, '*'))
		images = [ read_image(i_path, by='cv') for i_path in image_paths]

	nb_images = len(images)
	for idx, img in enumerate(images):
		img = cv2.resize(img.astype('uint8'), (640, 480))
		writer.write(img)
		logger.sucess(f'frame was created {idx:05d}/{nb_images}')

	writer.release()
