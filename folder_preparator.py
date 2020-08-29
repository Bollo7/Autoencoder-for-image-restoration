"""
Author: Nikita Kolesnichenko
Matr.Nr.: 11778609
Folder preparator
"""

'''
Script takes the unprocessed files from subfolders that contain readily checked images, resizes, then crops the images and 
saves cropped part (target), cropped image and real image into separate folders.
'''

import numpy as np
from PIL import Image
import random
import os
from glob import glob


def cropper(image_array, crop_size, crop_center):
	if type(image_array) != np.ndarray:
		raise ValueError('not a numpy array')
	if type(crop_size) != tuple or type(crop_center) != tuple:
		raise ValueError('use tuple as input for crop_size and crop_center')
	if len(crop_size) != 2 or len(crop_center) != 2:
		raise ValueError('invalid amount of values')
	if crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
		raise ValueError('crop_size values should be odd!')

	img = image_array.copy()

	crop_array = np.zeros_like(image_array)

	size_x =  round(crop_center[0] - (crop_size[0]-1)/2)
	center_x = round(crop_center[0] + (crop_size[0]+1)/2)
	size_y = round(crop_center[1] - (crop_size[1]-1)/2)
	center_y = round(crop_center[1] + (crop_size[1]+1)/2)

	border_right = size_y
	border_left = size_x
	border_up = image_array.shape[0] - center_y
	border_down = image_array.shape[1] - center_x

	# if border_right < 20 or border_left < 20:
	# 	raise ValueError('minimal distance between crop and border is less than 20')
	# if border_up < 20 or border_down < 20:
	# 	raise ValueError('minimal distance between crop and border is less than 20')


	target_array = image_array[int(size_x):int(center_x), int(size_y):int(center_y)]

	pad = np.negative(target_array)
	ones_pad = np.ones_like(target_array)

	img[int(size_x):int(center_x), int(size_y):int(center_y)] += pad

	crop_array[int(size_x):int(center_x), int(size_y):int(center_y)] += ones_pad

	image_array = img

	return (image_array, crop_array, target_array)

root = 'data'
try:
	os.makedirs('output/crops')
	os.mkdir('output/targets')
	os.mkdir('output/cropped_imgs')
	os.mkdir('output/real_imgs')
except:
	pass


def prepare_folders(root = root, dir_to_crops = 'output/crops',
                   dir_to_masks = 'output/targets', crop_img_dir = 'output/cropped_imgs',
                   real_img_dir = 'output/real_imgs', resizing_size = (100, 100)):

	for enum, f in enumerate(sorted(glob(os.path.join(root, '**/*.*'), recursive=True))):
		crop_out_size = (random.randrange(5, 21, 2), random.randrange(5, 21, 2))
		crop_out_center = (random.randrange(35, 65), random.randrange(35, 65))
		img = Image.open(f)
		img = img.resize(resizing_size)
		img_copy = img.copy()
		array = np.asarray(img)
		img, crop, target = cropper(array, crop_out_size, crop_out_center)

		img_copy.save(real_img_dir + f'/real_{enum}.jpg')
		Image.fromarray(img).save(crop_img_dir + f'/cropped_{enum}.jpg')
		Image.fromarray(crop).save(dir_to_crops + f'/crop_{enum}.jpg')
		Image.fromarray(target).save(dir_to_masks + f'/target_{enum}.jpg')
