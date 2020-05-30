import numpy as np
from PIL import Image


def ex4(image_array, crop_size, crop_center):
	if type(image_array) != np.ndarray:
		raise ValueError('not a numpy array')
	if type(crop_size) != tuple or type(crop_center) != tuple:
		raise ValueError('use tuple as input for crop_size and crop_center')
	if len(crop_size) != 2 or len(crop_center) != 2:
		raise ValueError('invalid amount of values')
	if crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
		raise ValueError('crop_size values should be odd!')
	border_x = image_array.shape[0] - (crop_center[0] + crop_size[0])
	border_y = image_array.shape[1] - (crop_center[1] + crop_size[1])
	if border_x < 20:
		raise ValueError('minimal distance between crop and border x is less than 20')
	if border_y < 20:
		raise ValueError('minimal distance between crop and border y is less than 20')

	img = image_array.clone()

	crop_array = np.zeros_like(image_array)
	target_array = image_array[crop_center[0]: crop_center[0] + crop_size[0],
	               crop_center[1]:crop_center[1] + crop_size[1]]

	pad = np.negative(target_array)
	ones_pad = np.ones_like(target_array)

	img[crop_center[0]:crop_center[0] + target_array.shape[0],
	crop_center[1]:crop_center[1] + target_array.shape[1]] += pad

	crop_array[crop_center[0]:crop_center[0] + target_array.shape[0],
	crop_center[1]:crop_center[1] + target_array.shape[1]] += ones_pad

	image_array = img

	return (image_array, crop_array, target_array)