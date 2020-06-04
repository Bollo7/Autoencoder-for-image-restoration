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

	if border_right < 20 or border_left < 20:
		raise ValueError('minimal distance between crop and border is less than 20')
	if border_up < 20 or border_down < 20:
		raise ValueError('minimal distance between crop and border is less than 20')


	target_array = image_array[size_x:center_x, size_y:center_y]

	pad = np.negative(target_array)
	ones_pad = np.ones_like(target_array)

	img[size_x:center_x, size_y:center_y] += pad

	crop_array[size_x:center_x, size_y:center_y] += ones_pad

	image_array = img

	return (image_array, crop_array, target_array)