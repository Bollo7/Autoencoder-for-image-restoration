"""
Author: Nikita Kolesnichenko


Image checker/unpacker
"""

''' 
Script checks the images for 5 different criterions, every flaw is logged into logfile,
valid images are copied into separate folder and renamed. If copied image already exists in the folder (matching hashes),
then it is not copied and filename with error number is written into logfile.
'''


import os
from glob import glob
from PIL import Image
import shutil
from tqdm import tqdm
import numpy as np
import hashlib

input_dir = 'data'
output_dir = 'outputs'
logfile = 'logfile.txt'


def open_img(file, logfile, abspath): #3
	try:
		Image.open(file)
	except IOError:
		logfile.write(f'{os.path.relpath(file, abspath)};3\n')
	else:
		return True

# take the
def hasher(output_dir, input_dir, logfile, abspath): #6
	lst = []
	for img_out in os.listdir(output_dir):
		with open(os.path.join(output_dir, img_out), 'rb') as k:
			lst.append(hashlib.md5(k.read()).hexdigest())

	with open(input_dir, 'rb') as f:
		hash = hashlib.md5(f.read()).hexdigest()
		if hash not in lst:
			return True
		else:
			logfile.write(f'{os.path.relpath(input_dir, abspath)};6\n')

# check the shape of an image and dimensionality (should be 2D and 100x100 matrix)
def np_check(im, file, logfile, abspath): #5
	arr = np.asarray(im)
	if arr.shape >= (100, 100) and arr.ndim == 2:
		return True
	else:
		logfile.write(f'{os.path.relpath(file, abspath)};5\n')

# check the variance of pixels within the image
def var_check(im, file, logfile, abspath): #4
	arr = np.asarray(im)
	if np.var(arr) > 0:
		return True
	else:
		logfile.write(f'{os.path.relpath(file, abspath)};4\n')

# checks size of file (should not be more than 1 Mb)
def check_size(file, logfile, abspath): #2
	if os.path.getsize(file) >= 10000:
		return True
	else:
		logfile.write(f'{os.path.relpath(file, abspath)};2\n')

# renaming function
def renamer(output_dir):
	for num, file in enumerate(os.listdir(output_dir)):
		oldext = os.path.splitext(file)[1]
		new_filename = str(num +1).zfill(6) + oldext
		new_filename = os.path.join(output_dir, new_filename)
		old_filename = os.path.join(output_dir, file)
		#new_path = os.path.join(output_dir, new_filename)
		os.rename(old_filename, new_filename)



def image(input_dir, output_dir, logfile):

	abspath = os.path.basename(input_dir)

	try:
		if os._exists(output_dir) == False:
			os.mkdir(output_dir)
	except FileExistsError:
		print('Directory "outputs" already exists!')

	logfile = open(logfile, 'w+')

	le = []
	exts = ['jpg', 'jpeg', 'JPG', 'JPEG']
	for f in sorted(glob(os.path.join(input_dir, '**/*.*'), recursive=True)):
		if any(f.endswith(ext) for ext in exts):
			le.append(f)
		else:
			logfile.write(f'{os.path.relpath(f, abspath)};1\n') #1. extension checks

	dirs = le

	for i, file in tqdm(enumerate(dirs, 1), desc='Processing copied files!',
		                          total=len(input_dir)):
		if open_img(file, logfile, abspath) == True:
			img = Image.open(file)

			if check_size(file, logfile, abspath) == True:# 2. size of file is > 10kB
				if var_check(img, file, logfile, abspath) == True:  # 4. image variance > 0
					if np_check(img, file, logfile, abspath) == True:  # 5. shape is H,W and >= 100,100
						oldext = os.path.splitext(file)[1]
						new_filename = str(i).zfill(6) + '.jpg'
						new_path = os.path.join(output_dir, new_filename)
						# 6. check hash-values of files from input and output
						if hasher(output_dir, file, logfile, abspath) == True:
							shutil.copy(file, new_path)
							renamer(output_dir)
	logfile.close()
	return len(os.listdir(output_dir))

image(input_dir, output_dir, logfile)

