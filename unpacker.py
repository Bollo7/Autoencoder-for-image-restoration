import os
from glob import glob
from PIL import Image
import shutil
from tqdm import tqdm
import numpy as np
import hashlib


input_dir = 'data/dataset_part_1'
output_dir = 'outputs2'
logfile = 'logfile.txt'

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

		new_filename = str(i).zfill(6) + '.jpg'
		new_path = os.path.join(output_dir, new_filename)

		if hasher(output_dir, file, logfile, abspath) == True:
			shutil.copy(file, new_path)
			#renamer(output_dir)

	logfile.close()
	return len(os.listdir(output_dir))

image(input_dir, output_dir, logfile)