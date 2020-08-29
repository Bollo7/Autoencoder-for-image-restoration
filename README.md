# Autoencoder-for-image-restoration - PyTorch project

Autoencoders are great for feature learning and good fit for restoration of missing parts of the image. This project was made from scratch and is a part of subject at my university.

Quick guide on usage:

folder_preparator.py --> set the path to a folder with images to prepare them for the network (resize to 100x100 and cut-out the random rectangle)

image.py --> script to separate the valid images from the uncleaned directory

dataset.py --> dataset class 

architecture.py --> holds different autoencoder architectures

train.ipynb --> jupyter notebook that contains the main training loop and test restoration

At the moment project files need some brush-up to make code more reproducible and cleaner. Will be updated soon...
