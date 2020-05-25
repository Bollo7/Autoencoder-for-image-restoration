import numpy as np
from PIL import Image

image = Image.open('08_example_image.jpg')
image_array = np.asarray(image)

print(image_array.shape)
