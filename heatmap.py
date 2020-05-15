#  this will create the heatmap

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

image = np.loadtxt(sys.argv[1])
print("Matrix rows (Images)" + str(image.shape[0]))

image = image
image = image * 255 #TODO: remove scaling here not needed, grayscale
image = np.uint8(image)

# plt.imshow(heatmap)
plt.imshow(image, cmap='jet')
plt.colorbar()

plt.show()