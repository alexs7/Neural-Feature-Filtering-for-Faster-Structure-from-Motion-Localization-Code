import matplotlib.pyplot as plt
import numpy as np
import cv2

image = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/complete_model_visibility_matrix.txt")
# image = image.transpose()
image = image * 2.55
image = np.uint8(image)

heatmap = cv2.applyColorMap(np.uint8(image), cv2.COLORMAP_JET)

cv2.imshow('heatmap', heatmap)
cv2.waitKey()