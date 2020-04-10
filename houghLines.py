import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt

# img = cv2.imread('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images/frame_1585312454258.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# # minLineLength = 10
# # maxLineGap = 1
# # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 10,minLineLength = 10 ,maxLineGap = 5)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow("result", img)
# cv2.waitKey(0)

# # Read image
# img = cv2.imread('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images/frame_1585312454258.jpg', cv2.IMREAD_COLOR) # road.png is the filename
# # Convert the image to gray-scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Find the edges in the image using canny detector
# edges = cv2.Canny(gray, 50, 200)
# # Detect points that form a line
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 220, minLineLength=10, maxLineGap=1)
# # Draw lines on the image
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# # Show result
# cv2.imshow("result", img)
# cv2.waitKey(0)

#Read gray image

img = cv2.imread('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images/frame_1585312454258.jpg', cv2.IMREAD_COLOR) # road.png is the filename
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

#Detect lines in the image
lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
# pdb.set_trace()

new_lines = np.empty([0,4])
gradients = np.empty([0,1])
for i in range(lines.shape[0]):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]
    grad = (y1 - y2)/ (x2 - x1)
    gradients = np.r_[gradients, np.array([grad]).reshape(1, 1)]
    if(grad):
        new_lines = np.r_[new_lines, np.array([x1, y1, x2, y2]).reshape(1, 4)]

np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/gradients.txt', gradients)

new_lines = new_lines.reshape(len(new_lines), 1, 4).astype(np.float32)

# pdb.set_trace()

#Draw detected lines in the image
drawn_img = lsd.drawSegments(img,new_lines)

#Show image
cv2.imshow("LSD",drawn_img )
cv2.waitKey(0)