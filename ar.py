# v0 - calculates the homography from scratch at each step
import cv2
import numpy as np
import math
from object_module import *
import sys
import aruco_module as aruco 
from my_constants import *
from utils import get_extended_RT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
import tqdm

#marker setup
marker_colored = cv2.imread('data/m1.png')
assert marker_colored is not None, "Could not find the aruco marker image file"
#accounts for lateral inversion caused by the webcam
marker_colored = cv2.flip(marker_colored, 2)

marker_colored =  cv2.resize(marker_colored, (480,480), interpolation = cv2.INTER_CUBIC )
marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)

h,w = marker.shape
#considering all 4 rotations
marker_sig1 = aruco.get_bit_sig(marker, np.array([[0,0],[0,w], [h,w], [h,0]]).reshape(4,1,2))
marker_sig2 = aruco.get_bit_sig(marker, np.array([[0,w], [h,w], [h,0], [0,0]]).reshape(4,1,2))
marker_sig3 = aruco.get_bit_sig(marker, np.array([[h,w],[h,0], [0,0], [0,w]]).reshape(4,1,2))
marker_sig4 = aruco.get_bit_sig(marker, np.array([[h,0],[0,0], [0,w], [h,w]]).reshape(4,1,2))
sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]

# Colorize texture

texture = cv2.imread('data/3d_objects/fox/texture.png')
mask = cv2.inRange(texture, np.array([41, 135, 219]), np.array([41, 135, 219]))
masked_image = np.copy(texture)
masked_image[mask != 0] = [0, 0, 0]
cmap = 'hsv'
step=255
#change_texture_color(texture,cmap,step)
texture_color = np.load('data/3d_objects/fox/fox_texture_{}.npz'.format(cmap))['texture_color']

s = np.linspace(2,6,step)

# Camera setup
print("trying to access the webcam")
cv2.namedWindow("webcam")
vc = cv2.VideoCapture(0)
assert vc.isOpened(), "couldn't access the webcam"
rval, frame = vc.read()
assert rval, "couldn't access the webcam"

h2, w2,  _ = frame.shape
h_canvas = max(h, h2)
w_canvas = w + w2

i=0
flag = 0
while rval:
    rval, frame = vc.read() #fetch frame from webcam
    key = cv2.waitKey(20) 
    if key == 27: # Escape key to exit the program
        break
    canvas = np.zeros((h_canvas, w_canvas, 3), np.uint8) #final display
    canvas[:h, :w, :] = marker_colored #marker for reference
    success, H = aruco.find_homography_aruco(frame, marker, sigs)
    # success = False
    if not success:
        # print('homograpy est failed')
        canvas[:h2 , w: , :] = np.flip(frame, axis = 1)
        cv2.imshow("webcam", canvas )
        continue

    R_T = get_extended_RT(A, H)
    transformation = A.dot(R_T)
    
    texture = texture_color[i]
    texture[mask == 0] = [0, 0, 0]
    
    final_image = texture + masked_image
    obj = three_d_object('data/3d_objects/fox/low-poly-fox-by-pixelmannen.obj', final_image)
    
    # availability: frame, marker, transformation, obj
    augmented = np.flip(augment(frame, obj, transformation, marker), axis = 1)
    canvas[:h2 , w: , :] = augmented
    cv2.imshow("webcam", canvas)

    if(flag==0):
        i+=1
        if(i>=step):
            flag=1
    if(flag==1):
        i-=1
        if(i<=0):
            flag=0

