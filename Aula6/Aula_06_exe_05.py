# Aula_06_exe_05.py
#
# Stereo Image Rectification
#
# Filipe Gonçalves - 11/2022

import numpy as np
import cv2
import glob
from functools import partial

def mouse_handler(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        color = np.random.randint(0, 255, 3).tolist()

        cv2.line(img_undistort_l, (int(map1x[0][0]), int(map1y[0][0])), (int(map1x[0][-1]), int(map1y[0][-1])), color, 2)
        cv2.line(img_undistort_r, (int(map2x[0][0]), int(map2y[0][0])), (int(map2x[0][-1]), int(map2y[0][-1])), color, 2)
        cv2.imshow("Left Undistort", img_undistort_l)
        cv2.imshow("Right Undistort", img_undistort_r)

with np.load('stereoParams.npz') as data: 
    intrinsics1 = data['intrinsics1'] 
    distortion1 = data['distortion1']
    intrinsics2 = data['intrinsics2'] 
    distortion2 = data['distortion2']
    R = data['R']
    T = data['T']
    E = data['E']
    F = data['F']
    
# Read images
images_l = sorted(glob.glob('.//images//left*.jpg'))
images_r = sorted(glob.glob('.//images//right*.jpg'))

img_l = cv2.imread(images_l[0])
img_r = cv2.imread(images_r[0])

img_undistort_l = cv2.undistort(img_l, intrinsics1, distortion1)
img_undistort_r = cv2.undistort(img_r, intrinsics2, distortion2)

R1 = np.zeros(shape=(3,3)) 
R2 = np.zeros(shape=(3,3)) 
P1 = np.zeros(shape=(3,4)) 
P2 = np.zeros(shape=(3,4)) 
Q = np.zeros(shape=(4,4))

height, width = img_undistort_l.shape[:2]

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(intrinsics1, distortion1, intrinsics2, distortion2 ,(width, height), R, T, R1, R2, P1, P2, Q, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))

map1x, map1y = cv2.initUndistortRectifyMap(intrinsics1, distortion1, R1, P1, (width,height), cv2.CV_32FC1) 
map2x, map2y = cv2.initUndistortRectifyMap(intrinsics2, distortion2, R2, P2, (width,height), cv2.CV_32FC1)

cv2.imshow("Left Undistort", img_undistort_l)
cv2.imshow("Right Undistort", img_undistort_r)

cv2.setMouseCallback("Left Undistort", mouse_handler)
cv2.setMouseCallback("Right Undistort", mouse_handler)

cv2.waitKey(-1)
cv2.destroyAllWindows()