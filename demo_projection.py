#!/usr/bin/env python

'''
Planar augmented reality
==================
This sample shows an example of augmented reality overlay over a planar object
tracked by PlaneTracker from plane_tracker.py. solvePnP function is used to
estimate the tracked object location in 3d space.
video: http://www.youtube.com/watch?v=pzVbhxx6aog
Usage
-----
plane_ar.py [<video source>]
Keys:
   SPACE  -  pause video
   c      -  clear targets
Select a textured planar object to track by drawing a box with a mouse.
Use 'focal' slider to adjust to camera focal length for proper video augmentation.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# Simple model of a house - cube with a triangular prism "roof"
# We will use this "house" to test our proposed projection
# Even though we already have a homography matrix, this method provides
# a solution for camera displacement easily. It can still be done with a series
# of related function calls (of course each needs to be explained briefly in thesis)
# yet this is cleaner and easier to understand. But we need to see that ugly birdview image as
# a whole :(
ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                       [0, 0.5, 2], [1, 0.5, 2]])
ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]

if __name__ == '__main__':

    #Read original and birdview images
    org = cv2.imread("birdview.jpg")
    warped = cv2.imread("segmented_birdview.jpg")

    h,w = org.shape[:2]

    ppa     = (w/2., h/2.)
    fm      = 7.5/1e3
    sensor  = (7.14e-3, 5.36e-3)
    focal   = w*(fm / sensor[0])

    #Match points between
    K = np.float64([
        [focal,0, ppa[0]],
        [0,focal, ppa[1]],
        [0,0,1]
    ])

    quad_3d = np.float64([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]])
    wrapped_3d = np.float64([[[427, 0]], [[858, 0]], [[w, h]], [[0, h]]])
    wrapped_3d = np.float64([[427, 0], [858, 0], [w, h], [0, h]])

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    dist_coef = np.zeros(4)
    _ret, rvec, tvec = cv2.solvePnP(quad_3d, wrapped_3d, K, dist_coef )

    print(rvec)
    print(tvec)
