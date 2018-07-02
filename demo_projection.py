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
Use 'focal_unity' slider to adjust to camera focal_unity length for proper video augmentation.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from skimage import transform

# Simple model of a house - cube with a triangular prism "roof"
# We will use this "house" to test our proposed projection
# Even though we already have a homography matrix, this method provides
# a solution for camera displacement easily. It can still be done with a series
# of related function calls (of course each needs to be explained briefly in thesis)
# yet this is cleaner and easier to understand. But we need to see that ugly birdview image as
# a whole :(

#Useless
# ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
#                        [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
#                        [0, 0.5, 2], [1, 0.5, 2]])
# ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
#             (4, 5), (5, 6), (6, 7), (7, 4),
#             (0, 4), (1, 5), (2, 6), (3, 7),
#             (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]

class CameraParameterWriter:

    def __init__(self):
        self.writer = open("unityCamCalibration.txt","w+")
    def write(self, input_line):
        self.writer.write(input_line)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()


if __name__ == '__main__':

    #Read original and birdview images
    org = cv2.imread("./unityTest_2.jpg")

    h,w = org.shape[:2]

    #FOCAL CALCULATION TEST
    horizon_pts = np.array([[0,266,1],[942,266,1]]) #Normal, doesn't look right at all
    horizon = np.cross(horizon_pts[0],horizon_pts[1])
    horizon = horizon / np.sqrt(horizon[0]**2 + horizon[1]**2)

    postures = np.array([[310,329,1],[317,431,1], [667,328,1], [660,432,1]])
    lines = np.array([])
    for p in range(len(postures)//2):
        line = np.cross(postures[p*2],postures[p*2+1])
        line = line / np.sqrt(line[0]**2 + line[1]**2)
        lines = np.append(lines,np.array(line))

    zenith = np.cross(lines[:3], lines[3:])
    # zenith = zenith / np.sqrt(zenith[0]**2 + zenith[1]**2)
    zenith = zenith / zenith[2]

    plt.title("Source image")
    plt.imshow(org)
    plt.plot(horizon_pts[:, 0], horizon_pts[:, 1], 'r')
    plt.plot([zenith[0]], [zenith[1]], 'ro')
    plt.show()

    image_center = [w,h,1]
    center_vp_dist = (np.linalg.norm(zenith[:2] - image_center[:2]))
    center_hor_dist = np.dot(np.array(image_center),horizon)
    focal_length = np.sqrt(center_vp_dist * center_hor_dist)
    print("Calculated focal_unity length {}".format(focal_length))

    ppa     = (w/2., h/2.)
    # focal_length   = h * (1 / np.tan(np.deg2rad(60))) / 2
    #
    # print("Unity's Focal Length: {}".format(focal_length))

    #Match points between
    K = np.float64([
        [focal_length, 0, ppa[0]],
        [0, focal_length, ppa[1]],
        [0,0,1]
    ])

    # #Quad 3d is the area where the model's base should be located in the image (From top)
    # model = np.float64([[-10, 0, 0], [10, 0, 0], [25, 60, 0], [-25, 60, 0]])
    # #This is what is seen on the image (From perspective)
    # wrapped_3d = np.float64([[[115, 495]], [[827,495]], [[636,307]], [[312, 307]]])

    #Quad 3d is the area where the model's base should be located in the image (From top)
    model = np.float64([[-10, 0, 0], [10, 0, 0], [10, 20, 0], [-10, 20, 0]])
    model_normal = np.float64([[0,0,0],[10,0,0],[0,10,0],[0,0,20]])
    #This is what is seen on the image (From perspective)
    wrapped_3d = np.float64([[[320, 422]], [[628,423]], [[627,111]], [[318, 111]]])



def extractCameraParameters(modelPoints, imagePoints, K):
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    dist_coef = np.zeros(4)

    # TODO: Write these results to an output file
    camWriter = CameraParameterWriter()

    #From experiments, p3p seems like the best
    algorithms = { "iterative": cv2.SOLVEPNP_ITERATIVE, "p3p": cv2.SOLVEPNP_P3P, "epnp": cv2.SOLVEPNP_EPNP}

    for v,k in enumerate(algorithms):

        _ret, rvec, tvec = cv2.solvePnP(model, wrapped_3d, K, dist_coef, flags=v)

        (normal, _) = cv2.projectPoints(model_normal, rvec,
                                                        tvec,
                                                         K, dist_coef)

        cv2.line(org, tuple(map(int,normal[0][0])), tuple(map(int,normal[1][0])), (0, 0, 255), 2)
        cv2.line(org, tuple(map(int, normal[0][0])), tuple(map(int, normal[2][0])), (0, 255, 0), 2)
        cv2.line(org, tuple(map(int, normal[0][0])), tuple(map(int, normal[3][0])), (255, 0, 0), 2)

        rvec, _ = cv2.Rodrigues(rvec)

        print(k)
        print("Rotation {}".format(rvec))
        print("Translation {}".format(tvec))

        # Display image
        cv2.imshow("Output", org)
        cv2.waitKey(0)

        #Intrinsic Line
        camWriter.write("{} {} {} {} {} {}\n".format(w,h,ppa[0], ppa[1],focal_length,focal_length))

        #Extrinsic Line
        tvec = [t[0] for t in tvec]
        camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(rvec[:,0]),*(rvec[:,1]),*(rvec[:,2]),*tvec))



