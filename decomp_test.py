
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from skimage import transform



def get_four_points(im):

    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 10:
            data['points'].append([x, y])

class CameraParameterWriter:

    def __init__(self):
        self.writer = open("unityCamCalibration.txt","w+")
    def write(self, input_line):
        self.writer.write(input_line)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()




if __name__ == '__main__':

    #Read original and birdview images
    test = cv2.imread("./unityTest_test.jpg")
    # test = cv2.imread("./unityTest_test_basic_forward.jpg")
    # test = cv2.imread("./unityTest_test_extreme.jpg")

    # Canvas for homography
    org = np.zeros(test.shape)

    h_org, w_org = org.shape[:2]
    h_test, w_test= test.shape[:2]

    # model =  np.float64([[-10, 0, 0], [10, 0, 0], [10, 0,20], [-10, 0,20]])

    side = get_four_points(test)
    top = get_four_points(org)

    H, status = cv2.findHomography(side, top)

    # For testing the H
    plt.title("Side homode to top")
    side_to_top = cv2.warpPerspective(test, H, (test.shape[1], test.shape[0]))
    plt.imshow(side_to_top)
    plt.show()

    print("MODEL: {}".format(top))

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    dist_coef = np.zeros(4)

    #Write the results to an output file
    camWriter = CameraParameterWriter()

    # For simplification, omit the focal length calculation

    ppa     = (w_test/2., h_test/2.)
    focal_length   = 277.1281
    #
    # print("Unity's Focal Length: {}".format(focal_length))

    #Match points between
    K = np.float64([
        [focal_length, 0, ppa[0]],
        [0, focal_length, ppa[1]],
        [0,0,1]
    ])

    H /= H[2,2]

    image_points = np.float64([[x] for x in side])
    model_points = np.float64([[x[0], 0, h_org - x[1]] for x in top])

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    dist_coef = np.zeros(4)

    _ret, rvec_side, tvec_side = cv2.solvePnP(model_points, image_points, K, dist_coef,
                                              flags=cv2.SOLVEPNP_ITERATIVE)

    plt.title("SIDE")
    plt.imshow(test)
    col = "rgb"

    for center in model_points:

        center_offset = 50
        model_normal = np.float64([center, np.add([center_offset, 0, 0], center),
                                   np.add([0, center_offset, 0], center), np.add([0, 0, center_offset], center)])

        (normal, _) = cv2.projectPoints(model_normal, rvec_side,
                                        tvec_side,
                                        K, dist_coef)

        normal = [x[0] for x in normal]

        for i in range(len(normal) - 1):
            plt.plot([normal[0][0], normal[i + 1][0]], [normal[0][1], normal[i + 1][1]], col[i])

    plt.show()

    rvec_side, _ = cv2.Rodrigues(rvec_side)
    rvec_side = np.transpose(rvec_side)
    tvec_side = np.dot(-rvec_side, tvec_side)
    print("SIDE NEGRO: R: {} T: {}".format(rvec_side, tvec_side))
    cam_height = -tvec_side[2]
    print("The height of the camera {}".format(cam_height))

    # rvec_top, _ = cv2.Rodrigues(rvec_top)
    # print("The pnp normal reversed {}".format(np.dot(rvec_top, np.float64([0, 0, -1]))))
    # print("The pnp normal {}".format(np.dot(rvec_top, np.float64([0, 0, 1]))))
    #
    # rvec_top = np.transpose(rvec_top)
    # tvec_top = np.dot(-rvec_top, tvec_top)
    # print("TOP NEGRO: R: {} T: {}".format(rvec_top, tvec_top))
    # cam_height = -tvec_top[2]
    # print("The height of the camera {}".format(cam_height))
    #
    # # Intrinsic Line
    # camWriter.write("{} {} {} {} {} {}\n".format(w_warped, h_warped, K_warped[0][2], K_warped[1][2], K_warped[0][0],
    #                                              K_warped[1][1]))
    #
    # # Extrinsic Line
    # tvec_top = [t[0] for t in tvec_top]
    # camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(rvec_top[:, 0]), *(rvec_top[:, 1]),
    #                                                                      *(rvec_top[:, 2]), *tvec_top,
    #                                                                      w_warped, h_warped))
    #
    # # Adapt the decomposition placement
    # # Decomposition part
    # _sol, rvec_sol, tvec_sol, nvec_sol = cv2.decomposeHomographyMat(np.linalg.inv(H), K_warped)
    # for i in range(_sol):
    #     rvec = rvec_sol[i];
    #     tvec = tvec_sol[i] * cam_height;
    #     nvec = nvec_sol[i]
    #
    #     # rvec, _ = cv2.Rodrigues(rvec)
    #
    #     print("-" * 100)
    #     print("Rotation {}".format(rvec))
    #     print("Translation {}".format(tvec))
    #     print("Normals {}".format(nvec))
    #
    #     tvec = [t[0] for t in tvec]
    #     camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(rvec[:, 0]), *(rvec[:, 1]),
    #                                                                          *(rvec[:, 2]), *tvec,
    #                                                                          w_warped, h_warped))
    #
    #
    # # ---------------------------------------------------------------------------------
    # # Decomposition test area (was successful on displacement cases)
    # _sol, rvec_sol, tvec_sol, nvec_sol = cv2.decomposeHomographyMat(np.linalg.inv(H), K)
    #
    # top = np.float64([[x] for x in top])
    # _ret, rvec_top, tvec_top = cv2.solvePnP(model, top, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
    #
    # plt.title("TOP")
    # plt.imshow(org)
    # col = "rgb"
    # center = [10,0,0]
    # center_offset = 50
    # model_normal = np.float64([center, np.add([center_offset, 0, 0], center),
    #                            np.add([0, center_offset, 0], center), np.add([0, 0, center_offset], center)])
    #
    # (normal, _) = cv2.projectPoints(model_normal, rvec_top,
    #                                 tvec_top,
    #                                 K, dist_coef)
    #
    # normal = [x[0] for x in normal]
    #
    # for i in range(len(normal) - 1):
    #     plt.plot([normal[0][0], normal[i + 1][0]], [normal[0][1], normal[i + 1][1]], col[i])
    #
    # plt.show()
    # rvec_top, _ = cv2.Rodrigues(rvec_top)
    # print("The pnp normal reversed {}".format(np.dot(rvec_top,np.float64([0,0,-1]))))
    # print("The pnp normal {}".format(np.dot(rvec_top, np.float64([0, 0, 1]))))
    #
    # rvec_top = np.transpose(rvec_top)
    # tvec_top = np.dot(-rvec_top, tvec_top)
    # print("TOP NEGRO: R: {} T: {}".format(rvec_top, tvec_top))
    #
    #
    #
    #
    # side = np.float64([[x] for x in side])
    # _ret, rvec_side, tvec_side = cv2.solvePnP(model, side, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
    #
    # plt.title("Test")
    # plt.imshow(test)
    # col = "rgb"
    # for center in model:
    #     center_offset = 50
    #     model_normal = np.float64([center, np.add([center_offset, 0, 0], center),
    #                                np.add([0, center_offset, 0], center), np.add([0, 0, center_offset], center)])
    #
    #     (normal, _) = cv2.projectPoints(model_normal, rvec_side,
    #                                     tvec_side,
    #                                     K, dist_coef)
    #
    #     normal = [x[0] for x in normal]
    #
    #     for i in range(len(normal) - 1):
    #         plt.plot([normal[0][0], normal[i + 1][0]], [normal[0][1], normal[i + 1][1]], col[i])
    #
    # plt.show()
    #
    # rvec_side, _ = cv2.Rodrigues(rvec_side)
    # rvec_side = np.transpose(rvec_side)
    # tvec_side = np.dot(-rvec_side, tvec_side)
    # print("TEST NEGRO: R: {} T: {}".format(rvec_side, tvec_side))
    # cam_height = -tvec_top[2]
    # print("The height of the camera {}".format(cam_height))
    #
    # #Intrinsic Line
    # camWriter.write("{} {} {} {} {} {}\n".format(w_test, h_test,ppa[0], ppa[1],focal_length,focal_length))
    #
    # #Extrinsic Line
    # tvec_top = [t[0] for t in tvec_top]
    # camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(rvec_top[:,0]),*(rvec_top[:,1]),*(rvec_top[:,2])
    #                                                                ,*tvec_top,w_test, h_test))
    #
    #
    # for i in range(_sol):
    #
    #     rvec = rvec_sol[i]; tvec = tvec_sol[i] * cam_height; nvec = nvec_sol[i]
    #
    #     # rvec, _ = cv2.Rodrigues(rvec)
    #
    #     print("-" * 100)
    #     print("Rotation {}".format(rvec))
    #     print("Translation {}".format(tvec))
    #     print("Normals {}".format(nvec))
    #     tvec = [t[0] for t in tvec]
    #     camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(rvec[:, 0]), *(rvec[:, 1]),
    #                                                                          *(rvec[:, 2]), *tvec,
    #                                                                          w_test, h_test))
    #
    # camWriter.writer.close()
    #
    #
    #
    #
    #
