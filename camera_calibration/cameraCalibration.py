import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.transform as transform

__all__ = ["CameraCalibration"]

class CameraParameterWriter:

    def __init__(self):
        self.writer = open("unityCamCalibration.txt","w+")
    def write(self, input_line):
        self.writer.write(input_line)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

class CameraCalibration:

    camWriter = CameraParameterWriter()

    # TODO: Comment and explain
    @staticmethod
    def extractCameraParameters(image, warped_img, model_points, image_points, K, H=None):

        h_org, w_org, _ = image.shape
        h_warped, w_warped, _ = warped_img.shape

        clean_img = np.copy(image)

        # For displaying the axes of the placed model
        # The image below the vanishing line is used for mapping

        model_points = transform.matrix_transform(image_points, H)

        plt.figure()
        plt.imshow(warped_img)

        for i in range(len(model_points)):
            plt.plot([model_points[i][0], model_points[(i + 1) % len(model_points)][0]],
                     [model_points[i][1], model_points[(i + 1) % len(model_points)][1]], 'b')

        plt.show()

        # On model, the lower right corner will be considered as the axis
        # The Y coordinate needs to be reversed as in Unity, y (z) is forward
        model_points = np.float64([[x[0], 0, h_warped - x[1]] for x in model_points])

        image_points = np.float64([[x] for x in image_points])

        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))

        dist_coef = np.zeros(4)

        # From experiments, p3p seems like the best
        algorithms = {
            "p3p": cv2.SOLVEPNP_P3P}  # , "iterative": cv2.SOLVEPNP_ITERATIVE, "decomp": -1} #, "epnp": cv2.SOLVEPNP_EPNP}
        # algorithms = { "decomp": -1}

        for v, k in enumerate(algorithms):

            if k == "decomp":  # Decomposition version
                _sol, rvec_sol, tvec_sol, nor_sol = cv2.decomposeHomographyMat(np.linalg.inv(H), K)

            else:  # Pnp solutions


                _ret, rvec, tvec = cv2.solvePnP(model_points, image_points,
                                                K, dist_coef,
                                                flags=v)
                if k == "iterative":
                    for iteration in range(100):
                        _ret, rvec, tvec = cv2.solvePnP(model_points, image_points,
                                                        K, dist_coef,
                                                        rvec, tvec,
                                                        useExtrinsicGuess=True,
                                                        flags=v)
                _sol = 1

            for solution in range(_sol):

                if k == "decomp":
                    rvec = rvec_sol[solution]
                    tvec = tvec_sol[solution]
                    nvec = nor_sol[solution]

                plt.title("The solution no:" + str(solution))
                plt.imshow(clean_img)
                col = "rgb"

                for center in model_points:

                    center_offset = 100
                    model_normal = np.float64([center, np.add([center_offset, 0, 0], center),
                                               np.add([0, center_offset, 0], center),
                                               np.add([0, 0, center_offset], center)])
                    #
                    # #TODO: FOR DEBUGGING ONLY: If decomp, project points according to perpendicular camera as it adds distortion
                    # if k == "decomp":
                    #     image_points = np.float64([[x[0], h_warped - x[1]] for x in model_points])
                    #
                    #     _, d_rvec, d_tvec = cv2.solvePnP(model_points, image_points, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
                    #
                    #     (model_normal, _) = cv2.projectPoints(model_normal, d_rvec,
                    #                                     d_tvec,
                    #                                     K, dist_coef)
                    #
                    #     # warped_img = transform.warp(image, np.linalg.inv(H))
                    #     #
                    #     # cv2.line(warped_img, tuple(map(int, model_normal[0][0])), tuple(map(int,model_normal[1][0])), (0, 0, 255), 2)
                    #     # cv2.line(warped_img, tuple(map(int, model_normal[0][0])), tuple(map(int, model_normal[2][0])), (0, 255, 0), 2)
                    #     # cv2.line(warped_img, tuple(map(int, model_normal[0][0])), tuple(map(int, model_normal[3][0])), (255, 0, 0), 2)
                    #     #
                    #     # cv2.imshow("W",warped_img)
                    #     # cv2.waitKey(0)
                    #
                    #     model_normal = [x[0] for x in model_normal]
                    #
                    #     normal = transform.matrix_transform(model_normal, np.linalg.inv(H))
                    #     normal = model_normal
                    #
                    #     cv2.line(warped_img, tuple(map(int, normal[0])), tuple(map(int, normal[1])), (0, 0, 255), 2)
                    #     cv2.line(warped_img, tuple(map(int, normal[0])), tuple(map(int, normal[2])), (0, 255, 0), 2)
                    #     cv2.line(warped_img, tuple(map(int, normal[0])), tuple(map(int, normal[3])), (255, 0, 0), 2)
                    #
                    #     center = [0,0,0]
                    #     model_normal = np.float64([center, np.add([center_offset, 0, 0], center),
                    #                                np.add([0, center_offset, 0], center),
                    #                                np.add([0, 0, center_offset], center)])

                    (normal, _) = cv2.projectPoints(model_normal, rvec,
                                                    tvec,
                                                    K, dist_coef)

                    normal = [x[0] for x in normal]

                    for i in range(len(normal) - 1):
                        plt.plot([normal[0][0], normal[i + 1][0]], [normal[0][1], normal[i + 1][1]], col[i])

                plt.show()

                if k != "decomp":
                    rvec, _ = cv2.Rodrigues(rvec)

                # As the pnp gives us the object translation, we need to get the camera translation by transposing
                rvec = np.transpose(rvec)
                tvec = np.dot(-rvec, tvec)

                if k == "decomp":
                    print("Normal {}".format(nvec))

                image = np.copy(clean_img)

                # Intrinsic Line
                CameraCalibration.camWriter.write("{} {} {} {} {} {}\n".format(w_org, h_org, K[0][2], K[1][2], K[0][0], K[1][1]))

                # Extrinsic Line
                tvec = [t[0] for t in tvec]
                CameraCalibration.camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(rvec[:, 0]), *(rvec[:, 1]),
                                                                                     *(rvec[:, 2]), *tvec, w_warped,
                                                                                     h_warped))

                print(k)
                print("Rotation Rodrigues {}".format(rvec))

                rvec, _ = cv2.Rodrigues(rvec)

                rvec[0] = np.rad2deg(rvec[0])
                rvec[1] = np.rad2deg(rvec[1])
                rvec[2] = np.rad2deg(rvec[2])
                print("Rotation Euler Angles {}".format(rvec))
                print("Translation {}".format(tvec))
