import numpy as np
import cv2
import utils
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

    @staticmethod
    def displayPlacement (image, model_points, rvec, tvec, K, dist_coef, horizon_method):

        col = "rgb"
        model_on_image = []

        fig, ax = plt.subplots()

        ax.set_title("The resulting placement from " + horizon_method)
        ax.imshow(image)

        # Visualize the model endpoints with axes
        for center in model_points:

            center_offset = 100
            model_normal = np.float64([center, np.add([center_offset, 0, 0], center),
                                       np.add([0, -center_offset, 0], center),
                                       np.add([0, 0, center_offset], center)])

            (normal, _) = cv2.projectPoints(model_normal,
                                            rvec,
                                            tvec,
                                            K,
                                            dist_coef)

            normal = [x[0] for x in normal]

            for i in range(len(normal) - 1):
                ax.plot([normal[0][0], normal[i + 1][0]],
                         [normal[0][1], normal[i + 1][1]],
                         col[i])

            model_on_image.append(normal[0])

        plt.show(fig)

        return np.array(model_on_image), fig

    @staticmethod
    def extractCameraParameters(horizon_method, image, warped_img, model_points, image_points, K):

        h_org, w_org, _ = image.shape
        h_warped, w_warped, _ = warped_img.shape

        clean_img = np.copy(image)

        # For displaying the axes of the placed model
        # The image below the vanishing line is used for mapping
        plt.figure()
        plt.title("Before correction, mesh model")
        plt.imshow(warped_img)

        for i in range(len(model_points)):
            plt.plot([model_points[i][0], model_points[(i + 1) % len(model_points)][0]],
                     [model_points[i][1], model_points[(i + 1) % len(model_points)][1]], 'b')

        plt.show()

        # Initially, the model is left in openCV coordinate system, as homography correction
        # process is only valid in identical coordinate systems.

        # temp_image_points = np.float64([[x[0], h_org - x[1]] for x in image_points])
        temp_image_points = np.float64([[x] for x in image_points])

        # The corrected image would grow large so we need to assign extra space for it
        model_resize = 2
        warped_image_large = np.zeros((h_warped * model_resize, w_warped * model_resize, 3))
        warped_image_large[int(h_warped * ((model_resize - 1) / 2)) : int(h_warped * ((model_resize + 1) / 2))
                        , int(w_warped * ((model_resize - 1) / 2)): int(w_warped * ((model_resize + 1) / 2)), :] = warped_img

        h_warped_large, w_warped_large, _ = warped_image_large.shape

        model_points_large = np.array(list( map( lambda x: x + [w_warped * ((model_resize - 1) / 2),
                                                                h_warped * ((model_resize - 1) / 2)], model_points)))


        plt.figure()
        plt.title("Before correction, large mesh model")
        plt.imshow(warped_image_large)

        for i in range(len(model_points_large)):
            plt.plot([model_points_large[i][0], model_points_large[(i + 1) % len(model_points_large)][0]],
                     [model_points_large[i][1], model_points_large[(i + 1) % len(model_points_large)][1]], 'b')

        plt.show()

        # From experiments, p3p seems like the best
        algorithms = {
            "iterative": cv2.SOLVEPNP_ITERATIVE, #Working
        }

        for v, k in enumerate(algorithms):

            pnp_method = algorithms[k]
            print("Solving with {}\n".format(k))
            image = np.copy(clean_img)

            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            dist_coef = np.zeros(4)

            # Adjust the solution to match the endpoints of the image to the endpoints of the model
            temp_image_points = np.float64([[x] for x in image_points])
            temp_model_points = np.float64([[x[0], 0, h_warped_large - x[1]] for x in model_points_large])

            number_of_corrections = 100
            allignment_error = np.inf #In pixel distance
            allignment_error_threshold = 3

            # If the method is upnp, delete focal length from K to check if its possible
            # to get a good result
            # if k == "upnp":
            #     K[:2,:2] = 0

            correction_iteration = 0

            for _ in range(number_of_corrections):

                # Solve the pnp problem, and determine the image location of
                # endpoints of the model
                _ret, rvec, tvec = cv2.solvePnP(
                    temp_model_points, temp_image_points,
                    K, dist_coef,
                    flags=pnp_method)

                # for _ in range(10):
                #     _ret, rvec, tvec = cv2.solvePnP(
                #         temp_model_points, temp_image_points,
                #         K, dist_coef,
                #         rvec, tvec,
                #         useExtrinsicGuess = True,
                #         flags=pnp_method)

                # Display the result of the solution and return the points of the model on the image
                model_on_image, result_fig = CameraCalibration.displayPlacement(image,
                                                                    temp_model_points,
                                                                    rvec, tvec,
                                                                    K, dist_coef,
                                                                    horizon_method + "_" + k +
                                                                    " - err:" + "{0:.2f}".format(allignment_error))

                if correction_iteration == 0:
                    result_fig.savefig("placement_axes_" + horizon_method + "_initial.png")

                adjustment_homography, status = cv2.findHomography(model_on_image, temp_image_points)


                if all(status):

                    # Apply the homography on horizon and zenith points to adjust them (potentially)
                    # to appropriate focal length

                    wrapped_model = np.array(list(map(lambda x: [x[0], h_warped_large - x[2]], temp_model_points)))
                    wrapped_model = transform.matrix_transform(wrapped_model, adjustment_homography)
                    wrapped_model = np.array(list(map(lambda x: [x[0], 0, h_warped_large - x[1]], wrapped_model)))

                    model_update_coefficient = 0.1
                    temp_model_points = temp_model_points * (1 - model_update_coefficient) +\
                                        wrapped_model * model_update_coefficient

                    allignment_error = sum(np.linalg.norm(model_on_image - image_points, axis=1))

                    correction_iteration += 1

                    # If the distance between points is smaller than a threshold, stop
                    if allignment_error < allignment_error_threshold:

                        # Display and save for one last time
                        _, result_fig = CameraCalibration.displayPlacement( image,
                                                            temp_model_points,
                                                            rvec, tvec,
                                                            K, dist_coef,
                                                            horizon_method + "-" + k +
                                                            " - err:" + str(allignment_error))

                        result_fig.savefig("placement_axes_" + horizon_method + "_final.png")
                        break
                else:
                    break

            # region warped_image_correction

            # The warped image is updated to match the adjustments of the model

            # Recalculate the adjustment homography as the warped results were interpolated in the mesh
            temp_model_points = np.array(list(map(lambda x: [x[0], h_warped_large - x[2]], temp_model_points)))

            adjustment_homography, _ = cv2.findHomography(model_points_large, temp_model_points)

            # Find the required translation to keep navigable region inside the image

            tx_min = min(0, temp_model_points[:,0].min())
            ty_min = min(0, temp_model_points[:,1].min())

            tx_max = max(w_warped_large, temp_model_points[:,0].max()) - w_warped_large
            ty_max = max(h_warped_large, temp_model_points[:,1].max()) - h_warped_large

            if tx_min == 0:
                tx = tx_max
            else:
                tx = tx_min

            if ty_min == 0:
                ty = ty_max
            else:
                ty = ty_min

            translation_matrix = np.array([[1, 0, -tx],
                          [0, 1, -ty],
                          [0, 0, 1]])

            adjustment_homography = np.dot(translation_matrix, adjustment_homography)

            pnp_warped_img = transform.warp(warped_image_large, np.linalg.inv(adjustment_homography),
                                        output_shape = (h_warped_large, w_warped_large),
                                        preserve_range=False)

            plt.figure()
            plt.title("Updated warped image for {}".format(k))
            plt.imshow(pnp_warped_img)
            plt.imsave("warped_result_final_{}.png".format(k), pnp_warped_img)

            # Re-adjust the model points according to translated and scaled warped image
            temp_model_points = transform.matrix_transform(temp_model_points, translation_matrix)

            for i in range(len(temp_model_points)):
                plt.plot([temp_model_points[i][0], temp_model_points[(i + 1) % len(temp_model_points)][0]],
                         [temp_model_points[i][1], temp_model_points[(i + 1) % len(temp_model_points)][1]], 'b')

            plt.show()

            # endregion

            #region camera_parameters

            # As the pnp gives us the object translation, we need to get the camera translation by transposing and
            # applying the inverse rotation to translation
            rvec, _ = cv2.Rodrigues(rvec)
            rvec = np.transpose(rvec)
            tvec = np.dot(-rvec, tvec)

            # Intrinsic Line
            CameraCalibration.camWriter.write("{} {} {} {} {} {}\n".format(w_org, h_org,
                                                                           K[0][2], K[1][2],
                                                                           K[0][0], K[1][1]))

            # Extrinsic Line
            tvec = [t[0] for t in tvec]

            # Adjustment for translation in model
            tvec[0] -= tx
            tvec[2] += ty

            CameraCalibration.camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n"
                                              .format(*(rvec[:, 0]), *(rvec[:, 1]),*(rvec[:, 2]),
                                                      *tvec,
                                                      w_warped, h_warped))

            # Write solution information
            CameraCalibration.camWriter.write("{},{},{},{}".format(k,
                                                                   correction_iteration,
                                                                   allignment_error,
                                                                   utils.focalToFOV(K[0,0], h_org)
                                                                   ))

            print(k)
            print("Rotation Rodrigues {}".format(rvec))

            # For debugging purposes
            rvec, _ = cv2.Rodrigues(rvec)

            rvec[0] = np.rad2deg(rvec[0])
            rvec[1] = np.rad2deg(rvec[1])
            rvec[2] = np.rad2deg(rvec[2])

            print("Rotation Euler Angles {}".format(rvec))
            print("Translation {}".format(tvec))

            #endregion
