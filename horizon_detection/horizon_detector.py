import sys
from os import path

import cv2
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

__all__ = ["HorizonDetectorLib"]

class HorizonDetectorLib(object):
    # Parameter
    # Threshold to be used for computing inliers in degrees.Angle between
    # edgelet direction and vanishing point is thresholded.
    RANSAC_ITERATION_COUNT = 10000

    BILATERAL_D = 9
    BILATERAL_SIGMA_C = 60
    BILATERAL_SIGMA_S = 60

    HOUGH_LINE_LENGTH = 30
    HOUGH_LINE_GAP = 10

    # These should be tuned according to the inliers for each
    INLIER_THRESHOLD_HORIZON_FIRST = 40
    INLIER_THRESHOLD_HORIZON_SECOND = 20

    # Keep this high in order to focus on vertical lines, as they cannot contribute to horizon but
    # other horizontal lines can influence nadir
    INLIER_THRESHOLD_NADIR = 5

    # Extracts the edges and hough lines from the image
    # Taken from IMAGE RECTIFICATION
    @staticmethod
    def extract_image_lines(org_img, output_folder="", displayLines=True, skimage_solution=True):

        image = np.copy(org_img)

        # Bilateral filtering which keeps the edges sharp, but textures blurry
        # seems to decrease the noisy edges that cause too many detection results
        # Read bilateral filters: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
        # Everyone is affected by similar and close pixels. If neighbour is not similar, then its effect is small
        # Makes things more "comical"
        image = cv2.bilateralFilter(image, HorizonDetectorLib.BILATERAL_D,
                                    HorizonDetectorLib.BILATERAL_SIGMA_C,
                                    HorizonDetectorLib.BILATERAL_SIGMA_S)

        # image = ndimage.gaussian_filter(image, 4)
        # The image needs to be converted to grayscale first
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if skimage_solution:

            from skimage.transform import (probabilistic_hough_line)
            from skimage.feature import canny

            grayscale_img = grayscale_img / 255.0
            grayscale_img = canny(grayscale_img, sigma=3)

            line_segments = probabilistic_hough_line(grayscale_img,
                                                     line_length=HorizonDetectorLib.HOUGH_LINE_LENGTH,
                                                     line_gap=HorizonDetectorLib.HOUGH_LINE_GAP)

            # Taken from skimage's own tutorial
            if displayLines:
                # region show_edges
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
                ax = axes.ravel()

                ax[0].imshow(image, cmap=cm.gray)
                ax[0].set_title('Input image')

                ax[1].imshow(grayscale_img, cmap=cm.gray)
                ax[1].set_title('Canny edges')

                ax[2].imshow(grayscale_img * 0)
                for line in line_segments:
                    p0, p1 = line
                    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
                ax[2].set_xlim((0, image.shape[1]))
                ax[2].set_ylim((image.shape[0], 0))
                ax[2].set_title('Probabilistic Hough')

                for a in ax:
                    a.set_axis_off()

                plt.tight_layout()
                plt.show()

                # endregion show_edges
        else:

            # Edge detection (canny for now)
            grayscale_img = cv2.Canny(grayscale_img, threshold1=150,
                                      threshold2=200, apertureSize=3)
            # grayscale_img = cv2.Sobel(grayscale_img,cv2.CV_8UC1,1,1)
            # grayscale_img = cv2.Laplacian(grayscale_img, cv2.CV_8UC1)

            plt.imshow(grayscale_img, cmap='gray')
            plt.show()

            # # Hough lines (OpenCV version)
            line_segments = cv2.HoughLinesP(grayscale_img, 1, np.pi / 180, threshold=100,
                                            minLineLength=HorizonDetectorLib.HOUGH_LINE_LENGTH,
                                            maxLineGap=HorizonDetectorLib.HOUGH_LINE_GAP)

        with open(path.join(output_folder, "line_parameters.txt"), "w") as lineWriter:
            lineWriter.write('''Bilateral D {}, SIGMA_C {}, SIGMA_S {}, HOUGH_LENGTH {}, 
                             HOUGH_GAP {}, HOR THRESHOLDS {},{},{}'''.format(
                HorizonDetectorLib.BILATERAL_D,
                HorizonDetectorLib.BILATERAL_SIGMA_C,
                HorizonDetectorLib.BILATERAL_SIGMA_S,
                HorizonDetectorLib.HOUGH_LINE_LENGTH,
                HorizonDetectorLib.HOUGH_LINE_GAP,
                HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_FIRST,
                HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_SECOND,
                HorizonDetectorLib.INLIER_THRESHOLD_NADIR))

        return HorizonDetectorLib.lineProperties(line_segments, image)

    # By parsing the detection data, it extracts the paths of each pedestrian for every x frames
    # The head-feet positions depend on the selected method; using bounding boxes or pedestrian postures
    # Note: Pedestrian postures are more prone to noise
    # In addition, Using the pedestrian postures, determine the orthogonal vanishing point using RANSAC for minimizing
    # distance between candidate line pairs
    @staticmethod
    def parse_pedestrian_detection(image, detection_data, frames_per_check=30,
                                   tracker_id=None):
        latest_loc = {}
        paths = []
        postures = []
        trajectories = []

        num_of_frames = 0
        latest_frame = 0

        with open(detection_data) as detections:
            for i, line in enumerate(detections):

                # Checks pedestrian data every x frames
                # Read the line
                agent = line.split(',')
                frameID = int(agent[0])

                # As multiple lines can belong to the same frame, only count change in frames
                # if frameID > latest_frame or tracker_id is not None:
                #     latest_frame = frameID
                #     if tracker_id is None :
                #         num_of_frames += 1

                if frameID > latest_frame:
                    latest_frame = frameID
                    num_of_frames += 1

                if (agent[1] not in latest_loc.keys() or num_of_frames - (latest_loc[agent[1]])[2] >= frames_per_check) \
                        and (tracker_id is None or (int(agent[1]) in tracker_id)):

                    # Different methods for extracting head and feet

                    # Add the agent id to the dictionary latest_loc
                    headPos = list(map(float, agent[-2].split('/')))
                    feetPos = list(map(float, agent[-1].split('/')))

                    try:
                        prev_pos = latest_loc[agent[1]]
                        # print("Last seen {} at frame number: {}".format(agent[1], prev_pos[2]))
                        head_path = [prev_pos[0], headPos]
                        feet_path = [prev_pos[1], feetPos]

                        # Detect and remove outliers.
                        # Outliers are inconsistent detection box sizes. For example, a detection box that
                        # shrinks while getting closer to the camera is considered as an outlier
                        currentHeight = np.linalg.norm(np.array(feetPos) - np.array(headPos))
                        prev_height = np.linalg.norm(np.array(prev_pos[1]) - np.array(prev_pos[0]))
                        size_increased = currentHeight > prev_height
                        higher_position = (headPos[1] + feetPos[1]) / 2 < (prev_pos[0][1] + prev_pos[1][1]) / 2

                        if size_increased ^ higher_position:
                            # The paths are held as pairs
                            paths.append(feet_path)
                            trajectories.append(feet_path)
                            paths.append(head_path)

                            postures.append([headPos, feetPos])
                            latest_loc[agent[1]] = [headPos, feetPos, num_of_frames]

                    except:
                        latest_loc[agent[1]] = [headPos, feetPos, num_of_frames]

        detections.close()

        return HorizonDetectorLib.lineProperties(paths, image), \
               HorizonDetectorLib.lineProperties(postures, image), \
               HorizonDetectorLib.lineProperties(trajectories, image)

    # Extracts the line properties from given line segments
    # Returns centers, directions and strengths
    # Uses image for displaying
    @staticmethod
    def lineProperties(lines, image):
        line_directions = []
        line_centers = []
        line_strengths = []

        for line in lines:
            line = list(np.array(line).flatten())  # Compatibility

            if len(line) > 4:  # Homogenous coordinates are given
                line = line[:2] + line[3:5]

            x0 = int(line[0])
            y0 = int(line[1])
            x1 = int(line[2])
            y1 = int(line[3])

            # Filter points
            if (y1 - y0 and x1 - x0) != 0:
                line_centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
                line_directions.append((x1 - x0, y1 - y0))
                line_strengths.append(np.linalg.norm([x1 - x0, y1 - y0]))

            # Draw the detected lines on the original image
            # cv2.line(image, (x0, y0), (x1, y1), (0, 0, 255), 1)

        # Taken from https://github.com/chsasank/Image-Rectification
        # The direction vectors are normalized for easier calculation afterwards
        if len(line_directions) > 0:
            line_directions = np.array(line_directions) / \
                              (np.linalg.norm(line_directions, axis=1)[:, np.newaxis] + sys.float_info.epsilon)

        # cv2.imshow("Extracted Lines", image)
        # cv2.waitKey(0)

        return tuple(map(np.array, [line_centers, line_directions, line_strengths]))

    # Takes horizon vertices and nadir vp in homogenous coordinates
    # Finds the focal length using the orthocenter of the triangle defined
    # by VP's.
    @staticmethod
    def find_focal_length(horizon, nadir):

        left_vp = np.array(horizon[0])
        right_vp = np.array(horizon[1])

        horizon_homogenous = np.cross(left_vp, right_vp)

        v1_v2_len = np.linalg.norm(left_vp - right_vp)
        v1_zen_len = np.linalg.norm(left_vp - nadir)
        v2_zen_len = np.linalg.norm(right_vp - nadir)

        v1_angle = np.tan(np.arccos(np.dot(right_vp - left_vp, nadir - left_vp) / (v1_zen_len * v1_v2_len)))
        v2_angle = np.tan(np.arccos(np.dot(left_vp - right_vp, nadir - right_vp) / (v2_zen_len * v1_v2_len)))
        nadir_angle = np.tan(np.arccos(np.dot(right_vp - nadir, left_vp - nadir) / (v1_zen_len * v2_zen_len)))

        image_center = [0, 0, 1]

        image_center[0] = (left_vp[0] * v1_angle +
                           right_vp[0] * v2_angle +
                           nadir[0] * nadir_angle) / (v1_angle + v2_angle + nadir_angle)

        image_center[1] = (left_vp[1] * v1_angle +
                           right_vp[1] * v2_angle +
                           nadir[1] * nadir_angle) / (v1_angle + v2_angle + nadir_angle)

        center_hor_dist = np.abs(np.dot(np.array(image_center), horizon_homogenous)) / np.linalg.norm(
            horizon_homogenous[:2])

        center_v1 = np.linalg.norm(image_center - left_vp)
        center_v2 = np.linalg.norm(image_center - right_vp)

        focal_length = np.sqrt(
            np.abs((np.sqrt(center_v1 ** 2 - center_hor_dist ** 2)) * (np.sqrt(center_v2 ** 2 - center_hor_dist ** 2))
                   - (center_hor_dist ** 2)))

        return focal_length

    @staticmethod
    def show_inliers(path_lines, inliers, point, plot_axis, color):

        centers, directions, strengths = path_lines

        centers = centers[inliers]
        directions = directions[inliers]
        strengths = strengths[inliers]

        for i in range(centers.shape[0]):
            plot_axis.plot([centers[i][0] - (directions[i][0] * strengths[i]),
                            centers[i][0] + (directions[i][0] * strengths[i])],
                           [centers[i][1] - (directions[i][1] * strengths[i]),
                            centers[i][1] + (directions[i][1] * strengths[i])], color + "-", lw=0.75)

        # for i in range(centers.shape[0]):
        #     xax = [centers[i, 0], point[0]]
        #     yax = [centers[i, 1], point[1]]
        #     plot_axis.plot(xax, yax, 'b:')

    # Determines the vanishing points on horizon using information coming from pedestrian paths
    # OR uses the trajectory information and/or edges from the image to detect vanishing points
    # which will determine the horizon
    # TODO: Parameters for thresholds
    @staticmethod
    def determineVP(path_lines, image_center, plot_axis,
                    postures=None, draw_features=False):

        # Using RANSAC method on trajectories
        model = HorizonDetectorLib.ransac_vanishing_point(path_lines,
                                                          HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_FIRST)
        # model = HorizonDetectorLib.reestimate_model(model, path_lines, 5)
        vp1 = model / model[2]

        # Before determining the second VP, remove inliers as they already contributed to first VP
        path_lines_reduced, inliers = HorizonDetectorLib.remove_inliers(vp1, path_lines,
                                                                        HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_FIRST)

        # Display the inliers
        if draw_features:
            HorizonDetectorLib.show_inliers(path_lines, inliers, vp1, plot_axis, 'r')

        path_lines = path_lines_reduced

        # Find second vanishing point
        model2 = HorizonDetectorLib.ransac_vanishing_point(path_lines,
                                                           HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_SECOND)
        # model2 = HorizonDetectorLib.reestimate_model(model2, path_lines, 5)
        vp2 = model2 / model2[2]

        # Test if we can find the nadir vanishing point
        # Before determining the second VP, remove inliers as they already contributed to first VP
        path_lines_reduced, inliers = HorizonDetectorLib.remove_inliers(vp2, path_lines,
                                                                        HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_SECOND)

        # Display the inliers
        if draw_features:
            HorizonDetectorLib.show_inliers(path_lines, inliers, vp2, plot_axis, 'g')

        path_lines = path_lines_reduced

        # Find nadir

        # If postures are provided, use them
        # Else continue using the hough lines
        if postures is not None:
            path_lines = postures

        model3 = HorizonDetectorLib.ransac_zenith_vp(path_lines, [vp1, vp2], image_center,
                                                     HorizonDetectorLib.INLIER_THRESHOLD_NADIR)
        # model3 = HorizonDetectorLib.reestimate_model(model3, path_lines, 5)
        vp3 = model3 / model3[2]

        _, inliers = HorizonDetectorLib.remove_inliers(vp3, path_lines,
                                                       HorizonDetectorLib.INLIER_THRESHOLD_NADIR)

        if draw_features:
            HorizonDetectorLib.show_inliers(path_lines, inliers, vp3, plot_axis, 'b')

        # The vanishing point with highest y value is taken as the nadir 8as we are looking at the world birdview)
        vanishers = [vp1, vp2, vp3]
        vanishers.sort(key=lambda v: v[1])
        horizon_points = vanishers[:2]
        horizon_points.sort(key=lambda v: v[0])

        vp_left, vp_right, vp_zenith = horizon_points[0], horizon_points[1], vanishers[2]

        return [vp_left, vp_right, vp_zenith]

    # Using posture data, finds nadir vanishing point only
    @staticmethod
    def ransac_zenith_vp(edgelets, horizon, image_center, threshold_inlier):

        # If no nadir vp is given, calculate it from postures and image lines
        locations, directions, strengths = edgelets
        lines = HorizonDetectorLib.edgelet_lines(edgelets)

        num_pts = strengths.size

        first_index_space, second_index_space = HorizonDetectorLib.generate_search_bins(edgelets)

        best_model = None
        best_votes = np.zeros(num_pts)

        # Normalize the horizon
        horizon_homogenous = np.cross(horizon[0], horizon[1])

        # These two are actually the same
        horizon_homogenous = horizon_homogenous / horizon_homogenous[2]

        if best_model is None:
            for ransac_iter in range(HorizonDetectorLib.RANSAC_ITERATION_COUNT):
                ind1 = np.random.choice(first_index_space)
                ind2 = np.random.choice(second_index_space)

                while ind2 == ind1 and len(first_index_space) != 1:  # Protection against low line count
                    ind2 = np.random.choice(second_index_space)

                l1 = lines[ind1]
                l2 = lines[ind2]

                current_model = np.cross(l1, l2)  # Potential vanishing point
                current_model = current_model / current_model[2]

                if np.sum(current_model ** 2) < 1 or current_model[2] == 0 or current_model[1] < 0:
                    # reject degenerate candidates, which lie on the wrong side of the horizon
                    continue

                current_votes = HorizonDetectorLib.compute_votes(
                    edgelets, current_model, threshold_inlier)

                if current_votes.sum() > best_votes.sum():
                    best_model = current_model
                    best_votes = current_votes
                    # logging.info("Current best model has {} votes at iteration {}".format(
                    #     current_votes.sum(), ransac_iter))

        return best_model

    @staticmethod
    def generate_search_bins(edgelets):

        _, directions, strengths = edgelets

        #
        # # Number of neighbour bins to consider (symmetric)
        # peak_margin = 3
        #
        # directions = directions[np.argsort(-strengths)[:len(directions) //
        #                                      (100 // top_strength_percentage)]]
        # # The search space is initiated as the peak of the current histogram
        # angles = np.array(list(map(lambda line: np.rad2deg(np.arctan(line[1] / line[0])),
        #                   directions)))
        #
        # sorted_angles , histogram, _ = plt.hist(angles, bins=180 // bin_size, range=(-90, 90))
        # max_angle = -90 + np.argmax(sorted_angles) * bin_size + (bin_size / 2)
        #
        # # TODO: Visualize the histogram Test area
        # fig, ax = plt.subplots()
        # ax.plot(histogram)
        # ax.axis([-90, 90, 0, sorted_angles[np.argmax(sorted_angles)] + 10])
        # plt.show(block=False)
        #
        # selected_angles = np.argsort(angles)
        #
        # right_neighbor = max_angle + (bin_size * (0.5 + peak_margin))
        # left_neighbor = max_angle - (bin_size * (0.5 + peak_margin))
        #
        # # Include neighbors on the exact opposite side of the spectrum. Applies to angles very close to -90/90
        # right_neighbor = ((right_neighbor + 90) % 180) - 90
        # left_neighbor = ((left_neighbor + 90) % 180) - 90
        #
        # if right_neighbor < left_neighbor:
        #     first_index_space = np.sort(np.array([index for index in selected_angles if
        #                 (angles[index] <= right_neighbor or angles[index] >= left_neighbor)]))
        # else:
        #     first_index_space = np.sort(np.array([index for index in selected_angles if
        #                                           (angles[index] <= right_neighbor and angles[index] >= left_neighbor)]))
        #
        #
        # # Second space will be sorted according to the strength of the lines. The top %50 of all lines are able to
        # # contribute + lines in first_first index space
        # second_index_space = np.argsort(-strengths)
        # second_index_space = np.array([i for i in second_index_space if i not in first_index_space])
        # second_index_space = np.sort(np.append(np.array(second_index_space[:len(second_index_space) //
        #                                                                     (100 // top_strength_percentage)]),
        #                                                                     first_index_space))

        sorted_strengths = np.argsort(-strengths)
        num_pts = len(sorted_strengths)
        first_index_space = sorted_strengths if num_pts < 20 else sorted_strengths[:num_pts // 5]  # Top 20 percentile
        second_index_space = sorted_strengths if num_pts < 20 else sorted_strengths[:num_pts // 2]  # Top 50 percentile

        return first_index_space, second_index_space

    @staticmethod
    def ransac_vanishing_point(edgelets, threshold_inlier):
        """Estimate vanishing point using Ransac.

        Parameters
        ----------
        edgelets: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_edgelets`.

        Returns
        -------
        best_model: ndarry of shape (3,)
            Best model for vanishing point estimated.

        Reference
        ---------
        Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
        "Auto-rectification of user photos." 2014 IEEE International Conference on
        Image Processing (ICIP). IEEE, 2014.
        """

        locations, directions, strengths = edgelets
        lines = HorizonDetectorLib.edgelet_lines(edgelets)

        num_pts = strengths.size

        first_index_space, second_index_space = HorizonDetectorLib.generate_search_bins(edgelets)

        best_model = None
        best_votes = np.zeros(num_pts)

        for ransac_iter in range(HorizonDetectorLib.RANSAC_ITERATION_COUNT):

            ind1 = np.random.choice(first_index_space)
            ind2 = np.random.choice(second_index_space)

            while ind2 == ind1 and len(first_index_space) != 1:  # Protection against low line count
                ind2 = np.random.choice(second_index_space)

            l1 = lines[ind1]
            l2 = lines[ind2]

            current_model = np.cross(l1, l2)  # Potential vanishing point

            if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
                # reject degenerate candidates
                continue

            current_votes = HorizonDetectorLib.compute_votes(
                edgelets, current_model, threshold_inlier)

            if current_votes.sum() > best_votes.sum():
                best_model = current_model
                best_votes = current_votes
                # logging.info("Current best model has {} votes at iteration {}".format(
                #     current_votes.sum(), ransac_iter))

        return best_model

    @staticmethod
    def compute_votes(edgelets, model, threshold_inlier):
        """Compute votes for each of the edgelet against a given vanishing point.

        Votes for edgelets which lie inside threshold are same as their strengths,
        otherwise zero.

        Parameters
        ----------
        edgelets: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_edgelets`.
        model: ndarray of shape (3,)
            Vanishing point model in homogenous cordinate system

        Returns
        -------
        votes: ndarry of shape (n_edgelets,)
            Votes towards vanishing point model for each of the edgelet.

        """
        vp = model[:2] / model[2]

        locations, directions, strengths = edgelets

        est_directions = locations - vp
        dot_prod = np.sum(est_directions * directions, axis=1)
        abs_prod = np.linalg.norm(directions, axis=1) * \
                   np.linalg.norm(est_directions, axis=1)
        abs_prod[abs_prod == 0] = sys.float_info.epsilon

        cosine_theta = dot_prod / (abs_prod + sys.float_info.epsilon)
        theta = np.arccos(np.abs(cosine_theta))

        theta_thresh = threshold_inlier * np.pi / 180

        return (theta < theta_thresh) * strengths

    @staticmethod
    def remove_inliers(model, edgelets, threshold_inlier):
        """Remove all inlier edglets of a given model.

        Parameters
        ----------
        model: ndarry of shape (3,)
            Vanishing point model in homogenous coordinates which is to be
            reestimated.
        edgelets: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_edgelets`.

        Returns
        -------
        edgelets_new: tuple of ndarrays
            All Edgelets except those which are inliers to model.
        """
        inliers = HorizonDetectorLib.compute_votes(edgelets, model, threshold_inlier) > 0
        locations, directions, strengths = edgelets
        locations = locations[~inliers]
        directions = directions[~inliers]
        strengths = strengths[~inliers]
        edgelets = (locations, directions, strengths)
        return edgelets, inliers

    def reestimate_model(model, edgelets, threshold_reestimate=5):
        """Reestimate vanishing point using inliers and least squares.

        All the edgelets which are within a threshold are used to reestimate model

        Parameters
        ----------
        model: ndarry of shape (3,)
            Vanishing point model in homogenous coordinates which is to be
            reestimated.
        edgelets: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_edgelets`.
            All edgelets from which inliers will be computed.
        threshold_inlier: float
            threshold to be used for finding inlier edgelets.

        Returns
        -------
        restimated_model: ndarry of shape (3,)
            Reestimated model for vanishing point in homogenous coordinates.
        """
        locations, directions, strengths = edgelets

        inliers = HorizonDetectorLib.compute_votes(edgelets, model, threshold_reestimate) > 0
        locations = locations[inliers]
        directions = directions[inliers]
        strengths = strengths[inliers]

        lines = HorizonDetectorLib.edgelet_lines((locations, directions, strengths))

        a = lines[:, :2]
        b = -lines[:, 2]
        est_model = np.linalg.lstsq(a, b)[0]
        return np.concatenate((est_model, [1.]))
