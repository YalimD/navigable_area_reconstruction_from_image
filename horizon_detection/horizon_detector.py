import sys
from os import path

import cv2
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from horizon_detection import image_rectification

__all__ = ["HorizonDetectorLib"]


class HorizonDetectorLib:
    # Parameter
    # Threshold to be used for computing inliers in degrees.Angle between
    # edgelet direction and vanishing point is thresholded.
    RANSAC_ITERATION_COUNT = 10000

    BILATERAL_D = 9
    BILATERAL_SIGMA_C = 60
    BILATERAL_SIGMA_S = 60

    # A good set of lines (both length and quantity) are necessary to accurately define the scene
    HOUGH_LINE_LENGTH = 50
    HOUGH_LINE_GAP = 5

    # These should be tuned according to the inliers for each
    INLIER_THRESHOLD_HORIZON_FIRST = 10
    INLIER_THRESHOLD_HORIZON_SECOND = 10

    # Keep this high in order to focus on vertical lines, as they cannot contribute to horizon but
    # other horizontal lines can influence nadir
    INLIER_THRESHOLD_NADIR = 10

    # For stability, we remove the closest set from the found inliers
    INLIER_FINAL_THRESHOLD = 5

    # Extracts the edges and hough lines from the image
    # Taken from IMAGE RECTIFICATION
    @staticmethod
    def extract_image_lines(org_img, output_folder="", display_lines=True, skimage_solution=True):

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
            if display_lines:
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

            # Edge detection
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

        return HorizonDetectorLib.line_properties(line_segments)

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

        return HorizonDetectorLib.line_properties(paths), \
               HorizonDetectorLib.line_properties(postures), \
               HorizonDetectorLib.line_properties(trajectories), \
               trajectories

    # Extracts the line properties from given line segments
    # Returns centers, directions and strengths
    # Uses image for displaying
    @staticmethod
    def line_properties(lines):
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

        # Taken from https://github.com/chsasank/Image-Rectification
        # The direction vectors are normalized for easier calculation afterwards
        if len(line_directions) > 0:
            line_directions = np.array(line_directions) / \
                              (np.linalg.norm(line_directions, axis=1)[:, np.newaxis] + sys.float_info.epsilon)

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

        # Show dashed lines towards vanishing point
        for i in range(centers.shape[0]):
            xax = [centers[i, 0], point[0]]
            yax = [centers[i, 1], point[1]]
            plot_axis.plot(xax, yax, 'b:')

    # Determines the vanishing points on horizon using information coming from pedestrian paths
    # OR uses the trajectory information and/or edges from the image to detect vanishing points
    # which will determine the horizon
    @staticmethod
    def determineVP(path_lines, plot_axis,
                    postures=None, constraints = None, draw_features=False):

        #Used constraint vp's that are below trajectories

        # Using RANSAC method on trajectories
        model = image_rectification.ransac_vanishing_point(path_lines,
                                                           constraints,
                                                           HorizonDetectorLib.RANSAC_ITERATION_COUNT,
                                                           HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_FIRST)
        vp1 = model / model[2]

        # Before determining the second VP, remove inliers as they already contributed to first VP
        path_lines_reduced, inliers = image_rectification.remove_inliers(vp1, path_lines,
                                                                         HorizonDetectorLib.INLIER_FINAL_THRESHOLD)

        # Display the inliers
        if draw_features:
            HorizonDetectorLib.show_inliers(path_lines, inliers, vp1, plot_axis, 'r')

        path_lines = path_lines_reduced

        # Find second vanishing point
        model2 = image_rectification.ransac_vanishing_point(path_lines,
                                                            constraints,
                                                            HorizonDetectorLib.RANSAC_ITERATION_COUNT,
                                                            HorizonDetectorLib.INLIER_THRESHOLD_HORIZON_SECOND)
        vp2 = model2 / model2[2]

        # Test if we can find the nadir vanishing point
        # Before determining the second VP, remove inliers as they already contributed to first VP
        path_lines_reduced, inliers = image_rectification.remove_inliers(vp2, path_lines,
                                                                         HorizonDetectorLib.INLIER_FINAL_THRESHOLD)

        # Display the inliers
        if draw_features:
            HorizonDetectorLib.show_inliers(path_lines, inliers, vp2, plot_axis, 'g')

        path_lines = path_lines_reduced

        # Find nadir

        # If postures are provided, use them
        # Else continue using the hough lines
        if postures is not None:
            path_lines = postures

        model3 = HorizonDetectorLib.ransac_nadir_vp(path_lines, constraints, HorizonDetectorLib.INLIER_THRESHOLD_NADIR)
        vp3 = model3 / model3[2]

        _, inliers = image_rectification.remove_inliers(vp3, path_lines,
                                                        HorizonDetectorLib.INLIER_FINAL_THRESHOLD)

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
    def ransac_nadir_vp(edgelets, constraints, threshold_inlier):

        # If no nadir vp is given, calculate it from postures and image lines
        locations, directions, strengths = edgelets
        lines = image_rectification.edgelet_lines(edgelets)

        num_pts = strengths.size
        arg_sort = np.argsort(-strengths)

        first_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 5]
        second_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 2]

        best_model = None
        best_votes = np.zeros(num_pts)

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

                # reject degenerate candidates, which lie on the wrong side of the horizon
                if np.sum(current_model ** 2) < 1 or current_model[2] == 0 or current_model[1] < 0 \
                        or not image_rectification.check_constraint(current_model, constraints, True):
                    continue


                current_votes = image_rectification.compute_votes(
                    edgelets, current_model, threshold_inlier)

                if current_votes.sum() > best_votes.sum():
                    best_model = current_model
                    best_votes = current_votes

        return best_model
