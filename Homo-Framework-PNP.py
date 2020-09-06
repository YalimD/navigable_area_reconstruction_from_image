import argparse
import math
import warnings
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as transform

import utils
from camera_calibration import cameraCalibration
from horizon_detection import horizon_detector

'''
Homography framework that applies perspective correction to given surveillance image
by converting it to bird-view perspective as close as possible.

Works on a single image, camera is assumed to be not calibrated and focal_unity length is not known.

Algorithm steps
- Using RANSAC, find the best point that represents the vanishing points
- Identify the vanishing line as the combination of left and right VP (manhattan world assumption) 
- Apply the homography logic
- Using the vanishing points, approximate the focal_unity length, using triangle properties
Reference: https://www.coursera.org/learn/robotics-perception/lecture/jnaLs/how-to-compute-intrinsics-from-vanishing-points

author Yalım Doğan

Some ideas and code segments are from: https://github.com/chsasank/Image-Rectification (uses Scikit rather than OpenCV)
which is an implementation of Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe. "Auto-rectification of user photos." 
2014 IEEE International Conference on Image Processing (ICIP). IEEE, 2014.
'''


# For stratified rectification, affine matrix requires 2 parameters that are related to
# the circular points that lie on the absolute conic. In order to obtain them, lines with
# known ratios are necessary. For this, we will process the given trajectory lines and pick
# paths that have constant velocity according to re-projection.

# We also assume that no consecutive trajectory is parallel and we take only positive B values
# as negative values cause mirror effect on ground planes

# The logic is adopted from "Ground Plane Rectification by Tracking Moving Objects"
def extract_circular_points(trajectory_lines, P, method, output_path=""):
    from matplotlib.patches import Circle

    circles = []
    intersections = set([])

    fig, ax = plt.subplots()

    centers, directions, strengths = trajectory_lines

    # For every 2 line, calculate centre and radius of the circle using line endpoints
    for i in range(len(trajectory_lines[0]) // 2):
        line_1 = [[centers[i][0] - (directions[i][0] * strengths[i]),
                   centers[i][0] + (directions[i][0] * strengths[i]), 1],
                  [centers[i][1] - (directions[i][1] * strengths[i]),
                   centers[i][1] + (directions[i][1] * strengths[i]), 1]]
        line_2 = [[centers[i + 1][0] - (directions[i + 1][0] * strengths[i + 1]),
                   centers[i + 1][0] + (directions[i + 1][0] * strengths[i + 1]), 1],
                  [centers[i + 1][1] - (directions[i + 1][1] * strengths[i + 1]),
                   centers[i + 1][1] + (directions[i + 1][1] * strengths[i + 1]), 1]]

        # Apply projection to lines
        line_1 = np.array([np.dot(P, np.array(t).T) for t in line_1])
        line_1[0] = line_1[0] / line_1[0][2]
        line_1[1] = line_1[1] / line_1[1][2]

        line_2 = np.array([np.dot(P, np.array(t).T) for t in line_2])
        line_2[0] = line_2[0] / line_2[0][2]
        line_2[1] = line_2[1] / line_2[1][2]

        # Assume length ratio of the trajectories are 1 in the world plane
        s = 1

        delta_x1 = line_1[1][0] - line_1[0][0]
        delta_y1 = line_1[1][1] - line_1[0][1]
        delta_x2 = line_2[1][0] - line_2[0][0]
        delta_y2 = line_2[1][1] - line_2[0][1]

        c_alpha, c_beta = ((delta_x1 * delta_y1 - pow(s, 2) * delta_x2 * delta_y2)
                           / (pow(delta_y1, 2) - pow((s * delta_y2), 2)), 0)
        radius = np.abs(s * (delta_x2 * delta_y1 - delta_x1 * delta_y2) / (pow(delta_y1, 2) - pow((s * delta_y2), 2)))

        # For every circle found, find its intersection with every other circle
        # Formula: http://mathworld.wolfram.com/Circle-CircleIntersection.html
        for circle in circles:  # The "circle" here shadows the one above
            d = circle.center[0] - c_alpha
            r = circle.radius
            intersection_x = c_alpha + (pow(d, 2) - pow(r, 2) + pow(radius, 2)) \
                             / (2 * d)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                intersection_y = np.sqrt((r - radius - d) * (- d - r + radius) * (-d + r + radius) * (d + r + radius)) \
                                 / (2 * d)

            if not math.isnan(intersection_y):  # If there is intersection
                intersections.add((intersection_x, abs(intersection_y)))

                # Draw the intersection points
                ax.plot([intersection_x], [intersection_y], 'ro')
                ax.plot([intersection_x], [-intersection_y], 'ro')

        ax.tick_params(axis='both', which='major', labelsize=14)

        # Add the circle to the list
        circle = Circle((c_alpha, c_beta), radius, color='b', fill=False)
        circles.append(circle)

        ax = plt.gca()
        ax.add_patch(circle)

    mean_intersection = intersections
    # Find and plot the mean of the intersection points
    if len(intersections) > 1:
        mean_intersection = np.mean(list(intersections), axis=0)

    if len(intersections) > 0:
        ax.plot(mean_intersection[0], mean_intersection[1], 'bo')

    ax.axis('scaled')
    ax.set_title("Trajectory Circular Points For {}".format(method))

    plt.xlabel("alpha", fontsize=17)
    plt.ylabel("beta", fontsize=17)
    plt.show(block=False)
    fig.savefig(path.join(output_path, method + "_circular.png"))

    return intersections, mean_intersection


# Modified version of Google's paper
def compute_homography_and_warp(image,
                                vp1, vp2,
                                trajectories,
                                corners,
                                method="posture",
                                output_dir=""):
    height, width, _ = image.shape

    # Find Projective Transform
    vanishing_line = np.cross(vp1, vp2)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2]  # As h32 needs to be 1 in projection

    final_homography = H

    # If trajectories are not provided, we cannot rely on them for affine correction
    if trajectories is not None:
        # Determine a and b for the affine component of the homography
        intersections, mean_intersection = extract_circular_points(trajectories, H, method, output_dir)

        # for mean_intersection in intersections:
        A = np.eye(3, 3)

        if len(mean_intersection) > 0:
            a = mean_intersection[0]
            b = mean_intersection[1]

            A[0, 0] = 1 / b
            A[0, 1] = -a / b

        final_homography = np.dot(A, H)

    # region Translation and scaling operations

    # The image corners are transformed by the current matrix to determine the
    # endpoints of the resulting wrapped image. Each column is a corner in homogenous
    # coordinates

    image_corners = np.array([
        [0, 0, width, width],
        [0, height, 0, height],
        [1, 1, 1, 1]
    ])

    # Apply the current transformation
    cords = np.dot(final_homography, image_corners)

    # Normalize the points
    cords = cords[:2] / cords[2]

    # Now, the lines contain x and y coordinates of all points of the model
    # The smallest ones are translated to 0-0 borders by determining the min
    tx = min(0, cords[0].min())
    ty = min(0, cords[1].min())

    # Considering the applied transformation, determine the farthest points to the
    # upper left corner
    max_x = int(cords[0].max() - tx)
    max_y = int(cords[1].max() - ty)

    T = np.array([[1, 0, -tx],
                  [0, 1, -ty],
                  [0, 0, 1]])

    final_homography = np.dot(T, final_homography)

    S = np.array([[width / max_x, 0, 0],
                  [0, height / max_y, 0],
                  [0, 0, 1]])

    final_homography = np.dot(S, final_homography)

    # We end up with a image that has the same size but perspective corrected
    # We don't clamp the result as losing information is not an option

    # endregion

    warped_img = transform.warp(image,
                                np.linalg.inv(final_homography),
                                clip=False,
                                output_shape=(height, width))

    transformed_corners = transform.matrix_transform(corners, final_homography)

    return warped_img, transformed_corners, final_homography


# Checks the polygon convexity by finding the determinant for each 3 corners in cc order
def check_polygon_convexity(model_points):
    for corner_index in range(model_points.shape[0]):

        corner = model_points[corner_index]
        next_corner = model_points[(corner_index + 1) % model_points.shape[0]]

        det_matrix = np.array([
            [1, 1, 1],
            [corner[0], corner[1], 0],
            [next_corner[0], next_corner[1], 0],
        ])

        if np.linalg.det(det_matrix <= 0):
            return False

    return True


# region ground_truth_determination


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(data) < 8:
            data['points'].append([x, y, 1])
            cv2.circle(data['image'], (x, y), 3, (0, 0, 255), 5, 16)
            cv2.imshow(data['windowName'], data['image'])


def processGTlines(image_dir, data):
    cv2.destroyAllWindows()

    lineA = utils.normalized_cross(*data['points'][0:2])
    lineB = utils.normalized_cross(*data['points'][2:4])
    point1 = list(map(lambda x: int(x), utils.normalized_cross(lineA, lineB)))

    lineA = utils.normalized_cross(*data['points'][4:6])
    lineB = utils.normalized_cross(*data['points'][6:8])
    point2 = list(map(lambda x: int(x), utils.normalized_cross(lineA, lineB)))

    # Determine left and right
    horizon_points = [point1, point2]
    horizon_points.sort(key=lambda v: v[0])

    horizon = utils.normalized_cross(point1, point2)

    # As every line pair represents the two heads, every even/odd point creates a pair to be used
    # for a nadir vanishing point

    zenith_lines = list()

    zenith_lines.append([data['points'][0], data['points'][2]])
    zenith_lines.append([data['points'][1], data['points'][3]])
    zenith_lines.append([data['points'][4], data['points'][6]])
    zenith_lines.append([data['points'][5], data['points'][7]])

    # Give those pairs to horizon library to obtain a nadir vanishing point
    zenith_edgelets = horizon_detector.HorizonDetectorLib.line_properties(zenith_lines)
    ground_zenith = horizon_detector.HorizonDetectorLib.ransac_nadir_vp(zenith_edgelets, 15)

    ground_focal = horizon_detector.HorizonDetectorLib.find_focal_length(horizon_points, ground_zenith)

    fig, axis = plt.subplots()
    axis.imshow(data['image'])
    axis.set_title("Ground truth horizon and nadir")

    axis.plot((point1[0]), (point1[1]), 'ro')
    axis.plot((point2[0]), (point2[1]), 'ro')
    axis.plot(int(ground_zenith[0]), int(ground_zenith[1]), 'ro')

    axis.plot((point1[0], point2[0]), (point1[1], point2[1]), color='c')

    plt.show(fig)

    with open(path.join(image_dir, "gt.txt"), "w") as groundTruthWriter:
        groundTruthWriter.write("Horizon {}, Horizon Points {}, Zenith {}, Focal {}"
                                .format(horizon, horizon_points, ground_zenith, ground_focal))

    return horizon, horizon_points, ground_zenith, ground_focal


# endregion

# Finds the mask pixel with lowest y
def find_segmentation_offset(segmented_img):
    indices = np.nonzero(segmented_img)

    return min(indices[0]) if len(indices[0]) > 0 else 0


# Main method that is used to rectify the ground plane
def rectify_groundPlane(image_path,
                        segmented_img_path,
                        detection_data_file,
                        frames_per_check,
                        determine_ground_truth,
                        ground_truth_horizon,
                        ground_truth_zenith,
                        ground_truth_focal,
                        provided_horizon,
                        provided_zenith,
                        provided_focal,
                        draw_features):
    image_dir = path.dirname(image_path)

    # Manuel testing debugging part:
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    image_points = np.array([[0, height], [width, height], [width, 0], [0, 0]])
    center = [width / 2, height / 2, 1]

    segmented_img = cv2.imread(segmented_img_path)
    ground_truth_horizon_pnts = [0, 0]

    if determine_ground_truth:
        clicked_points = dict()
        clicked_points['image'] = np.copy(image)
        clicked_points['points'] = []

        window_name = "Click on 8 points on the image whree every 2 consecutive point describes a line and every 2 lines" \
                      "meets at horizon"
        clicked_points['windowName'] = window_name

        cv2.imshow(window_name, image)
        cv2.setMouseCallback(window_name, mouse_handler, clicked_points)
        cv2.waitKey(0)

        ground_truth_horizon, ground_truth_horizon_pnts, \
        ground_truth_zenith, ground_truth_focal = processGTlines(image_dir, clicked_points)

    use_ground_truth = all(
        list(map(lambda x: x is not None, [ground_truth_horizon, ground_truth_zenith, ground_truth_focal])))

    if use_ground_truth:
        print("Ground-truth vanishing line {}".format(ground_truth_horizon))
        print("Ground-truth nadir vp {}".format(ground_truth_zenith))
        print("Ground-truth focal length {}".format(ground_truth_focal))
        print("Ground-truth horizon corners {}".format(ground_truth_horizon_pnts))

        ground_fov = utils.focal_to_fov(ground_truth_focal, height)

        print("Ground-truth fov {}".format(ground_fov))

    # Extract the lines from the whole image
    image_lines = horizon_detector.HorizonDetectorLib.extract_image_lines(image, image_dir)

    # Extract the pedestrian paths as lines (postures or bounding boxes)

    # Obtain the postures of the pedestrian as lines too, to find the VP
    # The sample taken every few frames can become a performance concern
    pedestrian_posture_paths, pedestrian_postures, pedestrian_posture_trajectory, raw_trajectories = \
        horizon_detector.HorizonDetectorLib.parse_pedestrian_detection(np.copy(image),
                                                                       detection_data_file,
                                                                       frames_per_check)

    # Single posture implementation requires pedestrian ID's
    # pedestrian_posture_paths_single, pedestrian_postures_single, pedestrian_posture_trajectory_single = \
    # horizon_detector.HorizonDetectorLib.parse_pedestrian_detection(np.copy(image),
    #                                                                detection_data_file,
    #                                                                5,
    #                                                                tracker_id=[2])  # Parameter

    # We assume the people don't change their velocities much
    # and calculate a homography between a path with multiple detections

    # - Postures as both feet and head locations and trajectories (too sensitive to noise, not preferable)
    # - Single tracker posture
    # - Feet Trajectory only, RANSAC based
    # - Feet Trajectory + Hough Lines from the navigable area, RANSAC based
    # - Hough Lines from navigable area only

    # Second property is used for nadir point determination
    vp_determination_methods = {
        # 'posture_lines': [pedestrian_posture_paths, pedestrian_postures, pedestrian_posture_trajectory],
        # 'single_lines': [pedestrian_posture_paths_single, pedestrian_postures_single,
        #                  pedestrian_posture_trajectory_single],
        # 'hough': [image_lines, None, None, None],
        'postures_hough': [
            [np.concatenate((pedestrian_posture_paths[j], image_lines[j]), axis=0)
             for j in range(3)],  # Used for horizon
            pedestrian_postures,  # Used for nadir
            pedestrian_posture_trajectory,  # Used for circular points
            raw_trajectories]
    }

    row = int(np.sqrt(len(vp_determination_methods.keys())))
    col = int(np.sqrt(len(vp_determination_methods.keys())))

    horizon_fig, horizon_axis = plt.subplots(row, col)
    rectified_fig, rectified_axis = plt.subplots(row, col)

    for i, k in enumerate(vp_determination_methods):

        # Determine the horizon
        lines = vp_determination_methods[k][0]
        postures = vp_determination_methods[k][1]
        trajectories = vp_determination_methods[k][2]
        raw_trajectories = vp_determination_methods[k][3]

        if row == 1 and col == 1:
            plot_axis = horizon_axis
        else:
            plot_axis = horizon_axis[i // col, i % col]

        # plot_axis.set_title(str(k))
        plot_axis.imshow(image)

        data_not_provided = any(list(map(lambda x: x is None, [provided_horizon, provided_zenith, provided_focal])))

        # In case no suitable lines for horizon calculation is found, skip
        if len(lines[0]) > 0:

            if data_not_provided:

                leftVP, rightVP, nadir_vp = horizon_detector.HorizonDetectorLib.determineVP(
                    lines,
                    plot_axis=plot_axis,
                    postures=postures,
                    constraints=raw_trajectories,
                    draw_features=draw_features)

                # TODO: TESTING GROUND_TRUTH
                # leftVP, rightVP = ground_truth_horizon_pnts
                # zenith_vp = np.array(ground_truth_zenith)

                # Get the horizon points
                horizon = [leftVP, rightVP]

                # If the data was trajectory, only horizon is found.
                # Nadir needs to be found independently using the postures
                if nadir_vp is None and postures is not None:
                    nadir_vp = horizon_detector.HorizonDetectorLib.ransac_nadir_vp(pedestrian_postures,
                                                                                   horizon,
                                                                                   raw_trajectories,
                                                                                   center)
                nadir_vp = nadir_vp / nadir_vp[2]
                focal_length = horizon_detector.HorizonDetectorLib.find_focal_length(horizon, nadir_vp)

            else:

                leftVP = np.array(provided_horizon[:3])
                rightVP = np.array(provided_horizon[3:])
                horizon = [leftVP, rightVP]

                nadir_vp = np.array(provided_zenith)
                focal_length = provided_focal

            # region plot_horizons

            plot_axis.imshow(image)

            line_X = np.arange(leftVP[0], rightVP[0])[:, np.newaxis]
            horizon_line = np.cross(leftVP, rightVP)
            horizon_line = horizon_line / horizon_line[2]

            # line_X = [[vp_left[0], vp_right[0]]]
            line_Y = list(
                map(lambda point: (-horizon_line[0] * point[0] - horizon_line[2]) / horizon_line[1], line_X))

            if use_ground_truth:
                gt_Y = np.array(list(
                    map(lambda point: (-ground_truth_horizon[0] * point[0] - ground_truth_horizon[2])
                                      / ground_truth_horizon[1], line_X)))
                plot_axis.plot(line_X, gt_Y, color='c')

                s = 0
                for i in range(len(gt_Y)):
                    s += (abs(line_Y[i] - gt_Y[i]) ** 1)

            plot_axis.plot(line_X, line_Y, color='r')

            # Plot the points
            plot_axis.plot(leftVP[0], leftVP[1], 'bo')
            plot_axis.plot(rightVP[0], rightVP[1], 'bo')

            plot_axis.tick_params(axis='both', which='major', labelsize=14)

            # Plot the zenith points

            # plot_axis.plot((zenith_vp[0]), (zenith_vp[1]), 'ro')
            # plot_axis.plot((center[0]), (center[1]), 'yo')
            #
            # if use_ground_truth:
            #     plot_axis.plot((ground_truth_zenith[0]), (ground_truth_zenith[1]), 'co')

            # endregion

            # Determine the region under the horizon in the image
            # as rectifying the image above horizon
            # causes problems. The image should be updated as such.

            # region report horizon

            # Find the focal_unity length from the triangle of vanishing points
            if data_not_provided:
                focal_length = horizon_detector.HorizonDetectorLib.find_focal_length(horizon, nadir_vp)

            fov = utils.focal_to_fov(focal_length, height)

            print("Method: {}, left: {}, right: {}".format(k, leftVP[:2], rightVP[:2]))
            print("nadir: {}, focal: {} with fov: {}".format(nadir_vp[:2], focal_length, fov))

            with open(path.join(image_dir, "found_horizon.txt"), "w") as HorizonWriter:
                HorizonWriter.write("Method: {}, left: {}, right: {},  nadir: {}, focal: {} with fov: {}"
                                    .format(k, leftVP[:2], rightVP[:2], nadir_vp[:2], focal_length, fov))

            # endregion

            # Complete stratified approach that rectifies and translates the navigable area
            warped_result_segmented, model_points, H = compute_homography_and_warp(segmented_img,
                                                                                   list(leftVP),
                                                                                   list(rightVP),
                                                                                   trajectories,
                                                                                   image_points,
                                                                                   method=k,
                                                                                   output_dir=image_dir)

            # region result_validation

            # There are some constraints on the resulting fov and found horizon
            # - As the videos are taken from a high altitude, the nadir is expected to be below the horizon (birdview)
            # - The rectified region should be a convex polygon
            # If any of them is not satisfied, then the original image should be returned

            if not check_polygon_convexity(model_points):
                # The rectification was unsucessful, restore the image
                print("The perspective correction was unsuccessful, returning original result")
                warped_result_segmented = image
                model_points = image_points

            # endregion

            if row == 1 and col == 1:
                plot_axis = rectified_axis
            else:
                plot_axis = rectified_axis[i // col, i % col]

            plot_axis.set_title(str(k))
            plot_axis.imshow(warped_result_segmented)

            for i in range(len(model_points)):
                plot_axis.plot([model_points[i][0], model_points[(i + 1) % 4][0]],
                               [model_points[i][1], model_points[(i + 1) % 4][1]], 'b')

            plt.imsave(path.join(image_dir, "warped_result_" + k + ".png")
                       , warped_result_segmented)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            warped_org = transform.warp(image,
                                        np.linalg.inv(H),
                                        clip=False,
                                        output_shape=(height, width))

            plt.imsave(path.join("warped_result_original_" + k + ".png")
                       , warped_org)

            # Intrinsic matrix for camera, same for both model (rectification camera) and scene camera
            intrinsic = np.array([[focal_length, 0, warped_result_segmented.shape[1] / 2],
                                  [0, focal_length, warped_result_segmented.shape[0] / 2],
                                  [0, 0, 1]])

            # Output the internal and external parameters through a text file
            # Using the pnp methods, map the model points to image points
            # Has concepts from:
            #   Ezio Malis, Manuel Vargas, and others. Deeper understanding of the homography decomposition for vision-based control. 2007.

            cameraCalibration.CameraCalibration.extract_camera_parameters(k,
                                                                          image,
                                                                          warped_org,
                                                                          warped_result_segmented,
                                                                          model_points,
                                                                          image_points,
                                                                          intrinsic,
                                                                          image_dir)

    plt.show(horizon_fig)
    horizon_fig.savefig(path.join(image_dir, "horizons.png"))
    plt.show(rectified_fig)


if __name__ == "__main__":
    print("Welcome to the perspective corrector")

    aparser = argparse.ArgumentParser(description="Using the image perspective cues and pedestrian detection data"
                                                  "in order to rectify the ground plane to be used for navigation")
    aparser.add_argument("--image_path", help="Image to be corrected")
    aparser.add_argument("--segmented_img_path", help="Segmented version of the given image")
    aparser.add_argument("--detection_data_file", help="Detection txt to be used")
    aparser.add_argument("--frames_per_check", help="How many frames needs to pass before new position is sampled",
                         type=int, default=60)
    aparser.add_argument("--determine_ground_truth", nargs='?',
                         help="If true, program will find gt first with the aid of the user,"
                              "Ignored when it is provided",
                         const=True, default=False)
    aparser.add_argument("--ground_truth_horizon", nargs='*',
                         help="Homogenous coordinates of the ground truth, all 3 coordinates",
                         type=float, default=None)
    aparser.add_argument("--ground_truth_zenith", nargs='*', help="nadir VP coordinates of the ground truth",
                         type=float, default=None)
    aparser.add_argument("--ground_truth_focal", help="Focal length of the ground truth",
                         type=float, default=None)
    aparser.add_argument("--provided_horizon", nargs='*',
                         help="Homogenous coordinates for each vanishing point from left to right",
                         type=float, default=None)
    aparser.add_argument("--provided_zenith", nargs='*', help="nadir VP coordinates of the to be used",
                         type=float, default=None)
    aparser.add_argument("--provided_focal", help="Focal length to be used for reconstruction",
                         type=float, default=None)
    aparser.add_argument("--draw_features", nargs='?',
                         help="Draw the postures, image lines etc. on the image when determining the horizon",
                         const=True, default=False)

    args = vars(aparser.parse_args())

    rectify_groundPlane(**args)
