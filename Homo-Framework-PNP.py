import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import skimage.transform as transform
import math
import argparse

import camera_calibration
from horizon_detection import horizon_detector
from camera_calibration import cameraCalibration

'''
Homography framework that applies perpective correction to given survailance image
by converting it to bird-view perspective as close as possible.

Works on a single image, camera is assumed to be not calibrated and focal_unity length is not known.

Algorithm steps
- Using RANSAC, find the best point that represents the vanishing points, according to
its reachabilty from other edges in the image
- Identify the vanishing line as the combination of left and right VP (manhattan world assumption) 

IF THE VANISHING LINE IS NOT PARALLEL:
- ROTATE THE IMAGE WITH THE ANGLE THAT THE VANISHING LINE HAS WITH THE X AXIS
- CLIP THE IMAGE ACCORDINGLY
- PROCEED WITH THE HOMOGRAPHY 
- AFTER THE UNITY'S CAMERA IS PLACED AND ADJUSTED ACCORDING TO HOMOGRAPHY,
APPLY THE INVERSE OF THIS ROTATION

- Apply the homography logic we have talked before

- Using the vanishing points, approximate the focal_unity length, using triangle properties
https://www.coursera.org/learn/robotics-perception/lecture/jnaLs/how-to-compute-intrinsics-from-vanishing-points

author Yalım Doğan

Some ideas and code segments are from: https://github.com/chsasank/Image-Rectification (uses Scikit rather than OpenCV)
which is an implementation of Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe. "Auto-rectification of user photos." 
2014 IEEE International Conference on Image Processing (ICIP). IEEE, 2014.
'''


#For stratified rectification, affine matrix requires 2 parameters that are related to
#the circular points that lie on the absolute conic. In order to obtain them, lines with
#known ratios are necessary. For this, we will process the given trajectory lines and pick
#paths that have constant velocity according to reporjection.

#We also assume that no consecutive trajectory is parallel and we take only positive B values
#as negative values cause mirror effect on ground planes

#The logic is adopted from "Ground Plane Rectification by Tracking Moving Objects"
def extract_circular_points(trajectory_lines, P, method):

    from matplotlib.patches import Circle

    circles = []
    intersections = set([])

    fig, ax = plt.subplots()

    centers, directions, strengths = trajectory_lines

    #For every 2 line, calculate centre and radius of the circle using line endpoints
    for i in range(len(trajectory_lines[0]) // 2):
        line_1 = [[centers[i][0] - (directions[i][0] * strengths[i]),
                        centers[i][0] + (directions[i][0] * strengths[i]),1],
                       [centers[i][1] - (directions[i][1] * strengths[i]),
                        centers[i][1] + (directions[i][1] * strengths[i]),1]]
        line_2 = [[centers[i+1][0] - (directions[i+1][0] * strengths[i+1]),
                        centers[i+1][0] + (directions[i+1][0] * strengths[i+1]),1],
                       [centers[i+1][1] - (directions[i+1][1] * strengths[i+1]),
                        centers[i+1][1] + (directions[i+1][1] * strengths[i+1]),1]]


        #Apply projection to lines
        line_1 = np.array([np.dot(P, np.array(t).T) for t in line_1])
        line_1[0] = line_1[0] / line_1[0][2]
        line_1[1] = line_1[1] / line_1[1][2]


        line_2 = np.array([np.dot(P, np.array(t).T) for t in line_2])
        line_2[0] = line_2[0] / line_2[0][2]
        line_2[1] = line_2[1] / line_2[1][2]

        #Assume length ratio of the trajectories are 1 in the world plane
        s = 1

        delta_x1 = line_1[1][0] - line_1[0][0]; delta_y1 = line_1[1][1] - line_1[0][1]
        delta_x2 = line_2[1][0] - line_2[0][0]; delta_y2 = line_2[1][1] - line_2[0][1]

        c_alpha, c_beta = ((delta_x1 * delta_y1 - pow(s,2) * delta_x2 * delta_y2)
                           / (pow(delta_y1,2) - pow((s * delta_y2),2)), 0)
        radius = np.abs(s * (delta_x2 * delta_y1 - delta_x1 * delta_y2) / (pow(delta_y1 , 2) - pow((s * delta_y2) , 2)))


        #For every circle found, find its intersection with every other circle
        #Formula: http://mathworld.wolfram.com/Circle-CircleIntersection.html
        for circle in circles: # The "circle" here shadows the one above
            d = circle.center[0] - c_alpha
            r = circle.radius
            intersection_x = c_alpha + (pow(d,2) - pow(r,2) + pow(radius,2)) \
                             / (2 * d)
            intersection_y = np.sqrt((r - radius - d) * (- d - r + radius) * (-d + r + radius) * (d + r + radius)) / (2*d)

            if not math.isnan(intersection_y): # If there is intersection
                intersections.add((intersection_x,abs(intersection_y)))
                # intersections.add((intersection_x, -intersection_y)) Negative value is unused

                # Draw the intersection points
                ax.plot([intersection_x], [intersection_y], 'ro')
                ax.plot([intersection_x], [-intersection_y], 'ro')

        # Add the circle to the list
        circle = Circle((c_alpha, c_beta), radius, color = 'b', fill=False)
        circles.append(circle)

        ax = plt.gca()
        ax.add_patch(circle)

    mean_intersection = intersections
    #Find and plot the mean of the intersection points
    if len(intersections) > 1:
        mean_intersection = np.mean(list(intersections), axis = 0)

    if len(intersections) > 0:
        ax.plot(mean_intersection[0], mean_intersection[1], 'bo')

    ax.axis('scaled')
    ax.set_title("Trajectory Circular Points For {}".format(method))
    plt.show(block=False)

    return intersections, mean_intersection

def compute_homography_and_warp(image, vp1, vp2, trajectories, corners, clip=True, clip_factor=3, method="posture"):
    """Compute homography from vanishing points and warp the image.

    It is assumed that vp1 and vp2 correspond to horizontal and vertical
    directions, although the order is not assumed.
    Firstly, projective transform is computed to make the vanishing points go
    to infinty so that we have a fronto parellel view. Then,Computes affine
    transfom  to make axes corresponding to vanishing points orthogonal.
    Finally, Image is translated so that the image is not missed. Note that
    this image can be very large. `clip` is provided to deal with this.

    Parameters
    ----------
    image: ndarray
        Image which has to be wrapped.
    vp1: ndarray of shape (3, )
        First vanishing point in homogenous coordinate system.
    vp2: ndarray of shape (3, )
        Second vanishing point in homogenous coordinate system.
    clip: bool, optional
        If True, image is clipped to clip_factor.
    clip_factor: float, optional
        Proportion of image in multiples of image size to be retained if gone
        out of bounds after homography.
    Returns
    -------
    warped_img: ndarray
        Image warped using homography as described above.
    """

    height,width,_ = image.shape

    # Find Projective Transform
    vanishing_line = np.cross(vp1, vp2)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2] #As h32 needs to be 1 in projection

    # Determine a and b for the affine component of the homography
    intersections, mean_intersection = extract_circular_points(trajectories, H, method)

    # for mean_intersection in intersections:
    A = np.eye(3, 3)
    a = np.nan; b = np.nan

    if len(mean_intersection) > 0:
        a = mean_intersection[0]
        b = mean_intersection[1]

        A[0, 0] = 1 / b
        A[0, 1] = -a / b

    H = np.dot(A, H)

    # Find directions corresponding to vanishing points
    v_post1 = np.dot(H, vp1) #Direction is found by multiplying with the projection matrix
    v_post2 = np.dot(H, vp2)

    #Normalize
    v_post1 = v_post1 / np.sqrt(v_post1[0]**2 + v_post1[1]**2)
    v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)

    # X and Y parts of the directions are used to find the angle it makes with

    #Arctan is used to find the small angle with the mentioned axis, having X at top gives result according to Y axis
    directions = np.array([[v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
                           [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]]])

    # TODO: Change 0 ad 1 to find it according to Y axis
    #Finds the angle it makes with the horizon
    thetas = np.arctan2(directions[0], directions[1])

    # Find the index for
    h_ind = np.argmin(np.abs(thetas))

    # Find positive angle among the rest for the vertical axis
    if h_ind // 2 == 0:# If the angle is positive
        v_ind = 2 + np.argmax([thetas[2], thetas[3]])
    else:
        v_ind = np.argmax([thetas[2], thetas[3]])

    #Rotation matrix
    A1 = np.array([[directions[0, v_ind], directions[0, h_ind], 0],
                   [directions[1, v_ind], directions[1, h_ind], 0],
                   [0, 0, 1]])
    # Might be a reflection. If so, remove reflection.
    if np.linalg.det(A1) < 0:
        A1[:, 0] = -A1[:, 0]

    A = np.linalg.inv(A1)

    # Translate so that whole of the image is covered
    inter_matrix = np.dot(A, H)

    #Cropping matrix
    cords = np.dot(inter_matrix, [[0, 0, image.shape[1], image.shape[1]],
                                  [0, image.shape[0], 0, image.shape[0]],
                                  [1, 1, 1, 1]])
    cords = cords[:2] / cords[2]

    tx = min(0, cords[0].min())
    ty = min(0, cords[1].min())

    max_x = cords[0].max() - tx
    max_y = cords[1].max() - ty

    if clip:
        # These might be too large. Clip them.
        max_offset = max(image.shape) * clip_factor / 2
        tx = max(tx, -max_offset)
        ty = max(ty, -max_offset)

        max_x = min(max_x, -tx + max_offset)
        max_y = min(max_y, -ty + max_offset)

    max_x = int(max_x)
    max_y = int(max_y)

    T = np.array([[1, 0, -tx],
                  [0, 1, -ty],
                  [0, 0, 1]])

    final_homography = np.dot(T, inter_matrix)

    S = np.array([[width / max_x , 0 ,0],
                 [0, height / max_y, 0],
                 [0 , 0 ,1]])

    final_homography = np.dot(S, final_homography)

    # final_homography = np.eye(3)

    warped_img = transform.warp(image, np.linalg.inv(final_homography),
                                output_shape=(height, width))


    transformed_corners = transform.matrix_transform(corners, final_homography)

    return warped_img, transformed_corners, final_homography


def mouse_handler(event, x, y, flags, data):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(data) < 8:
            data['points'].append([x, y, 1])
            cv2.circle(data['image'], (x, y), 3, (0, 0, 255), 5, 16)
            cv2.imshow(data['windowName'], data['image'])

def determineCross(line1, line2):

    cross = np.cross(line1, line2)
    return cross / cross[2]

def processPoints(data):

    lineA = determineCross(*data['points'][0:2])
    lineB = determineCross(*data['points'][2:4])
    point1 = list(map(lambda x: int(x), determineCross(lineA, lineB)))

    lineA = determineCross(*data['points'][4:6])
    lineB = determineCross(*data['points'][6:8])
    point2 = list(map(lambda x: int(x), determineCross(lineA, lineB)))

    vanishing = determineCross(point1, point2)

    cv2.circle(data['image'], (point1[0], point1[1]), 3, (0, 255, 0), 5, 16)
    cv2.circle(data['image'], (point2[0], point2[1]), 3, (0, 255, 0), 5, 16)

    cv2.line(data['image'], (point1[0], point1[1]), (point2[0], point2[1]), (255, 0, 0), 5, 16)
    cv2.imshow(data['windowName'], data['image'])
    cv2.waitKey(0)

    return vanishing


#Main method that is used to rectify the ground plane
def rectify_groundPlane(image_path, segmented_img_path, detection_data_file, frames_per_check, ground_truth_horizon, draw_features):

    # Manuel testing debugging part:
    image = cv2.imread(image_path)
    segmented_img = cv2.imread(segmented_img_path)



    if ground_truth_horizon is None:
        clicked_points = {}
        clicked_points['image'] = np.copy(image)
        clicked_points['points'] = []

        window_name = "Click on 8 points on the image whree every 2 consecutive point describes a line and every 2 lines" \
                   "meets at horizon"
        clicked_points['windowName'] = window_name

        cv2.imshow(window_name, image)
        cv2.setMouseCallback(window_name, mouse_handler, clicked_points)
        cv2.waitKey(0)

        ground_truth_horizon = processPoints(clicked_points)

    else:
        cv2.imshow("Image of Interest", image)
        cv2.waitKey(5)

    print("Ground-truth vanishing line {}".format(ground_truth_horizon))

    # Extract the lines from the whole image
    image_lines = horizon_detector.HorizonDetectorLib.extract_image_lines(image)

    #Extract the pedestrian paths as lines (postures or bounding boxes)

    #Obtain the postures of the pedestrian as lines too, to find the VP
    # Parameter : The sample taken every few frames can become a performance concern
    # TODO: Turn tracked pedestrian id's into parameters
    pedestrian_posture_paths, pedestrian_postures = horizon_detector.HorizonDetectorLib.parse_pedestrian_detection(np.copy(image), detection_data_file, 15, use_bounding_boxes=False, returnPosture= True)
    pedestrian_posture_paths_single, _ = horizon_detector.HorizonDetectorLib.parse_pedestrian_detection(np.copy(image), detection_data_file, 10, use_bounding_boxes=False, tracker_id=[56,17,10]) # Parameter

    #If we assume the people doesn't change their velocities much
    #and calculate a homography between a path with multiple detections

    #These methods use bounding boxes
    pedestrian_bb_paths, _ = horizon_detector.HorizonDetectorLib.parse_pedestrian_detection(np.copy(image), detection_data_file, 5)
    trajectory_lines, _ = horizon_detector.HorizonDetectorLib.parse_pedestrian_detection(np.copy(image), detection_data_file, 5, False, True) # Parameter

    # pedestrian_postures_image_lines = [np.concatenate((pedestrian_postures[j], image_lines[j]), axis=0)
    #                        for j in range(3)] UNUSED

    # - Postures as both feet and head trajectories (too sensitive to noise, not preferable)
    # - Single tracker posture (better than above)
    # - BB's head and feet postures again sensitive to noise
    # - Feet Trajectory only, RANSAC based
    # - Feet Trajectory + Hough Lines from the navigable area, RANSAC based
    # - Hough Lines from navigable area only
    vp_determination_methods = {
        # 'posture': [pedestrian_posture_paths, False],
        # 'single': [pedestrian_posture_paths_single, False],
        'bb': [pedestrian_bb_paths, False],
        # 'hough': [image_lines, True],
        # 'trajectory': [trajectory_lines, True], DONT USE
        # 'trajectory_hough': [[np.concatenate((trajectory_lines[j], image_lines[j]), axis=0)
        #                       for j in range(3)] , True],
        # 'postures_hough': [[np.concatenate((pedestrian_posture_paths[j], image_lines[j]), axis=0)
        #                       for j in range(3)], True]
    }

    row = 3
    col = 2
    horizon_fig, horizon_axis = plt.subplots(row, col)
    rectified_fig, rectified_axis = plt.subplots(row,col)
    height,width, _ = image.shape

    for i, k in enumerate(vp_determination_methods):

        # Determine the horizon
        lines = vp_determination_methods[k][0]

        plot_axis = horizon_axis[i//col, i % col]
        plot_axis.set_title(str(k))
        plot_axis.imshow(image)

        # In case no suitable lines for horizon calculation is found, skip
        if len(lines[0]) > 0:
            leftVP, rightVP, zenith_vp, r_mse = horizon_detector.HorizonDetectorLib.determineVP(lines,
                                                                                         np.copy(image),
                                                                                         plot_axis= plot_axis,
                                                                                         ground_truth=ground_truth_horizon,
                                                                                         asTrajectory=vp_determination_methods[k][1],
                                                                                         draw_features = draw_features)

            plot_axis.set_title(str(k) + " RMSE: " + str(r_mse))
            horizon = [leftVP, rightVP]

            #Get the image corners
            image_points = np.array([[0, height], [width, height], [width, 0], [0, 0]])

            #Find the focal_unity length from the triangle of vanishing points
            zenith_vp, focal_length, center = horizon_detector.HorizonDetectorLib.ransac_zenith_vp(pedestrian_postures, horizon, [width/2, height/2, 1], zenith=zenith_vp)
            zenith_vp = zenith_vp / zenith_vp[2]
            fov = math.degrees(2 * math.atan2(height , (2 * focal_length)))

            print("Method: {}, left: {}, right: {}".format(k,leftVP[:2],rightVP[:2]))
            print("Zenith: {}, focal: {} with fov: {}".format(zenith_vp[:2], focal_length, fov))

            plot_axis.imshow(image)
            plot_axis.plot((zenith_vp[0]), (zenith_vp[1]), 'go')
            plot_axis.plot((center[0]), (center[1]), 'yo')

        if False:
            # Complete stratified approach that rectifies and translates the navigable area
            warped_result, model_points, H = compute_homography_and_warp(segmented_img,
                                                                         list(leftVP),
                                                                         list(rightVP),
                                                                         trajectory_lines,
                                                                         image_points,
                                                                         clip=True,
                                                                         clip_factor=3,
                                                                         method=k)
            plot_axis = rectified_axis[i//col, i % col]
            plot_axis.set_title(str(k))
            plot_axis.imshow(warped_result)

            for i in range(len(model_points)):
                plot_axis.plot([model_points[i][0], model_points[(i + 1) % 4][0]],
                         [model_points[i][1], model_points[(i + 1) % 4][1]], 'b')


            plt.imsave("warped_result_" + k +".png", warped_result)

            #------------ABONDEN DON'T USE--------------
            # My method using imaginary plane which utilizes 4 point approach
            # allignedImg, leftVP, rightVP, alligned_corners = allignHorizon(segmented_img, horizon)
            # warped_result, model_points, H = applyParametretizedHomography(allignedImg, list(leftVP), list(rightVP), alligned_corners, k)
            # ------------------------------------------

            #Intrinsic matrix for camera, same for both model (rectification camera) and scene camera
            K = np.array([[focal_length,0,warped_result.shape[1]/2],
                         [0,focal_length,warped_result.shape[0]/2],
                         [0,0,1]])

            #Output the internal and external parameters through a text file
            # Using the p3p methods, map the model points to image points
            #TODO: Has concepts from:
            #   Ezio Malis, Manuel Vargas, and others. Deeper understanding of the homography decomposition for vision-based control. 2007.

            corners_homo = [[*corner, 1] for corner in image_points]

            left_border = np.cross(corners_homo[0], corners_homo[3])
            right_border = np.cross(corners_homo[1], corners_homo[2])
            right_border = right_border / right_border[2]

            horizon_line = np.cross(horizon[0], horizon[1])

            corners_homo[0] = np.cross(left_border, horizon_line)
            image_points[3] = (corners_homo[0] / corners_homo[0][2])[:2]
            corners_homo[1] = np.cross(right_border, horizon_line)
            image_points[2] = (corners_homo[1] / corners_homo[1][2])[:2]

            image_points = [[pnt[0], max(0,pnt[1])] for pnt in image_points]

            cameraCalibration.CameraCalibration.extractCameraParameters(image,
                                                                        warped_result,
                                                                        model_points,
                                                                        image_points,
                                                                        K, H)

    plt.show(horizon_fig)
    plt.show(rectified_fig)

# TODO:
# - Have a through test with all videos, coupled with Unity
# - We will also show the stratified approach, so be sure to document it good as well,
# - When writing, read about pnp as it is used for placement
if __name__ == "__main__":
    print("Welcome to the perspective corrector")

    aparser = argparse.ArgumentParser(description="Using the image perspective cues and pedestrian detection data"
                                                  "in order to rectify the ground plane to be used for navigation")
    aparser.add_argument("--image_path", help = "Image to be perspectively corrected")
    aparser.add_argument("--segmented_img_path", help = "Segmented version of the given image")
    aparser.add_argument("--detection_data_file", help = "Detection txt to be used")
    aparser.add_argument("--frames_per_check", help = "How many frames needs to pass before new position is sampled",
                         type=int, default=60)
    aparser.add_argument("--ground_truth_horizon", nargs='+', help = "Homogenous coordinates of the ground truth, all 3 coordinates",
                         type=float, default = None)
    aparser.add_argument("--draw_features", nargs='+', help = "Draw the postures, image lines and such features on the image when determining the horizon",
                         type=bool, default=False)

    args = vars(aparser.parse_args())

    rectify_groundPlane(**args)
