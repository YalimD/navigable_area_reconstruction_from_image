import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import argparse

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


#Extracts the edges and hough lines from the image
#Taken from IMAGE RECTIFICATION
def extract_image_lines(img):

    image = np.copy(img)

    #Bilateral filtering which keeps the edges sharp, but textures blurry
    #seems to decrease the noisy edges that cause too many detection results
    #Read bilateral filters: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
    #Everyone is affected by similar and close pixels. If neighbour is not similar, then its effect is small
    #Makes things more "comical"
    image = cv2.bilateralFilter(image,9,60,60)

    # The image needs to be converted to grayscale first
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection (canny for now)
    grayscale_img = cv2.Canny(grayscale_img, threshold1=75, threshold2=200, apertureSize=3)
    # grayscale_img = cv2.Sobel(grayscale_img,cv2.CV_8UC1,1,1)
    # grayscale_img = cv2.Laplacian(grayscale_img,cv2.CV_8UC1)

    # cv2.imshow("Detected Edges", grayscale_img)
    # cv2.waitKey(5)

    # Hough lines
    line_segments = cv2.HoughLinesP(grayscale_img, 1, np.pi / 180, threshold=10,minLineLength=10,
                                    maxLineGap=20)

    return lineProperties(line_segments,image)


#By parsing the detection data, it extracts the paths of each pedestrian for every x frames
#The head-feet positions depend on the selected method; using bounding boxes or pedestrian postures
#Note: Pedestrian postures are more prone to noise
#In addition, Using the pedestrian postures, determine the orthogonal vanishing point using RANSAC for minimizing
#distance between candidate line pairs
def parse_pedestrian_detection(image, detection_data, frames_per_check = 10, use_bounding_boxes = True, feet_only = False, tracker_id = None, returnPosture = False):

    latest_loc = {}
    paths = []
    postures = []

    num_of_frames = 0
    latest_frame =  0
    # fps = 30; resize = 1 #default
    # frames_per_check = 1

    use_bounding_boxes = False

    with open(detection_data) as detections:
        for i, line in enumerate(detections):

            #Checks pedestrian data every x frames
            #Read the line
            agent = line.split(',')
            frameID = int(agent[0])

            #Check every frames_per_check
            if frameID > latest_frame:
                latest_frame = frameID
                num_of_frames += 1
            if num_of_frames % frames_per_check == 0 and (tracker_id is None or int(agent[1]) == tracker_id):

                # Different methods for extracting head and feet
                if not use_bounding_boxes:
                    #Add the agent id to the dictionary latest_loc
                    headPos = list(map(float, agent[-2].split('/')))
                    feetPos = list(map(float, agent[-1].split('/')))
                else:
                    pos = list(map(float,agent[2:6]))
                    headPos = [pos[0] + pos[2] / 2, pos[1]]
                    feetPos = [pos[0] + pos[2] / 2, pos[1] + pos[3]]

                try:
                    prev_pos = latest_loc[agent[1]]

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
                        if not feet_only:
                            paths.append(head_path)

                        if returnPosture:
                            postures.append([headPos, feetPos])
                        latest_loc[agent[1]] = [headPos, feetPos]

                except:
                    latest_loc[agent[1]] = [headPos, feetPos]

            # else:
            #
            #     # if i == 0:
            #     #     l = line.split()
            #     #     fps = int(l[-1])
            #     #     resize = int(l[-2])
            #     # # Parse the pedestrian location
            #     # elif i - 1 == frame_index * fps * seconds_per_check:
            #     #     frame_index += 1
            #     #     locations = line.split(",")[1:]
            #     #
            #     #     # Every 7 values is an agent
            #     #     agents = [locations[i * 7:(i + 1) * 7] for i in range(len(locations) // 7)]
            #     #
            #     #     for agent in agents:
            #     #
            #     #         #Current agent
            #     #         headPos = (int(int(agent[1]) * resize),(float(agent[2]) - float(agent[6]) / 2) * resize)
            #     #         feetPos = (int(int(agent[1]) * resize),(float(agent[2]) + float(agent[6]) / 2) * resize)
            #     #
            #     #         try:
            #     #             #If the agent id is not found in the dictionary, this will raise KeyError
            #     #             prev_pos = latest_loc[agent[0]]
            #     #
            #     #             head_path = [prev_pos[0], headPos]
            #     #             feet_path = [prev_pos[1], feetPos]
            #     #
            #     #             #Detect and remove outliers.
            #     #             #Outliers are inconsistent detection box sizes. For example, a detection box that
            #     #             #shrinks while getting closer to the camera is considered as an outlier
            #     #             currentHeight = feetPos[1] - headPos[1]
            #     #             prev_height = prev_pos[1][1] - prev_pos[0][1]
            #     #             size_increased = currentHeight > prev_height
            #     #             higher_position = (headPos[1] + feetPos[1])/2 < (prev_pos[0][1] + prev_pos[1][1])/2
            #     #
            #     #             if size_increased ^ higher_position:
            #     #                 # The paths are held as pairs
            #     #                 paths.extend((head_path, feet_path))
            #     #                 latest_loc[agent[0]] = [headPos, feetPos]
            #     #
            #     #         except KeyError:
            #     #             latest_loc[agent[0]] = [headPos, feetPos]

    detections.close()

    return lineProperties(paths, image), lineProperties(postures, image)


#Extracts the line properties from given line segments
#Returns centers, directions and strengths
def lineProperties(lines, image):
    line_directions = []
    line_centers = []
    line_strengths = []

    for line in lines:
        line = np.array(line).flatten() #Compatibility

        x0 = int(line[0]); y0 = int(line[1])
        x1 = int(line[2]); y1 = int(line[3])

        line_centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
        line_directions.append((x1 - x0, y1 - y0))
        line_strengths.append(np.linalg.norm([x1 - x0, y1 - y0]))

        # Draw the detected lines on the original image
        # cv2.line(image, (x0, y0), (x1, y1), (0, 0, 255), 1)

    # Taken from https://github.com/chsasank/Image-Rectification
    # The direction vectors are normalized for easier calculation afterwards
    if len(line_directions) > 0:
        line_directions = np.array(line_directions) / np.linalg.norm(line_directions, axis=1)[:, np.newaxis]

    # cv2.imshow("Extracted Lines", image)
    # cv2.waitKey(0)

    return tuple(map(np.array, [line_centers, line_directions, line_strengths]))

#Taken from that github page
def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.

    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0] #as y is negative
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

#Determines the vanishing points on horizon using information coming from pedestrian paths
#OR uses the trajectory information and/or edges from the image to detect vanishing points
#which will determine the horizon
def determineVP(path_lines,image, plot_axis, asTrajectory = False):

    centers, directions, strengths = path_lines

    for i in range(centers.shape[0]):
        plot_axis.plot([centers[i][0] - (directions[i][0] * strengths[i]),
                        centers[i][0] + (directions[i][0] * strengths[i])],
                       [centers[i][1] - (directions[i][1] * strengths[i]),
                        centers[i][1] + (directions[i][1] * strengths[i])], 'r-')

    vxs = []
    vys = []

    if not asTrajectory:

        normals = edgelet_lines(path_lines)

        for i in range(len(normals)//2):

            head = normals[2*i]
            feet = normals[2*i+1]

            vx,vy,n = np.cross(head,feet); vx /= n; vy /= n
            if math.isfinite(vx) and math.isfinite(vy):
                vxs.append(vx)
                vys.append(vy)
                plot_axis.plot([vx],[vy],'bo')

        #Use RANSAC to determine the vanishing line

        # Parameter: Which top percentige of the points needs to be considered in order to get a good result on vanishing point
        ransac_ratio = 1

        sorted_ind = np.argsort(vys)
        sorted_ind = sorted_ind[:int(len(sorted_ind) * ransac_ratio)]

        vxs = np.array(vxs).reshape(-1, 1)[sorted_ind]
        vys = np.array(vys).reshape(-1, 1)[sorted_ind]

        ransac = linear_model.RANSACRegressor()
        ransac.fit(vxs,vys)
        line_X = np.arange(vxs.min(), vxs.max())[:, np.newaxis]
        line_Y = ransac.predict(line_X)

        vp_left = (line_X[0][0], line_Y[0][0],1)
        vp_right = (line_X[-1][0], line_Y[-1][0],1)

    else:
        model = ransac_vanishing_point(path_lines)
        vp1 = model / model[2]
        plot_axis.plot(vp1[0], vp1[1], 'bo')

        path_lines_reduced = remove_inliers(vp1, path_lines, 60)

        # Find second vanishing point
        model2 = ransac_vanishing_point(path_lines_reduced)
        vp2 = model2 / model2[2]
        plot_axis.plot(vp2[0], vp2[1], 'bo')

        # for i in range(centers.shape[0]):
        #     xax = [centers[i, 0], vp1[0]]
        #     yax = [centers[i, 1], vp1[1]]
        #     plot_axis.plot(xax, yax, 'b-.')
        #
        #     xax = [centers[i, 0], vp2[0]]
        #     yax = [centers[i, 1], vp2[1]]
        #     plot_axis.plot(xax, yax, 'b-.')

        line_X = [[vp1[0],vp2[0]]]
        line_Y = [[vp1[1],vp2[1]]]

        vp_left, vp_right = (vp1,vp2) if vp1[0] < vp2[0] else (vp2,vp1)

    plot_axis.plot(line_X, line_Y, color='g')

    return [vp_left,vp_right]

#Problem: The zenith vanishing point appears at the same side of the vanishing line against center of the image, which is impossible
#Solution: Find the distance to both and if distance to center is longer than the distance to horizon, ignore model
#Needs: center point and
def ransac_zenith_vp(edgelets, horizon, image_center, num_ransac_iter=2000, threshold_inlier=5):
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    #If number of points is lower than a threshold, don't create index spaces to compare models
    first_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 5] #Top 25 percentile
    second_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 2] #Top 50 percentile

    best_model = None
    best_votes = np.zeros(num_pts)

    #Normalize the horizon
    horizon_homogenous = np.cross(horizon[0],horizon[1])
    horizon_homogenous = horizon_homogenous / np.sqrt(horizon_homogenous[0]**2 + horizon_homogenous[1]**2)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        while ind2 == ind1: #Protection against low line count
            ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2) #Potential vanishing point
        current_model = current_model / current_model[2]

        #Its distance to center and the horizon
        horizon_distance = np.abs(np.dot(current_model.T, horizon_homogenous))
        centre_distance = np.linalg.norm(current_model[:2] - image_center[:2])

        if np.sum(current_model**2) < 1 or current_model[2] == 0 or horizon_distance < centre_distance:
            # reject degenerate candidates, which lie on the wrong side of the horizon
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            # logging.info("Current best model has {} votes at iteration {}".format(
            #     current_votes.sum(), ransac_iter))

    center_vp_dist = (np.linalg.norm(best_model[:2] - image_center[:2]))
    center_hor_dist = np.dot(np.array(image_center),horizon_homogenous)
    focal_length = np.sqrt(center_vp_dist * center_hor_dist)


    return best_model, focal_length

def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5):
    """Estimate vanishing point using Ransac.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.

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
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 5] #Top 25 percentile
    second_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 2] #Top 50 percentile

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        while ind2 == ind1: #Protection against low line count
            ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2) #Potential vanishing point

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            # logging.info("Current best model has {} votes at iteration {}".format(
            #     current_votes.sum(), ransac_iter))

    return best_model

def compute_votes(edgelets, model, threshold_inlier=5):
    """Compute votes for each of the edgelet against a given vanishing point.

    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.

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
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths

def remove_inliers(model, edgelets, threshold_inlier=10):
    """Remove all inlier edglets of a given model.

    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.

    Returns
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.
    """
    inliers = compute_votes(edgelets, model, 10) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets

#This function rotates the image according to the angle of the horizon in order to allign the horizon
#"horizontally"
def allignHorizon(image, horizon):

    angle = np.arctan((horizon[1][1] - horizon[0][1]) / (horizon[1][0] - horizon[0][0]))
    angle *= (180 / np.pi)
    print("Original angle of the horizon: {}".format(angle))

    row,col,_ = image.shape

    #Rotation without cropping (Thanks to Adrian Rosebrock for explanation)
	# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    centerx, centery = col // 2,row // 2

    rot = cv2.getRotationMatrix2D((centerx, centery), angle, 1)

    cos = np.abs(rot[0,0])
    sin = np.abs(rot[0,1])

    newwidth = int((row * sin) + (col * cos))
    newheight = int((row * cos) + (col * sin))

    #Also obtain the resulting y coordinate of the horizon

    #Rotate the end point according to found angle
    p = list(horizon[0])
    horizonImgCen = [newwidth // 2, np.matmul(rot,p)[1], 1] #Homogenous coord

    leftCorner = [0,newheight, 1]
    rightCorner = [newwidth, newheight,1]

    if horizonImgCen[1] >= 0: #If horizon is inside of the image

        #Interpolation variable adjusts how lower the top line of the polygon to be used for homography
        #is going to be lower than horizon.
        vpheight = 0.9
        leftP  = (int(horizonImgCen[0] * vpheight), int(horizonImgCen[1] * vpheight + leftCorner[1] * (1-vpheight)))
        rightP = (int(horizonImgCen[0] * vpheight + rightCorner[0] * (1 - vpheight)),
                  int(horizonImgCen[1] * vpheight + rightCorner[1] * (1 - vpheight)))
    else:
        leftP = np.cross(horizonImgCen,leftCorner)
        leftP = [-leftP[2] / leftP[0], 0]

        rightP = np.cross(horizonImgCen,rightCorner)
        rightP = [-rightP[2] / rightP[0],0]

    #Rotate the image without cropping the result
    rot[0, 2] += (newwidth / 2) - centerx
    rot[1, 2] += (newheight / 2) - centery

    #We need to put this rotation matrix in use when projecting
    dst = cv2.warpAffine(image, rot, (newwidth, newheight))

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))

    plt.plot([leftP[0],rightP[0]], [leftP[1],rightP[1]], 'bo')
    plt.plot()

    print("Left {}".format(leftP))
    print("Right {}".format(rightP))

    return dst, [*leftP,1], [*rightP,1]


#Apply the homography to the image according to the points found on
#lines that meet at the horizon
#TODO: Alligned image is not necessary, just determine the angle that is required
#for adjustment
def applyHomography(allignedImg, leftVP, rightVP, vp_method="posture"):

    #Old, imaginary plane solution

    height,width,_ = allignedImg.shape
    leftCorner = [0,height]
    rightCorner = [width, height]

    homographyPlane = np.array(
    [
        leftCorner,
        leftVP[:2],
        rightVP[:2],
        rightCorner
    ], np.float)

    # resize = 0.40  #Resize the resulting image
    # resize = 1 - resize
    # Old where the plane is enlarged
    # mappedPos = np.array([
    #     [width * resize / 2, height * (1 - resize / 2)],
    #     [width * resize / 2, height * resize / 2],
    #     [width * (1 - resize / 2), height * resize / 2],
    #     [width * (1 - resize / 2), height * (1 - resize / 2)]
    # ], np.float)

    # New, where the plane is shrunk
    mappedPos = np.array([
        [leftVP[0],height],
        leftVP[:2],
        rightVP[:2],
        [rightVP[0], height]
    ], np.float)

    homo_fitPlane, status = cv2.findHomography(homographyPlane, mappedPos)


    #Find the definition of the vanishing line as homogenous coordinates
    horizon = np.cross(leftVP,rightVP)
    # horizon = horizon / np.sqrt(horizon[0]**2 + horizon[1]**2)
    horizon = horizon / horizon[2]
    P = np.eye(3,3)

    P[2,:] = horizon
    P = P / P[2,2]

    print("P:{} with left {} and right {}".format(P,leftVP,rightVP))

    a = 1.3
    b = 0.12
    A = np.eye(3,3)
    A[0,0] = 1 / b
    A[0,1] = -a / b

    homo_stratified = np.dot(A, P)

    print("The homography old: {} and stratified: {}".format(homo_fitPlane, homo_stratified))

    from skimage import transform

    resize = 1 #As the resulting image is very big, it needs to be scaled down
    img_wrapped_stratified = transform.warp(allignedImg, np.linalg.inv(homo_stratified),
                                                output_shape=(int(width * resize),
                                                 int(height * resize)
                                                 ))


    img_wrapped_projection = cv2.warpPerspective(allignedImg, P,
                                                (int(width * resize),
                                                 int(height * resize)
                                                 ))

    img_wrapped_affine= cv2.warpPerspective(allignedImg, A,
                                                (int(width * resize),
                                                 int(height * resize)
                                                 ))

    img_wrapped_fitPlane = cv2.warpPerspective(allignedImg, homo_fitPlane,
                                                (allignedImg.shape[1]*2,
                                                 allignedImg.shape[0]*2
                                                 ))

    # img_wrapped_stratified = cv2.resize(img_wrapped_stratified, None, fx = 1/ resize, fy = 1 / resize,interpolation = cv2.INTER_NEAREST)


    cv2.imshow("The wrapped result of fitPLane with " + vp_method, img_wrapped_fitPlane)
    cv2.imshow("The wrapped result of stratified with " + vp_method, img_wrapped_stratified)
    cv2.imshow("The wrapped result of projection with " + vp_method, img_wrapped_projection)
    cv2.imshow("The wrapped result of affine with " + vp_method, img_wrapped_affine)

    cv2.waitKey(5)
    # cv2.destroyAllWindows()

    return img_wrapped_fitPlane


#For stratified rectification, affine matrix requires 2 parameters that are related to
#the circular points that lie on the absolute conic. In order to obtain them, lines with
#known ratios are necessary. For this, we will process the given trajectory lines and pick
#paths that have constant velocity according to reporjection.

#We also assume that no consecutive trajectory is parallel

#The logic is adopted from "Ground Plane Rectification by Tracking Moving Objects"
def extract_circular_points(trajectory_lines):

    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    circles = []

    fig, ax = plt.subplots()

    centers, directions, strengths = trajectory_lines

    #For every 2 line, calculate centre and radius of the circle using line endpoints
    for i in range(len(trajectory_lines[0]) // 2):
        line_1 = [[centers[i][0] - (directions[i][0] * strengths[i]),
                        centers[i][0] + (directions[i][0] * strengths[i])],
                       [centers[i][1] - (directions[i][1] * strengths[i]),
                        centers[i][1] + (directions[i][1] * strengths[i])]]
        line_2 = [[centers[i+1][0] - (directions[i+1][0] * strengths[i+1]),
                        centers[i+1][0] + (directions[i+1][0] * strengths[i+1])],
                       [centers[i+1][1] - (directions[i+1][1] * strengths[i+1]),
                        centers[i+1][1] + (directions[i+1][1] * strengths[i+1])]]

        #Assume length ratio of the trajectories are 1 in the world plane
        s = 1

        delta_x1 = line_1[1][0] - line_1[0][0]; delta_y1 = line_1[1][1] - line_1[0][1]
        delta_x2 = line_2[1][0] - line_2[0][0]; delta_y2 = line_2[1][1] - line_2[0][1]

        c_alpha, c_beta = ((delta_x1 * delta_y1 - pow(s,2) * delta_x2 * delta_y2)
                           / (pow(delta_y1,2) - pow((s * delta_y2),2)), 0)
        radius = np.abs(s * (delta_x2 * delta_y1 - delta_x1 * delta_y2) / (pow(delta_y1 , 2) - pow((s * delta_y2) , 2)))

        circle = Circle((c_alpha, c_beta), radius, color = 'b', fill=False)
        circles.append(circle)

        ax = plt.gca()
        ax.add_patch(circle)

    # Find the intersection of circles for each pair, then find their mean

    plt.axis('scaled')
    plt.show()



def compute_homography_and_warp(image, vp1, vp2, clip=True, clip_factor=3):
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

    # Find Projective Transform
    vanishing_line = np.cross(vp1, vp2)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2] #As h32 needs to be 1 in projection

    # Find directions corresponding to vanishing points
    v_post1 = np.dot(H, vp1) #The last element it 0 as it is the distance of the vanishing line to a vanishing point
    v_post2 = np.dot(H, vp2)
    v_post1 = v_post1 / np.sqrt(v_post1[0]**2 + v_post1[1]**2)
    v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)

    directions = np.array([[v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
                           [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]]])

    thetas = np.arctan2(directions[0], directions[1])

    # Find direction closest to horizontal axis
    h_ind = np.argmin(np.abs(thetas))

    # Find positve angle among the rest for the vertical axis
    if h_ind // 2 == 0:
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

    import skimage.transform as transform

    warped_img = transform.warp(image, np.linalg.inv(final_homography),
                                output_shape=(max_y, max_x))

    cv2.imshow("The warped result from google", warped_img)
    cv2.waitKey(0)


    return warped_img

class CameraParameterWriter:

    def __init__(self):
        self.writer = open("unityCamCalibration.txt","w+")
    def write(self, input_line):
        self.writer.write(input_line)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

def extractCameraParameters(image, model_points, image_points, K):

    h,w,_ = image.shape

    #For displaying the axes of the placed model
    model_normal = np.float64([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 20]])

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    dist_coef = np.zeros(4)

    #For writing these results to an output file
    camWriter = CameraParameterWriter()

    #From experiments, p3p seems like the best
    algorithms = { "iterative": cv2.SOLVEPNP_ITERATIVE, "p3p": cv2.SOLVEPNP_P3P, "epnp": cv2.SOLVEPNP_EPNP}

    for v,k in enumerate(algorithms):

        _ret, rvec, tvec = cv2.solvePnP(model_points, image_points, K, dist_coef, flags=v)

        (normal, _) = cv2.projectPoints(model_normal, rvec,
                                                        tvec,
                                                         K, dist_coef)

        cv2.line(image, tuple(map(int,normal[0][0])), tuple(map(int,normal[1][0])), (0, 0, 255), 2)
        cv2.line(image, tuple(map(int, normal[0][0])), tuple(map(int, normal[2][0])), (0, 255, 0), 2)
        cv2.line(image, tuple(map(int, normal[0][0])), tuple(map(int, normal[3][0])), (255, 0, 0), 2)

        rvec, _ = cv2.Rodrigues(rvec)

        print(k)
        print("Rotation {}".format(rvec))
        print("Translation {}".format(tvec))

        # Display image
        cv2.imshow("Output", image)
        cv2.waitKey(0)

        #Intrinsic Line
        camWriter.write("{} {} {} {} {} {}\n".format(w,h,K[0][2], K[1][2], K[0][0],K[1][1]))

        #Extrinsic Line
        tvec = [t[0] for t in tvec]
        camWriter.write("{} {} {} {} {} {} {} {} {} {} {} {}\n".format(*(rvec[:,0]),*(rvec[:,1]),*(rvec[:,2]),*tvec))





def rectify_groundPlane(image_OI, segmented_img_path, navigable_img_path, detection_data_file):

    # Manuel testing debugging part:
    image = cv2.imread(image_OI)
    segmented_img = cv2.imread(segmented_img_path)
    navigable_img = cv2.imread(navigable_img_path)

    cv2.imshow("Image to be corrected", image)
    cv2.imshow("The navigable area", navigable_img)
    cv2.waitKey(5)

    # Extract the lines from the navigable area only!
    image_lines = extract_image_lines(navigable_img)

    #Extract the pedestrian paths as lines (postures or bounding boxes)

    #Obtain the postures of the pedestrian as lines too, to find the VP
    pedestrian_posture_paths, pedestrian_postures = parse_pedestrian_detection(np.copy(image), detection_data_file, 50, use_bounding_boxes=False, returnPosture= True)
    pedestrian_posture_paths_single, _ = parse_pedestrian_detection(np.copy(image), detection_data_file, 10, use_bounding_boxes=False, tracker_id=56) # Parameter

    #If we assume the people doesn't change their velocities much
    #and calculate a homography between a path with multiple detections and
    pedestrian_bb_paths, _ = parse_pedestrian_detection(np.copy(image), detection_data_file, 10)
    trajectory_lines, _ = parse_pedestrian_detection(np.copy(image), detection_data_file, 5, False, True, tracker_id=56) # Parameter

    #Determine a and b for the affine component of the homography
    extract_circular_points(trajectory_lines)

    # - Postures as both feet and head trajectories (too sensitive to noise, not preferable)
    # - Single tracker posture (better than above)
    # - BB's head and feet postures again sensitive to noise
    # - Feet Trajectory only, RANSAC based
    # - Feet Trajectory + Hough Lines from the navigable area, RANSAC based
    # - Hough Lines from navigable area only

    vp_determination_methods = {
        'posture': [pedestrian_posture_paths, False],
        'single': [pedestrian_posture_paths_single, False],
        'bb': [pedestrian_bb_paths, False],
        'hough': [image_lines, True],
        'trajectory': [trajectory_lines, True],
        'trajectory_hough': [[np.concatenate((trajectory_lines[j], image_lines[j]), axis=0)
                              for j in range(3)] , True]
    }

    row = 3
    col = 2
    fig, axis = plt.subplots(row, col)

    height,width, _ = image.shape

    for i, k in enumerate(vp_determination_methods):
        lines = vp_determination_methods[k][0]

        plot_axis = axis[i//col,i%col]
        # plot_axis.figure()
        plot_axis.set_title(str(k))
        plot_axis.imshow(image)
        horizon = determineVP(lines, np.copy(image), plot_axis= plot_axis, asTrajectory=vp_determination_methods[k][1])

        leftVP, rightVP = horizon
        allignedImg, leftVP, rightVP = allignHorizon(image, horizon)
        img_wrapped_segmented = applyHomography(image, list(leftVP), list(rightVP), k)

        print("Method: {}, left: {}, right: {}".format(k,leftVP,rightVP))


        #TODO: Google's method - left for comparison
        # google_result = compute_homography_and_warp(image, list(leftVP), list(rightVP), clip=True, clip_factor=3)
        # cv2.imshow("Google's result with " + k, google_result)
        # cv2.waitKey(5)

        #Find the focal_unity length from the triangle of vanishing points

        pedestrian_postures = [np.concatenate((pedestrian_postures[j], image_lines[j]), axis=0)
         for j in range(3)]
        zenith_vp, focal_length = ransac_zenith_vp(pedestrian_postures, horizon, [width/2,height/2, 1])

        zenith_vp = zenith_vp / zenith_vp[2]
        plt.imshow(image)
        plt.plot((zenith_vp[0]), (zenith_vp[1]), 'bo')
        plt.show()

        #TODO: Output the internal and external parameters through a text file

        #Ezio Malis, Manuel Vargas, and others. Deeper understanding of the homography decomposition for vision-based control. 2007.




    plt.show()

    # # img_wrapped_segmented = applyHomography(allignedImg, leftVP, rightVP)
    # # cv2.imwrite("segmented_birdview.jpg", img_wrapped_segmented)
    #
    # img_wrapped = applyHomography(image, leftVP, rightVP)
    # # cv2.imwrite("birdview.jpg", img_wrapped)


# #TODO: Relook at that bin based papers.
# '''
# The edges needs to be clustered before they are used for finding vanishing points in the image
# Each edge preserves an angle that is used to identify which VP it would contribute its calculation to.
#
# For example: edges that have degrees between 0-60 and 180-240 will contribute to right VP where
# edges that have  60-120 and its reflection will contribute to zenith (3rd) VP.
#
# '''
# def cluster_edges(edges, angle_range = 60):
#
#     locations, directions, strengths = edges
#     clusters = [[],[],[]] #each for a VP
#
#     for p0, p1, i in enumerate(directions):
#
#         #Calculate the angle of the direction
#         ang = np.arctan(p1,p0)
#
#         clusters[(ang % 180) // angle_range].append(i)

if __name__ == "__main__":
    print("Welcome to the perspective corrector")

    aparser = argparse.ArgumentParser(description="Using the image perspective cues and pedestrian detection data"
                                                  "in order to rectify the ground plane to be used for navigation")
    aparser.add_argument("--image", help = "Image to be perspectively corrected")
    aparser.add_argument("--segmented", help = "Segmented version of the given image")
    aparser.add_argument("--detection", help = "Detection txt to be used")
    # aparser.add_argument("--navColor", help = "The colored image with navigable area only") TODO: UNUSED

    args = vars(aparser.parse_args())

    rectify_groundPlane(**args)