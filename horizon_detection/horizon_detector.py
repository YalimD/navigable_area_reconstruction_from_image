import cv2
import numpy as np
import math
import matplotlib.lines as lines

from sklearn import linear_model
from skimage import transform, feature

__all__ = ["HorizonDetectorLib"]

class HorizonDetectorLib:

    # Extracts the edges and hough lines from the image
    # Taken from IMAGE RECTIFICATION
    @staticmethod
    def extract_image_lines(img):
        image = np.copy(img)

        # Bilateral filtering which keeps the edges sharp, but textures blurry
        # seems to decrease the noisy edges that cause too many detection results
        # Read bilateral filters: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
        # Everyone is affected by similar and close pixels. If neighbour is not similar, then its effect is small
        # Makes things more "comical"
        image = cv2.bilateralFilter(image, 9, 60, 60)

        # The image needs to be converted to grayscale first
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = feature.canny(grayscale_img, 3)
        # Parameter: Parameter
        line_segments = transform.probabilistic_hough_line(edges, line_length=100,
                                                   line_gap=30)

        # Edge detection (canny for now)
        # grayscale_img = cv2.Canny(grayscale_img, threshold1=75, threshold2=200, apertureSize=3)
        # grayscale_img = cv2.Sobel(grayscale_img,cv2.CV_8UC1,1,1)
        # grayscale_img = cv2.Laplacian(grayscale_img,cv2.CV_8UC1)

        # cv2.imshow("Detected Edges", grayscale_img)
        # cv2.waitKey(5)

        # Hough lines (OpenCV version)
        # line_segments = cv2.HoughLinesP(grayscale_img, 1, np.pi / 180, threshold=100, minLineLength=30,
        #                                 maxLineGap=30)


        return HorizonDetectorLib.lineProperties(line_segments, image)


    # By parsing the detection data, it extracts the paths of each pedestrian for every x frames
    # The head-feet positions depend on the selected method; using bounding boxes or pedestrian postures
    # Note: Pedestrian postures are more prone to noise
    # In addition, Using the pedestrian postures, determine the orthogonal vanishing point using RANSAC for minimizing
    # distance between candidate line pairs
    @staticmethod
    def parse_pedestrian_detection(image, detection_data, frames_per_check=30, use_bounding_boxes=True, feet_only=False,
                                   tracker_id=None, returnPosture=False):
        latest_loc = {}
        paths = []
        postures = []

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


                # if num_of_frames % frames_per_check == 0 and (tracker_id is None or (tracker_id is not None and int(agent[1]) in tracker_id)):
                if (agent[1] not in latest_loc.keys() or num_of_frames - (latest_loc[agent[1]])[2] >= frames_per_check) \
                        and (tracker_id is None or (int(agent[1]) in tracker_id)):


                    # Different methods for extracting head and feet
                    if not use_bounding_boxes:
                        # Add the agent id to the dictionary latest_loc
                        headPos = list(map(float, agent[-2].split('/')))
                        feetPos = list(map(float, agent[-1].split('/')))
                    else:
                        pos = list(map(float, agent[2:6]))
                        headPos = [pos[0] + pos[2] / 2, pos[1]]
                        feetPos = [pos[0] + pos[2] / 2, pos[1] + pos[3]]

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
                            if not feet_only:
                                paths.append(head_path)

                            if returnPosture:
                                postures.append([headPos, feetPos])
                            latest_loc[agent[1]] = [headPos, feetPos, num_of_frames]

                    except:
                        latest_loc[agent[1]] = [headPos, feetPos, num_of_frames]


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

        return HorizonDetectorLib.lineProperties(paths, image), \
               HorizonDetectorLib.lineProperties(postures, image)


    # Extracts the line properties from given line segments
    # Returns centers, directions and strengths
    @staticmethod
    def lineProperties(lines, image):
        line_directions = []
        line_centers = []
        line_strengths = []

        for line in lines:
            line = np.array(line).flatten()  # Compatibility

            x0 = int(line[0]);
            y0 = int(line[1])
            x1 = int(line[2]);
            y1 = int(line[3])

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


    # Taken from that github page
    @staticmethod
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
        normals[:, 1] = -directions[:, 0]  # as y is negative
        p = -np.sum(locations * normals, axis=1)
        lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
        return lines


    # Determines the vanishing points on horizon using information coming from pedestrian paths
    # OR uses the trajectory information and/or edges from the image to detect vanishing points
    # which will determine the horizon
    # TODO: Parameters for thresholds
    @staticmethod
    def determineVP(path_lines, image, plot_axis, asTrajectory=False, ground_truth = None, draw_features = False):
        centers, directions, strengths = path_lines

        # Draw the path lines
        if draw_features:
            for i in range(centers.shape[0]):
                plot_axis.plot([centers[i][0] - (directions[i][0] * strengths[i]),
                                centers[i][0] + (directions[i][0] * strengths[i])],
                               [centers[i][1] - (directions[i][1] * strengths[i]),
                                centers[i][1] + (directions[i][1] * strengths[i])], 'r-')

        vxs = []
        vys = []

        if not asTrajectory:

            normals = HorizonDetectorLib.edgelet_lines(path_lines)

            for i in range(len(normals) // 2):

                head = normals[2 * i]
                feet = normals[2 * i + 1]

                vx, vy, n = np.cross(head, feet);
                vx /= n
                vy /= n
                if math.isfinite(vx) and math.isfinite(vy):
                    vxs.append(vx)
                    vys.append(vy)
                    plot_axis.plot([vx], [vy], 'bo')

            # Use RANSAC to determine the vanishing line

            sorted_ind = np.argsort(vys)
            sorted_ind = sorted_ind[:int(len(sorted_ind))]

            vxs = np.array(vxs).reshape(-1, 1)[sorted_ind]
            vys = np.array(vys).reshape(-1, 1)[sorted_ind]

            ransac = linear_model.RANSACRegressor()
            ransac.fit(vxs, vys)
            line_X = np.arange(vxs.min(), vxs.max())[:, np.newaxis]
            line_Y = ransac.predict(line_X)

            vp_left = (line_X[0][0], line_Y[0][0], 1)
            vp_right = (line_X[-1][0], line_Y[-1][0], 1)
            vp_zenith = None #We find the zenith vanishing point elsewhere, but I think the method below is better

        else:

            # Using RANSAC method on trajectories
            model = HorizonDetectorLib.ransac_vanishing_point(path_lines)
            vp1 = model / model[2]
            plot_axis.plot(vp1[0], vp1[1], 'bo')

            # Before determining the second VP, remove inliers as they already contributed to first VP
            path_lines_reduced = HorizonDetectorLib.remove_inliers(vp1, path_lines, 30)

            # Find second vanishing point
            model2 = HorizonDetectorLib.ransac_vanishing_point(path_lines_reduced)
            vp2 = model2 / model2[2]
            plot_axis.plot(vp2[0], vp2[1], 'bo')

            # Test if we can find the zenith vanishing point
            # Before determining the second VP, remove inliers as they already contributed to first VP
            path_lines_reduced_again = HorizonDetectorLib.remove_inliers(vp2, path_lines_reduced, 60)

            # Find second vanishing point
            model3 = HorizonDetectorLib.ransac_vanishing_point(path_lines_reduced_again)
            vp3 = model3 / model3[2]
            plot_axis.plot(vp3[0], vp3[1], 'bo')

            # # Parameter: Only use for debugging, overcrowds the image if used
            #
            # for i in range(centers.shape[0]):
            #     xax = [centers[i, 0], vp1[0]]
            #     yax = [centers[i, 1], vp1[1]]
            #     plot_axis.plot(xax, yax, 'b-.')
            #
            #     xax = [centers[i, 0], vp2[0]]
            #     yax = [centers[i, 1], vp2[1]]
            #     plot_axis.plot(xax, yax, 'b-.')

            # The vanishing point with highest y value is taken as the zenith 8as we are looking at the world birdview)
            vanishers = [vp1,vp2,vp3]
            vanishers.sort(key=lambda v: v[1])
            horizon_points = vanishers[:2]
            horizon_points.sort(key=lambda v:v[0])

            vp_left, vp_right, vp_zenith = horizon_points[0], horizon_points[1], vanishers[2]

            line_X = np.arange(vp_left[0], vp_right[0])[:, np.newaxis]
            horizon_line = np.cross(vp_left,vp_right)
            horizon_line = horizon_line / horizon_line[2]

            # line_X = [[vp_left[0], vp_right[0]]]
            line_Y = list(map(lambda point: (-horizon_line[0] * point[0] - horizon_line[2]) / horizon_line[1], line_X))

        plot_axis.plot(line_X, line_Y, color='r')

        if ground_truth:
            line_Y = list(map(lambda point: (-ground_truth[0] * point[0] - ground_truth[2]) / ground_truth[1], line_X))
            plot_axis.plot(line_X, line_Y, color='g')

        return [vp_left, vp_right, vp_zenith]


    # Problem: The zenith vanishing point appears at the same side of the vanishing line against center of the image, which is impossible
    # Solution: Find the distance to both and if distance to center is longer than the distance to horizon, ignore model
    # Needs: center point
    @staticmethod
    def ransac_zenith_vp(edgelets, horizon, image_center, num_ransac_iter=2000, threshold_inlier=5, zenith = None):

        # If no zenith vp is given, calculate it from postures and image lines

        locations, directions, strengths = edgelets
        lines = HorizonDetectorLib.edgelet_lines(edgelets)

        num_pts = strengths.size

        arg_sort = np.argsort(-strengths)
        # If number of points is lower than a threshold, don't create index spaces to compare models
        first_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 5]  # Top 25 percentile
        second_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 2]  # Top 50 percentile

        best_model = zenith
        best_votes = np.zeros(num_pts)

        # Normalize the horizon
        horizon_homogenous = np.cross(horizon[0], horizon[1])
        horizon_homogenous = horizon_homogenous / np.sqrt(horizon_homogenous[0] ** 2 + horizon_homogenous[1] ** 2)

        if best_model is None:
            for ransac_iter in range(num_ransac_iter):
                ind1 = np.random.choice(first_index_space)
                ind2 = np.random.choice(second_index_space)

                while ind2 == ind1:  # Protection against low line count
                    ind2 = np.random.choice(second_index_space)

                l1 = lines[ind1]
                l2 = lines[ind2]

                current_model = np.cross(l1, l2)  # Potential vanishing point
                current_model = current_model / current_model[2]

                # Its distance to center and the horizon
                horizon_distance = np.abs(np.dot(current_model.T, horizon_homogenous))
                centre_distance = np.linalg.norm(current_model[:2] - image_center[:2])

                if np.sum(current_model ** 2) < 1 or current_model[2] == 0 or horizon_distance < centre_distance:
                    # reject degenerate candidates, which lie on the wrong side of the horizon
                    continue

                current_votes = HorizonDetectorLib.compute_votes(
                    edgelets, current_model, threshold_inlier)

                if current_votes.sum() > best_votes.sum():
                    best_model = current_model
                    best_votes = current_votes
                    # logging.info("Current best model has {} votes at iteration {}".format(
                    #     current_votes.sum(), ransac_iter))

        center_vp_dist = (np.linalg.norm(best_model[:2] - image_center[:2]))
        center_hor_dist = np.dot(np.array(image_center), horizon_homogenous)
        focal_length = np.sqrt(center_vp_dist * center_hor_dist)

        return best_model, focal_length

    @staticmethod
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
        lines = HorizonDetectorLib.edgelet_lines(edgelets)

        num_pts = strengths.size

        arg_sort = np.argsort(-strengths)
        first_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 5]  # Top 20 percentile
        second_index_space = arg_sort if num_pts < 20 else arg_sort[:num_pts // 2]  # Top 50 percentile

        best_model = None
        best_votes = np.zeros(num_pts)

        for ransac_iter in range(num_ransac_iter):
            ind1 = np.random.choice(first_index_space)
            ind2 = np.random.choice(second_index_space)

            while ind2 == ind1:  # Protection against low line count
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

    @staticmethod
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
        inliers = HorizonDetectorLib.compute_votes(edgelets, model, threshold_inlier) > 0
        locations, directions, strengths = edgelets
        locations = locations[~inliers]
        directions = directions[~inliers]
        strengths = strengths[~inliers]
        edgelets = (locations, directions, strengths)
        return edgelets
