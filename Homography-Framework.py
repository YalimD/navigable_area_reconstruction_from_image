import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import argparse

'''
Homography framework that applies perpective correction to given survailance image
by converting it to bird-view perspective as close as possible.

Works on a single image, camera is assumed to be not calibrated and focal length is not known.

Algorithm steps TODO: Erase after implementation
- Using RANSAC, find the best point that represents the vanishing points, according to
its reachabilty from other edges in the image
-TODO: We need to find the third vanishing point too, otherwise we would be assuming that 
we have a horizon that is perfectly horizontal. That is not the case for all images, the image
can be tilted.
- Instead of mindlessly taking a random set of all edgelets, we first cluster them according to
their angles with X axis.
- Consider: between 0-45 and 180-225 degrees, are considered to be contributing to right VP. 
- Best thing is to assign a single angle for the zenith VP, which divides the 180 degrees in half.

- Identify the vanishing line as the combination of left and right VP (manhattan world assumption) 

IF THE VANISHING LINE IS NOT PARALLEL AS WE HOPED FOR WHICH IT ISN't:
- ROTATE THE IMAGE WITH THE ANGLE THAT THE VANISHING LINE HAS WITH THE X AXIS
- CLIP THE IMAGE ACCORDINGLY
- PROCEED WITH THE HOMOGRAPHY 
- AFTER THE UNITY'S CAMERA IS PLACED AND ADJUSTED ACCORDING TO HOMOGRAPHY,
APPLY THE INVERSE OF THIS ROTATION
- THIS WAY, WE CAN EVEN CATCH THE REVERSED IMAGES (!)

- Apply the homography logic we have talked before (TODO: Explain later)

- Using the vanishing points, approximate the focal length, using triangle properties
https://www.coursera.org/learn/robotics-perception/lecture/jnaLs/how-to-compute-intrinsics-from-vanishing-points

author Yalım Doğan

Some ideas and code segments are from: https://github.com/chsasank/Image-Rectification (uses Scikit rather than OpenCV)
which is an implementation of Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe. "Auto-rectification of user photos." 
2014 IEEE International Conference on Image Processing (ICIP). IEEE, 2014.
'''


#Extracts the edges and hough lines from the image
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
    # cv2.waitKey(0)

    # Hough lines
    line_segments = cv2.HoughLinesP(grayscale_img, 1, np.pi / 180, threshold=100,minLineLength=30,
                                    maxLineGap=20)

    return lineProperties(line_segments,image)


#By parsing the detection data, it extracts the paths of each pedestrian for 1 sec each
def extract_pedestrian_lines(image, detection_data):

    latest_loc = {}
    paths = []

    frame_index  = 0
    fps = 30; resize = 1 #default
    seconds_per_check = 1 #Checks pedestrian data every x seconds

    with open(detection_data) as frames:
        for i,line in enumerate(frames):
            if i == 0:
                l = line.split()
                fps = int(l[-1])
                resize = int(l[-2])
            # Parse the pedestrian location
            elif i - 1 == frame_index * fps * seconds_per_check:
                frame_index += 1
                locations = line.split(",")[1:]

                # Every 7 values is an agent
                agents = [locations[i * 7:(i + 1) * 7] for i in range(len(locations) // 7)]

                for agent in agents:

                    #Current agent
                    headPos = (int(int(agent[1]) * resize),(float(agent[2]) - float(agent[6]) / 2) * resize)
                    feetPos = (int(int(agent[1]) * resize),(float(agent[2]) + float(agent[6]) / 2) * resize)

                    try:
                        #If the agent id is not found in the dictionary, this will raise KeyError
                        prev_pos = latest_loc[agent[0]]

                        head_path = [prev_pos[0], headPos]
                        feet_path = [prev_pos[1], feetPos]

                        #Detect and remove outliers.
                        #Outliers are inconsistent detection box sizes. For example, a detection box that
                        #shrinks while getting closer to the camera is considered as an outlier
                        currentHeight = feetPos[1] - headPos[1]
                        prev_height = prev_pos[1][1] - prev_pos[0][1]
                        size_increased = currentHeight > prev_height
                        higher_position = (headPos[1] + feetPos[1])/2 < (prev_pos[0][1] + prev_pos[1][1])/2

                        if size_increased ^ higher_position:
                            # The paths are held as pairs
                            paths.extend((head_path, feet_path))
                            latest_loc[agent[0]] = [headPos, feetPos]

                    except KeyError:
                        latest_loc[agent[0]] = [headPos, feetPos]


    frames.close()

    return lineProperties(paths, image)

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
        cv2.line(image, (x0, y0), (x1, y1), (0, 0, 255), 1)

    # Taken from https://github.com/chsasank/Image-Rectification
    # The direction vectors are normalized for easier calculation afterwards
    line_directions = np.array(line_directions) / np.linalg.norm(line_directions, axis=1)[:, np.newaxis]

    # cv2.imshow("Extracted Lines", image)
    # cv2.waitKey(0)

    return (line_centers, line_directions, line_strengths)

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
    normals = np.zeros_like(directions) #Why not just np.zeros(directions.shape)
    #normals = np.zeros(directions.shape)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0] #as y is negative
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines



#Determines the vanishing points on horizon using information coming from pedestrian paths
def determineVPfromPaths(path_lines,image):

    centers, directions, strengths = path_lines
    normals = edgelet_lines(path_lines)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    vxs = []
    vys= []

    for i in range(len(normals)//2):
        # plt.clf()
        # plt.imshow(image)
        plt.plot([centers[2*i][0] - (directions[2*i][0] * strengths[2*i]),
                  centers[2 * i][0] + (directions[2 * i][0] * strengths[2 * i])],
                 [centers[2 * i][1] - (directions[2 * i][1] * strengths[2 * i]),
                  centers[2 * i][1] + (directions[2 * i][1] * strengths[2 * i])], 'r-')

        plt.plot([centers[2*i+1][0] - (directions[2*i+1][0] * strengths[2*i+1]),
                  centers[2 * i+1][0] + (directions[2 * i+1][0] * strengths[2 * i+1])],
                 [centers[2 * i+1][1] - (directions[2 * i+1][1] * strengths[2 * i+1]),
                  centers[2 * i+1][1] + (directions[2 * i+1][1] * strengths[2 * i+1])], 'r-')

        head = normals[2*i]
        feet = normals[2*i+1]

        vx,vy,n = np.cross(head,feet); vx /= n; vy /= n
        if math.isfinite(vx) and math.isfinite(vy):
            vxs.append(vx)
            vys.append(vy)
        plt.plot([vx],[vy],'bo')


    #Use RANSAC to determine the vanishing line

    #TODO: Which top percentige of the points needs to be considered in order to
    #get a good result on vanishing point
    ransac_ratio = 0.3

    sorted_ind = np.argsort(vys)
    sorted_ind = sorted_ind[:int(len(sorted_ind) * ransac_ratio)]

    vxs = np.array(vxs).reshape(-1, 1)[sorted_ind]
    vys = np.array(vys).reshape(-1, 1)[sorted_ind]

    ransac = linear_model.RANSACRegressor()
    ransac.fit(vxs,vys)
    line_X = np.arange(vxs.min(), vxs.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    plt.plot(line_X, line_y_ransac, color='g')

    plt.show()

    vp_left = (line_X[0][0],line_y_ransac[0][0])
    vp_right = (line_X[-1][0],line_y_ransac[-1][0])
    return [vp_left, vp_right]


#This function rotates the image according to the angle of the horizon in order to allign the horizon
#"horizontally"
def allignHorizon(image, horizon):

    angle = np.arctan((horizon[1][1] - horizon[0][1]) / (horizon[1][0] - horizon[0][0]))
    angle *= (180 / np.pi)
    print("Original angle of the horizon: {}".format(angle))

    row,col,_ = image.shape

    #Rotation without cropping (Thanks to Adrian Rosebrock for explanation)
    centerx, centery = col // 2,row // 2

    rot = cv2.getRotationMatrix2D((centerx, centery), angle, 1)

    cos = np.abs(rot[0,0])
    sin = np.abs(rot[0,1])

    newwidth = int((row * sin) + (col * cos))
    newheight = int((row * cos) + (col * sin))

    #Also obtain the resulting y coordinate of the horizon

    #Rotate the end point according to found angle
    p = list(horizon[0]) + [1]
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

    dst = cv2.warpAffine(image, rot, (newwidth, newheight))

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))

    plt.plot([leftP[0],rightP[0]], [leftP[1],rightP[1]], 'bo')
    plt.show()

    print("Left {}".format(leftP))
    print("Right {}".format(rightP))

    return dst, leftP, rightP


#Apply the homography to the image according to the points found on
#lines that meet at the horizon
def applyHomography(allignedImg, leftVP, rightVP):

    height,width,_ = allignedImg.shape
    leftCorner = [0,height]
    rightCorner = [width, height]

    homographyPlane = np.array(
    [
        leftCorner,
        leftVP,
        rightVP,
        rightCorner
    ], np.float)

    resize = 0.40  #Resize the resulting image
    resize = 1 - resize

    mappedPos = np.array([
        [width * resize / 2, height * (1 - resize / 2)],
        [width * resize / 2, height * resize / 2],
        [width * (1 - resize / 2), height * resize / 2],
        [width * (1 - resize / 2), height * (1 - resize / 2)]
    ], np.float)


    #TODO: Decompose this homography
    homo, status = cv2.findHomography(homographyPlane, mappedPos);
    img_wrapped_segmented = cv2.warpPerspective(allignedImg, homo,
                                                (width,
                                                 height));

    cv2.imshow("The wrapped result",img_wrapped_segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_wrapped_segmented


def extract_pedestrian_majors(video):

    #Create background subtractor
    #Process the frames until it gets a good result on background
    #Then get the frame for each second
    #Detect blobs and their major axis each. We did this in the shadow part too
    #Adjust the strength of the major axis as long as the blob
    #Write the lines and test

    capture = cv2.VideoCapture(video)
    if not capture.isOpened:
        print ("Video file: {} is not found".format(video))

    # backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(200,32,detectShadows=True)
    #CNT is too noisy but can be cleared using erosion
    #GSOC has a lot of initial noise, but gets better in time
    #LSBP has a lot of noise
    backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    cv2.namedWindow("Processed Video")
    cv2.namedWindow("Original Video")

    while capture.isOpened:
        ret, frame = capture.read()

        if not ret:
            capture.release()
            capture.open(video)
            continue

        foreground_mask = backgroundSubtractor.apply(frame,0.9)

        #Shadows aren't really detected therefore not removed here
        print(len(np.where(foreground_mask == 127)))
        ret, foreground_mask = cv2.threshold(foreground_mask,128,255,cv2.THRESH_BINARY)

        cv2.imshow("Processed Video",foreground_mask)
        cv2.imshow("Original Video", frame)
        cv2.waitKey(30)



#TODO: UI adjustments are necessary
#TODO: RANSAC can be improved and some elements can be corrected
#TODO: Redundant code exists
#TODO: Find K using pedestrian postures and lines in the image
#TODO: Also include the effect of the lines in the image when finding the horizon
if __name__ == "__main__":


    extract_pedestrian_majors("D:\VideoRVO\Data\BaseVideos\RVOVideos\P1070088.mp4")


    print("Welcome to the perspective corrector")
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--image", help = "Image to be perspectively corrected", default="test_shadowClear.jpg")
    aparser.add_argument("--segmented", help = "Segmented version of the given image", default = "segmented.png")
    aparser.add_argument("--detection", help = "Detection txt to be used", default = "output.txt")
    # aparser.add_argument("--verbose", help = "Display certain elements for debugging",
    #                      action="store_true")

    args = vars(aparser.parse_args())

    image_OI = args["image"]
    segmented_img_path = args["segmented"]
    detection_data_file = args["detection"]

    # Manuel testing debugging part:
    image = cv2.imread(image_OI)
    segmented_img = cv2.imread(segmented_img_path)

    cv2.imshow("Image to be corrected", image)
    #cv2.waitKey(0)

    # Extract the lines from the image.
    # TODO: UNUSED FOR NOW, WE MAY TRY COMBINING IT WITH LINES WE FOUND FROM PEDESTRIAN PATHS
    #image_lines = extract_image_lines(image)

    #Extract the pedestrian paths as lines
    path_lines = extract_pedestrian_lines(np.copy(image), detection_data_file)
    horizon = determineVPfromPaths(path_lines,np.copy(image)) #As a line; two endpoints
    print("Horizon found: {}".format(horizon))

    #Rotate the image according to the found horizon, sot he horizon is alligned horizontally
    allignedImg, leftVP, rightVP = allignHorizon(segmented_img, horizon)

    #Generate the polygon representing the ground plane
    img_wrapped_segmented = applyHomography(allignedImg,leftVP, rightVP)
    cv2.imwrite("segmented_birdview.jpg",img_wrapped_segmented)

    #TODO: Since the resulting image is small, we need to zoom in using
    #bounding box size of the contour

    #Extract major axes of pedestrians from given background subtracted images
    #WIP using tensorflow
    #pedestrian_majors = extract_pedestrian_majors(video)

    # Cluster the edges
    #edge_clusters = cluster_edges(edges)

    # Find Vanishing points

    # Rectify image

    # Show result


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
