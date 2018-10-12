import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#Read the image and display it

#On the image, let user select 8 points (2 for each head and feet)

#Combine the lines and determine the horizon. Output the angle and the two points

windowName = "Click on 8 points to determine the horizon in the image"

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(data['points']) < 8:
            data['points'].append([x, y, 1])
            cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
            cv2.imshow(windowName, data['im'])

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
    print("Vanishing line {}".format(vanishing))

    cv2.circle(data['im'], (point1[0], point1[1]), 3, (0, 255, 0), 5, 16)
    cv2.circle(data['im'], (point2[0], point2[1]), 3, (0, 255, 0), 5, 16)

    cv2.line(data['im'], (point1[0], point1[1]), (point2[0], point2[1]), (255, 0, 0), 5, 16)
    cv2.imshow(windowName, data['im'])
    cv2.waitKey(0)


if __name__ == "__main__":

    filename = "pet/pet.png"
    image = cv2.imread(filename)

    data = {}
    data['im'] = image.copy()
    data['points'] = []

    cv2.imshow(windowName, image)
    cv2.setMouseCallback(windowName, mouse_handler, data)
    cv2.waitKey(0)

    processPoints(data)






