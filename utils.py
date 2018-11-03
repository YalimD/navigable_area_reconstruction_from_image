import numpy as np
import math

# Finds the normalized cross product of two elements in homogenous coordinates
def normalizedCross(elem1, elem2):

    c = np.cross(elem1, elem2)
    return c / c[2]

def focalToFOV(focal, height):

    return math.degrees(2 * math.atan2(height , (2 * focal)))
