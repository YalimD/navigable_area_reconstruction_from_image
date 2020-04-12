import math

import numpy as np


# Finds the normalized cross product of two elements in homogenous coordinates
def normalized_cross(elem1, elem2):
    c = np.cross(elem1, elem2)
    return c / c[2]


def focal_to_fov(focal, height):
    return math.degrees(2 * math.atan2(height, (2 * focal)))
