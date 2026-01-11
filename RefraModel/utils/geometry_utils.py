"""
Geometry utility functions for the application
"""
import numpy as np


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)