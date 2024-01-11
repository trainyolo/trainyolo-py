import numpy as np

def rotate_around_point(point, angle, origin):
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Translation to the origin
    translated = point - origin

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Perform rotation and translate back
    rotated = np.dot(rotation_matrix, translated) + origin
    return rotated

def get_corners_from_one_corner(x0, y0, w, h, rotation):
    # Calculate other three corners relative to x0, y0
    corners = np.array([[x0, y0], [x0 + w, y0], [x0, y0 + h], [x0 + w, y0 + h]])

    # Rotate the corners around x0, y0
    rotated_corners = []
    for corner in corners:
        rotated = rotate_around_point(corner, rotation, np.array([x0, y0]))
        rotated_corners.append(rotated)

    return rotated_corners