import cv2
import numpy as np
import os


def draw_matches(img1, img2, keypoints1, keypoints2, save_path, label,
                 point_radius=10, line_thickness=2):
    """
    Draw matched keypoints on two images.

    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.
        keypoints1 (list): A list of [x, y] lists of keypoints in the first image.
        keypoints2 (list): A list of [x, y] lists of keypoints in the second image.
    """
    # Assert that the lists of keypoints are of the same size
    assert len(keypoints1) == len(keypoints2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Create a new image by appending the two images side by side
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    new_h = max(h1, h2)
    new_w = w1 + w2
    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_img[:h1, :w1, :] = img1
    new_img[:h2, w1:w1 + w2, :] = img2

    # Define colors for keypoints and lines (B, G, R)
    color_keypoints1 = (0, 255, 0)  # Green for the first image
    color_keypoints2 = (255, 0, 0)  # Blue for the second image
    color_lines = (0, 0, 255)  # Red for lines

    # Draw keypoints and lines
    for kp1, kp2 in zip(keypoints1, keypoints2):
        x1, y1 = kp1
        x2, y2 = kp2

        # Convert coordinates to integer
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # Draw keypoints
        cv2.circle(new_img, (x1, y1), point_radius, color_keypoints1, -1)
        cv2.circle(new_img, (x2 + w1, y2), point_radius, color_keypoints2, -1)

        # Draw lines
        cv2.line(new_img, (x1, y1), (x2 + w1, y2), color_lines, line_thickness)

    # Display the image
    cv2.imwrite(save_path + label + '_match.jpg', new_img)
