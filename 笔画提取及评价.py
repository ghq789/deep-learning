import cv2
import numpy as np


def calculate_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + np.finfo(float).eps)
    return hu_moments.flatten()


def extract_grid_image(image, position):
    height, width = image.shape[:2]
    grid_height = height // 3
    grid_width = width // 3
    x, y = position
    start_x = x * grid_width
    start_y = y * grid_height
    end_x = start_x + grid_width
    end_y = start_y + grid_height
    end_x = min(end_x, width)
    end_y = min(end_y, height)
    grid_image = image[start_y:end_y, start_x:end_x]
    return grid_image


def calculate_weighted_sum(handwriting_image_path, template_image_path):
    handwriting_image = cv2.imread(handwriting_image_path)
    template_image = cv2.imread(template_image_path)

    handwriting_hu_moments = calculate_hu_moments(handwriting_image)
    template_hu_moments = calculate_hu_moments(template_image)

    grid_positions = [(0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2),
                      (2, 0), (2, 1), (2, 2)]

    grid_weights = [0.1, 0.1, 0.1,
                    0.1, 0.5, 0.1,
                    0.1, 0.1, 0.1]

    handwriting_hu_moments_list = []
    template_hu_moments_list = []

    for position in grid_positions:
        handwriting_grid_image = extract_grid_image(handwriting_image, position)
        template_grid_image = extract_grid_image(template_image, position)

        handwriting_hu_moments = calculate_hu_moments(handwriting_grid_image)
        template_hu_moments = calculate_hu_moments(template_grid_image)

        handwriting_hu_moments_list.append(handwriting_hu_moments)
        template_hu_moments_list.append(template_hu_moments)

    correlation_coefficients = []
    for handwriting_hu_moments, template_hu_moments in zip(handwriting_hu_moments_list, template_hu_moments_list):
        correlation_coefficient = np.corrcoef(handwriting_hu_moments, template_hu_moments)[0, 1]
        correlation_coefficients.append(correlation_coefficient)

    weighted_sum = np.dot(correlation_coefficients, grid_weights)
    weighted_sum = weighted_sum*76.923
    return weighted_sum


# # Example usage:
# handwriting_image_path = 'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\3.png'
# template_image_path = 'E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png'
# weighted_sum = calculate_weighted_sum(handwriting_image_path, template_image_path)
# print("笔画相似度:", weighted_sum*76.923)
