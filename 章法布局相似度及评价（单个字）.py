import cv2
import numpy as np

def calculate_layout_similarity(char1, char2):
    # 转换为二值图像
    char1 = cv2.cvtColor(char1, cv2.COLOR_BGR2GRAY)
    char2 = cv2.cvtColor(char2, cv2.COLOR_BGR2GRAY)
    _, char1 = cv2.threshold(char1, 127, 255, cv2.THRESH_BINARY)
    _, char2 = cv2.threshold(char2, 127, 255, cv2.THRESH_BINARY)

    # 计算字宽和字高
    left_distance_char1 = float('inf')
    right_distance_char1 = float('-inf')
    top_distance_char1 = float('inf')
    bottom_distance_char1 = float('-inf')

    left_distance_char2 = float('inf')
    right_distance_char2 = float('-inf')
    top_distance_char2 = float('inf')
    bottom_distance_char2 = float('-inf')

    for i in range(char1.shape[0]):
        for j in range(char1.shape[1]):
            if char1[i, j] == 255:
                left_distance_char1 = min(left_distance_char1, j)
                right_distance_char1 = max(right_distance_char1, j)
                top_distance_char1 = min(top_distance_char1, i)
                bottom_distance_char1 = max(bottom_distance_char1, i)

    for i in range(char2.shape[0]):
        for j in range(char2.shape[1]):
            if char2[i, j] == 255:
                left_distance_char2 = min(left_distance_char2, j)
                right_distance_char2 = max(right_distance_char2, j)
                top_distance_char2 = min(top_distance_char2, i)
                bottom_distance_char2 = max(bottom_distance_char2, i)

    # 计算布局相似度
    width_similarity = 1 - abs(right_distance_char1 - right_distance_char2) / max(right_distance_char1, right_distance_char2)
    height_similarity = 1 - abs(bottom_distance_char1 - bottom_distance_char2) / max(bottom_distance_char1, bottom_distance_char2)
    layout_similarity = (width_similarity + height_similarity) / 2

    return layout_similarity

# 读取两个汉字图像
char1 = cv2.imread('E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png')
char2 = cv2.imread('E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\6.png')

# 计算布局相似度
similarity = calculate_layout_similarity(char1, char2)
print('布局相似度：', similarity)