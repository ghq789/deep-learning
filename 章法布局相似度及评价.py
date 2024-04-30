import cv2
import numpy as np

def calculate_similar_layout(image1_path, image2_path):
    # 读取两个图像
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # 将图像转换为二值图像
    _, img1_binary = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img2_binary = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算图像中的轮廓
    contours1, _ = cv2.findContours(img1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算每个字的自宽、自高
    box_widths1 = []
    box_heights1 = []
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        box_widths1.append(w)
        box_heights1.append(h)

    box_widths2 = []
    box_heights2 = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        box_widths2.append(w)
        box_heights2.append(h)

    # 计算每个字的布局特征
    layout_feature1 = []
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x
        right_distance = 399 - (x + w)
        top_distance = y
        bottom_distance = 399 - (y + h)
        layout_feature1.append([left_distance, right_distance, top_distance, bottom_distance])

    layout_feature2 = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x
        right_distance = 399 - (x + w)
        top_distance = y
        bottom_distance = 399 - (y + h)
        layout_feature2.append([left_distance, right_distance, top_distance, bottom_distance])

    # 计算布局相似度
    layout_similarities = []
    for i in range(len(layout_feature1)):
        for j in range(len(layout_feature2)):
            distance = np.sqrt(np.sum(np.square(np.array(layout_feature1[i]) - np.array(layout_feature2[j]))))
            similarity = 1 / (1 + distance)
            layout_similarities.append(similarity)

    return np.mean(layout_similarities)
#
# # 图片路径
# img_path1 = r'E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png'
# img_path2 = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\3.png'
#
# # 调用函数计算布局相似度
# similar_layout_score = calculate_similar_layout(img_path1, img_path2)
# print("布局相似度得分为：", similar_layout_score)