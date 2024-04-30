#主要是处理带背景的书写汉字，去除噪音
import cv2
import numpy as np

# 读取图像
img = cv2.imread('E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png')

def image_preprocessing(image):
        # 将图像转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 通过阈值处理将汉字背景变为白色
        ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)  # 大于阈值时置 0(黑)，否则置 255（白）
        # 反转二值化图像，将汉字变为黑色，背景变为白色
        thresh = cv2.bitwise_not(thresh)
        # 执行形态学操作去除噪点
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return opening

# pre_image = image_preprocessing(img)
# 显示结果
# cv2.imshow('original', img)
# cv2.imshow('Result', pre_image)

# cv2.imwrite('E:\\python_files\\pytorch\\myself\\Template_pictures\\1p.png',pre_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()