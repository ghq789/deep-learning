# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:53:50 2018
@author: Administrator
"""
import cv2 as cv
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions # type: ignore
from keras.utils import image_utils# type: ignore
import numpy as np
from keras.preprocessing import image# type: ignore
from sklearn.metrics.pairwise import cosine_similarity

model = VGG16(weights='imagenet', include_top=False)

def VThin(image, array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def HThin(image, array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for j in range(h):
        for i in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def Xihua(image, array, num=20):
    h = image.shape[0]
    w = image.shape[1]
    iXihua = np.zeros((w, h, 1), dtype=np.uint8)
    np.copyto(iXihua, image)
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua


#函数接受一个图像（假设它是一个 NumPy数组）作为输入，并返回一个二值化（binary）的图像
def Two(image):
    h = image.shape[0]
    w = image.shape[1]

    iTwo = np.zeros((w, h, 1), dtype=np.uint8)   #iTwo是一个单通道的二维图像（灰度图像）
    for i in range(h):
        for j in range(w):
            iTwo[i, j] = 0 if image[i, j] < 200 else 255
    return iTwo

array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]


#对汉字进行细化
def Image_processing(image):
     iOne = Two(image)
     Image = Xihua(iOne, array)
     return Image

# 加载图片并预处理
def load_and_process_image_array(img):
    img_array =  image.img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor

# 提取特征
def extract_features(model, img):
    img_tensor = load_and_process_image_array(img)
    img_tensor = np.tile(img_tensor, (1, 1, 1, 3))
    img_tensor = img_tensor[:, :, :, :3]
    features = model.predict(img_tensor)
    return features[0]
# 图片路径
#需要对图片调用图片预处理模块才可以
img1 = cv.imread('E:\\python_files\\pytorch\\myself\\gujiatiqu\\1.png')
img2 = cv.imread('E:\\python_files\\pytorch\\myself\\gujiatiqu\\2.png')

image1 = Image_processing(img1)
image2 = Image_processing(img2)

# # 提取特征向量,利用余弦进行相似度计算
vector1 = extract_features(model, image1)
vector2 = extract_features(model, image2)

normalized_distance = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
print(f"The Euclidean distance between vector1 and vector2 is: {normalized_distance}")
