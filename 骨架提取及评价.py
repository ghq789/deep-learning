# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:53:50 2018
@author: Administrator
"""
import cv2 as cv
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.utils import image_utils
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keras.preprocessing import image

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

print('4')

def Xihua(image, array, num=20):
    h = image.shape[0]
    w = image.shape[1]
    iXihua = np.zeros((w, h, 1), dtype=np.uint8)
    np.copyto(iXihua, image)
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua

def Two(image):
    h = image.shape[0]
    w = image.shape[1]

    iTwo = np.zeros((w, h, 1), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            iTwo[i, j] = 0 if image[i, j] < 200 else 255
    return iTwo

print('1')

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
print('5')

# 加载图片并预处理
def load_and_process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor

# 提取特征
def extract_features(model, img_path):
    img_tensor = load_and_process_image(img_path)
    features = model.predict(img_tensor)
    # 将特征扁平化为一维数组
    flattened_features = features.flatten()
    return flattened_features


# 图片路径
img_path1 = r'E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png'
img_path2 = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\2.png'
img_path3 = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\3.png'


# 提取特征向量
vector1 = np.array(extract_features(model, img_path1)).reshape(1, 25088)
vector2 = np.array(extract_features(model, img_path3)).reshape(1, 25088)
# vector1 = np.array(extract_features(model, img_path1))
# vector2 = extract_features(model, img_path3)
normalized_distance = cosine_similarity(vector1, vector2)
print(f"The Euclidean distance between vector1 and vector2 is: {normalized_distance}")


image = cv.imread('E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png', 0)
iTwo = Two(image)
iThin = Xihua(iTwo, array)
# cv.imshow('image', image)
# cv.imshow('iTwo', iTwo)
cv.imshow('iThin', iThin)

image1 = cv.imread('E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\2.png', 0)
iTwo1 = Two(image1)
iThin1 = Xihua(iTwo1, array)
cv.imshow('iThin1', iThin1)


cv.imwrite('E:\\python_files\\pytorch\\myself\\gujiatiqu\\a.png',iThin)
cv.imwrite('E:\\python_files\\pytorch\\myself\\gujiatiqu\\b.png',iThin1)
cv.waitKey(0)
