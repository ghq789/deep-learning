import 手写汉字图片预处理
import 骨架提取
import 骨架相似度评价
import 笔画提取及评价
import 章法布局相似度及评价
import cv2 as cv
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions # type: ignore
from keras.utils import image_utils# type: ignore
import numpy as np
from keras.preprocessing import image# type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing import image

#模板字和手写字读取
template_img_path  = r'E:\\python_files\\pytorch\\myself\\Template_pictures\\1.png'
handwriting_img = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\17.png'

#调用手写图片和模板图片进行预处理（去除背景，提取字）
img = cv.imread(handwriting_img)
pre_image = 手写汉字图片预处理.image_preprocessing(img)
cv.imwrite('E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\wang_copied.png',pre_image)

template_img =  cv.imread(template_img_path)
pretemplate_img  = 手写汉字图片预处理.image_preprocessing(template_img)
cv.imwrite('E:\\python_files\\pytorch\\myself\\Template_pictures\\wang_pretemplate.png',pretemplate_img)

#调用笔画提取及相似度评价
weighted_sum = 笔画提取及评价.calculate_weighted_sum('E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\wang_copied.png', 'E:\\python_files\\pytorch\\myself\\Template_pictures\\wang_pretemplate.png')
print("笔画相似度 weighted_sum:", weighted_sum)
# if(weighted_sum)

#调用章法布局相似度及评价
similar_layout_score = 章法布局相似度及评价.calculate_similar_layout('E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\wang_copied.png', 'E:\\python_files\\pytorch\\myself\\Template_pictures\\wang_pretemplate.png')
similar_layout_score = similar_layout_score * 100
print("布局相似度  similar_layout_score:", similar_layout_score)


#调用2对预处理手写图片进行骨架提取
model = VGG16(weights='imagenet', include_top=False)
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
filename = 'Handcopied_wang.png'
Handcopied_wang = 骨架提取.Image_processing(pre_image, filename)

#对模板字进行骨架提取
filename = 'template_wang.png'
template_wang = 骨架提取.Image_processing(pretemplate_img, filename)


#调用骨架相似度评价
vector1 = np.array(骨架相似度评价.extract_features(model, r'E:\\python_files\\pytorch\\myself\\gujiatiqu\\Handcopied_wang.png')).reshape(1, 25088)
vector2 = np.array(骨架相似度评价.extract_features(model, r'E:\\python_files\\pytorch\\myself\\gujiatiqu\\template_wang.png')).reshape(1, 25088)
normalized_distance = 100 * cosine_similarity(vector1, vector2)
print(f"骨架相似度  normalized_distance:" , normalized_distance)


# 笔画相似度评语
def evaluate_stroke_similarity(score):
    if score < 20:
        return '笔画极其不相似，与模板差异极大'
    elif score < 40:
        return '笔画很不相似，需要大幅度改进'
    elif score < 60:
        return '笔画不太相似，部分偏离模板'
    elif score < 80:
        return '笔画位置基本相似，但仍有改进空间'
    else:
        return '笔画位置非常相似，与模板高度一致'

    # 布局相似度评语
def evaluate_layout_similarity(score):
    if score < 20:
        return '布局极其混乱，与模板差异极大'
    elif score < 40:
        return '布局很不规整，需要大幅度调整'
    elif score < 60:
        return '布局不太规整，部分偏离模板'
    elif score < 80:
        return '布局基本规整，但仍有改进空间'
    else:
        return '布局非常规整，与模板高度一致'

    # 骨架相似度评语
def evaluate_skeleton_similarity(score):
    if score < 20:
        return '骨架极其不相似，与模板差异极大'
    elif score < 40:
        return '骨架很不相似，需要大幅度调整'
    elif score < 60:
        return '骨架不太相似，部分偏离模板'
    elif score < 80:
        return '骨架位置基本相似，但仍有改进空间'
    else:
        return '骨架位置非常相似，与模板高度一致'


#整体评价
Full_score = weighted_sum * 0.3 + similar_layout_score * 0.4 + normalized_distance  * 0.3
print('整体评分：', Full_score)

print(evaluate_stroke_similarity(weighted_sum))
print(evaluate_layout_similarity(similar_layout_score))
print(evaluate_skeleton_similarity(normalized_distance))




