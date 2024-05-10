import cv2
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions # type: ignore
from keras.utils import image_utils# type: ignore
import numpy as np
from keras.preprocessing import image# type: ignore
from sklearn.metrics.pairwise import cosine_similarity

model = VGG16(weights='imagenet', include_top=False)

#该函数对图片进行处理，将图片转为灰度图像，去除噪点，移除背景
def image_preprocessing(image,target_size=(299, 299)):
        # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 通过阈值处理将汉字背景变为白色
        ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)  # 大于阈值时置 0(黑)，否则置 255（白）
        # 反转二值化图像，将汉字变为黑色，背景变为白色
        thresh = cv2.bitwise_not(thresh)
        # 执行形态学操作去除噪点
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # 对图片的尺寸进行设定
        resized = cv2.resize(opening, target_size, interpolation=cv2.INTER_AREA)  # 添加了尺寸设定

        return resized


#VThin、HThin、Xihua函数对图片中的汉字进行细化
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


#该函数接受一个图像（假设它是一个 NumPy数组）作为输入，并返回一个二值化（binary）的图像
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


#对上述四个细化函数组装到一起
def Image_processing(image):
     iOne = Two(image)
     Image = Xihua(iOne, array)
     return Image

#该函数接收提取骨架后的汉字图片进行处理
def load_and_process_image_array(img):

    img_array = image.img_to_array(img)
    # 初始化img_tensor为None，如果图像已经是彩色，则稍后会被覆盖
    img_tensor = None
    # 检查图像是否是灰度图像（形状为(height, width)）
    if len(img_array.shape) == 2:
        # 将灰度值复制到三个通道以创建伪彩色图像
        img_array_rgb = np.stack((img_array,) * 3, axis=-1)
        # 调整图像大小并添加batch维度
        img_tensor = np.expand_dims(img_array_rgb, axis=0)
    else:
        # 如果图像已经是彩色的，则直接调整大小并添加batch维度
        img_tensor = np.expand_dims(img_array, axis=0)
    assert img_tensor is not None, "img_tensor was not properly defined"

    return img_tensor

#函数对提取特征，随后进行相似度计算
def extract_features(model, img):
    img_tensor = load_and_process_image_array(img)
    img_tensor = np.tile(img_tensor, (1, 1, 1, 3))
    img_tensor = img_tensor[:, :, :, :3]
    features = model.predict(img_tensor)
    return features[0]

#笔画结构计算，对处理之后的骨架图片进行计算
def calculate_hu_moments(image):
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
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


def calculate_weighted_sum(handwriting_image, template_image):

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
    weighted_sum = weighted_sum*76.924
    return weighted_sum


#计算布局相似度，直接对原图进行Contour_extraction函数进行轮廓提取，在使用calculate_similar_layout函数进行计算

def Contour_extraction(img):

    img = cv2.GaussianBlur(img, (3, 3), 3)
    th = cv2.adaptiveThreshold(img, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 11)
    th = cv2.bitwise_not(th)
    kernel = np.array([[0, 1, 1],
                           [0, 1, 0],
                           [1, 1, 0]], dtype='uint8')
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th

def calculate_similar_layout(img1, img2):

    # 确保图像有相同的高度和宽度，如果不相同，可以调整大小

    # 将图像转换为二值图像
    _, img1_binary = cv2.threshold(img1, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img2_binary = cv2.threshold(img2, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算图像中的轮廓
    contours1, _ = cv2.findContours(img1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算每个轮廓的布局特征
    layout_features1 = []
    for contour in contours1:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x / img1.shape[1]  # 归一化到[0, 1]
        right_distance = (img1.shape[1] - (x + w)) / img1.shape[1]
        top_distance = y / img1.shape[0]
        bottom_distance = (img1.shape[0] - (y + h)) / img1.shape[0]
        layout_features1.append([left_distance, right_distance, top_distance, bottom_distance, w / img1.shape[1],
                                 h / img1.shape[0]])  # 添加字宽和字高

    layout_features2 = []
    for contour in contours2:
        x, y, w, h = cv2.boundingRect(contour)
        left_distance = x / img2.shape[1]
        right_distance = (img2.shape[1] - (x + w)) / img2.shape[1]
        top_distance = y / img2.shape[0]
        bottom_distance = (img2.shape[0] - (y + h)) / img2.shape[0]
        layout_features2.append(
            [left_distance, right_distance, top_distance, bottom_distance, w / img2.shape[1], h / img2.shape[0]])

        # 计算布局相似度
    layout_similarities = []
    for feature1 in layout_features1:
        max_similarity = 0
        for feature2 in layout_features2:
            distance = np.sqrt(np.sum(np.square(np.array(feature1[:-2]) - np.array(feature2[:-2]))))  # 忽略字宽和字高
            similarity = 1 / (1 + distance)  # 简化的相似度度量
            if similarity > max_similarity:
                max_similarity = similarity
        layout_similarities.append(max_similarity)

        # 计算平均相似度
    return np.mean(layout_similarities)


#模板字和手写字读取
template_img_path  = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\1.png'
handwriting_img = r'E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\16.png'

handwriting_img =  cv2.imread(handwriting_img)
template_img =  cv2.imread(template_img_path)

#图片预处理，使用image_preprocessing函数
image1 = image_preprocessing(handwriting_img)
image2 = image_preprocessing(template_img)

#预处理之后进行细化处理
xihuaimage1 = Image_processing(image1)
xihuaimage2 = Image_processing(image2)

print('1')

#调用笔画提取及相似度评价
weighted_sum = calculate_weighted_sum(xihuaimage1, xihuaimage2)
weighted_sum = round(weighted_sum, 2)
print("笔画相似度 weighted_sum:", weighted_sum)

print('2')
#章法布局相似度及评价

img_contour1 = Contour_extraction(image1)
img_contour2 = Contour_extraction(image2)

similar_layout_score = calculate_similar_layout(img_contour1, img_contour2)
similar_layout_score = similar_layout_score * 100
similar_layout_score = round(similar_layout_score, 2)
print("布局相似度 similar_layout_score:", similar_layout_score)


#对提取的骨架进行特征提取，进行计算
vector1 = extract_features(model, xihuaimage1)
vector2 = extract_features(model, xihuaimage2)

print('3')

normalized_distance = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
normalized_distance = round(normalized_distance, 2)*100
normalized_distance = round(normalized_distance, 2)
print(f"骨架相似度  normalized_distance:" , normalized_distance)

print('4')


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


# print('书写“王”字时，需要注意以下几个要点：')
# print('结构:该字由三横一竖组成。最下面的横画是最长的，上面的两个横画是短横，两个短横长度差不多。长横与短横的对比要明显。上横画位于田字格上中线的中分位，中横画位于田字格的中线，第三横位于横中线下面一半的中间，而且要平直。竖画写在竖中线上')
# print('横画等距:三横之间应保持等距且平行')
# print('竖画的作用:竖画位于中间，起到平衡作用，使左右露出的横画长度相近。')
# print('整体是左边低右边高，微微上扬。')

#整体评价
Full_score = weighted_sum * 0.5 + similar_layout_score * 0.2 + normalized_distance  * 0.3
Full_score = round(Full_score, 2)
print('整体评分：', Full_score)

print(evaluate_stroke_similarity(weighted_sum))
print(evaluate_layout_similarity(similar_layout_score))
print(evaluate_skeleton_similarity(normalized_distance))