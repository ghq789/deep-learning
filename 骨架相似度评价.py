
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions # type: ignore
from keras.utils import image_utils# type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keras.preprocessing import image# type: ignore

model = VGG16(weights='imagenet', include_top=False)

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


# # 图片路径
# img_path1 = r'E:\\python_files\\pytorch\\myself\\gujiatiqu\\a.png'
# img_path2 = r'E:\\python_files\\pytorch\\myself\\gujiatiqu\\b.png'
#
#
# # 提取特征向量,利用余弦进行相似度计算
# vector1 = np.array(extract_features(model, img_path1)).reshape(1, 25088)
# vector2 = np.array(extract_features(model, img_path2)).reshape(1, 25088)
# # vector1 = np.array(extract_features(model, img_path1))
# # vector2 = extract_features(model, img_path3)
# normalized_distance = cosine_similarity(vector1, vector2)
# print(f"The Euclidean distance between vector1 and vector2 is: {normalized_distance}")


