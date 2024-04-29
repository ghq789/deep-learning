import cv2
import numpy as np


# 定义函数计算图像的Hu矩
def calculate_hu_moments(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 计算Hu矩
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments)
    # 归一化处理
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + np.finfo(float).eps)
    return hu_moments.flatten()


def extract_grid_image(image, position):
    # 假设九宫格每个模块的大小是原图像的三分之一
    height, width = image.shape[:2]
    grid_height = height // 3
    grid_width = width // 3

    # 根据位置提取模块图像
    x, y = position
    start_x = x * grid_width
    start_y = y * grid_height
    end_x = start_x + grid_width
    end_y = start_y + grid_height

    # 确保提取范围在图像尺寸之内
    end_x = min(end_x, width)
    end_y = min(end_y, height)

    # 提取模块图像
    grid_image = image[start_y:end_y, start_x:end_x]
    return grid_image

# 加载原始图像 预处理之后的王
image = cv2.imread('E:\\python_files\\pytorch\\myself\\Handcopied_pictures\\h.png')
if image is None:
    raise ValueError("Image not found or path is incorrect")

# 加载参考模块图像（模板字预处理之后的王）
reference_image = cv2.imread('E:\\python_files\\pytorch\\myself\\Template_pictures\\1p.png')  # 替换为你的参考图像路径
if reference_image is None:
    raise ValueError("Reference image not found or path is incorrect")

# 计算参考模块的Hu矩
reference_hu_moments = calculate_hu_moments(reference_image)

# 定义九宫格的位置和权重
grid_positions = [(0, 0), (0, 1), (0, 2),
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1), (2, 2)]

grid_weights = [0.1, 0.1, 0.1,
                0.1, 0.5, 0.1,
                0.1, 0.1, 0.1]

# 初始化存储每个模块Hu矩的列表
hu_moments_list = []

# 遍历九宫格的位置
for position in grid_positions:
    # 提取当前模块的图像
    grid_image = extract_grid_image(image, position)
    # 计算当前模块的Hu矩
    hu_moments = calculate_hu_moments(grid_image)
    # 将Hu矩添加到列表中
    hu_moments_list.append(hu_moments)

# 计算皮尔逊相关系数
correlation_coefficients = []
for hu_moments in hu_moments_list:
    correlation_coefficient = np.corrcoef(reference_hu_moments, hu_moments)[0, 1]
    correlation_coefficients.append(correlation_coefficient)

# 加权求和
weighted_sum = np.dot(correlation_coefficients, grid_weights)

# 打印结果
print("Weighted Sum:", weighted_sum)


#加载原始图像。
# 加载参考模块图像。
# 计算参考模块的Hu矩。

# 定义九宫格的位置和权重。
# 遍历九宫格的位置，对每个位置：
# a. 提取当前位置的模块图像。b. 计算当前模块的Hu矩。
# 计算每个模块Hu矩与参考模块Hu矩之间的皮尔逊相关系数。
# 根据权重计算加权和。