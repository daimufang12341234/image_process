import cv2
import numpy as np

def gaussian_kernel(radius, sigma):
    """
    创建高斯核
    
    参数:
        radius: 模糊半径
        sigma: 标准差
    返回:
        kernel: 高斯核
    """
    size = 2 * radius + 1
    kernel = np.zeros(size)
    
    # 高斯公式的两部分
    a = 1 / np.sqrt(2 * np.pi)
    b = -1 / (2 * sigma * sigma)
    
    # 计算高斯矩阵
    for i in range(-radius, radius + 1):
        kernel[i + radius] = a * np.exp(b * i * i)
    
    # 归一化
    kernel = kernel / np.sum(kernel)
    
    return kernel

def gaussian_filter(image_path, radius=1, sigma=1.0):
    """应用分离高斯滤波"""
    # 直接读取为灰度图
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    height, width = gray.shape
    # 创建全0数组
    temp = np.zeros_like(gray, dtype=np.float32)
    result = np.zeros_like(gray, dtype=np.float32)
    
    # 创建高斯核
    kernel = gaussian_kernel(radius, sigma)
    
    # X方向一维运算
    for y in range(height):
        for x in range(radius, width - radius):
            gauss_sum = 0
            # 加权求和
            for j in range(-radius, radius + 1):
                if 0 <= x + j < width:
                    gauss_sum += kernel[j + radius] * gray[y, x + j]
            temp[y, x] = gauss_sum
    
    # Y方向一维运算
    for x in range(width):
        for y in range(radius, height - radius):
            gauss_sum = 0
            # 加权求和
            for j in range(-radius, radius + 1):
                if 0 <= y + j < height:
                    gauss_sum += kernel[j + radius] * temp[y + j, x]
            result[y, x] = gauss_sum
    
    # 转换回uint8类型
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

if __name__ == "__main__":
    input_image = "C:/Users/admin/Desktop/roi.jpg"
    output_dir = "C:/Users/admin/Desktop"
    
    try:
        # 应用高斯滤波
        filtered = gaussian_filter(input_image, radius=1, sigma=1.0)
        
        # 保存结果
        cv2.imwrite(f"{output_dir}/gaussian.jpg", filtered)
        print(f"结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}") 