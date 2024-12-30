import cv2
import numpy as np

def create_kernel(size=3):
    """
    创建结构元素（方形核）
    
    参数:
        size: 核的大小
    返回:
        kernel: 结构元素
    """
    return np.ones((size, size), np.uint8)

def big(image_path, kernel_size=3):
    """
    实现膨胀操作
    
    参数:
        image_path: 输入图像路径
        kernel_size: 结构元素大小
    返回:
        dilated: 膨胀后的图像
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 创建结构元素
    kernel = create_kernel(kernel_size)
    
    # 应用膨胀操作
    dilated = cv2.dilate(img, kernel, iterations=1)
    
    return dilated

def small(image_path, kernel_size=3):
    """
    实现腐蚀操作
    
    参数:
        image_path: 输入图像路径
        kernel_size: 结构元素大小
    返回:
        eroded: 腐蚀后的图像
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 创建结构元素
    kernel = create_kernel(kernel_size)
    
    # 应用腐蚀操作
    eroded = cv2.erode(img, kernel, iterations=1)
    
    return eroded

def opening(image_path, kernel_size=3):
    """
    实现开运算（先腐蚀后膨胀）
    
    参数:
        image_path: 输入图像路径
        kernel_size: 结构元素大小
    返回:
        opened: 开运算后的图像
    """
    # 先腐蚀后膨胀
    eroded = small(image_path, kernel_size)
    
    # 将腐蚀结果保存为临时文件
    temp_path = "temp_eroded.jpg"
    cv2.imwrite(temp_path, eroded)
    
    # 对腐蚀结果进行膨胀
    opened = big(temp_path, kernel_size)
    
    # 删除临时文件
    import os
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return opened

def closing(image_path, kernel_size=3):
    """
    实现闭运算（先膨胀后腐蚀）
    
    参数:
        image_path: 输入图像路径
        kernel_size: 结构元素大小
    返回:
        closed: 闭运算后的图像
    """
    # 先膨胀后腐蚀
    dilated = big(image_path, kernel_size)
    
    # 将膨胀结果保存为临时文件
    temp_path = "temp_dilated.jpg"
    cv2.imwrite(temp_path, dilated)
    
    # 对膨胀结果进行腐蚀
    closed = small(temp_path, kernel_size)
    
    # 删除临时文件
    import os
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return closed

if __name__ == "__main__":
    # 设置输入输出路径
    input_image = "C:/Users/admin/Desktop/segmented.jpg"
    output_dir = "C:/Users/admin/Desktop"
    
    try:
        # 应用形态学操作
        dilated_image = big(input_image)
        eroded_image = small(input_image)
        opened_image = opening(input_image)
        closed_image = closing(input_image)
        
        # 保存结果
        cv2.imwrite(f"{output_dir}/big.jpg", dilated_image)
        cv2.imwrite(f"{output_dir}/small.jpg", eroded_image)
        cv2.imwrite(f"{output_dir}/opened.jpg", opened_image)
        cv2.imwrite(f"{output_dir}/closed.jpg", closed_image)
        
        print(f"结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}") 