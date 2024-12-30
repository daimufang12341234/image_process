import cv2
import numpy as np
import os


def gray_transform(image, transform_type='a', a=1.0, n=2):
    """实现基本灰度变换，支持多种变换类型"""
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检查是否为彩色图像，如果是则转换为灰度图
    
    img_float = image.astype(float) / 255.0
    # 将图像数据转换为0-1范围的浮点数
    
    if transform_type == 'a':
        output = img_float
        # 线性变换
    elif transform_type == 'b':
        output = a - img_float
        # 反转变换
    elif transform_type == 'log':
        output = np.log(1 + img_float) / np.log(2)
        # 对数变换
    elif transform_type == 'c':
        output = np.power(img_float, n)
        # 幂次变换
    elif transform_type == 'n':
        output = np.power(img_float, 1/n)
        # n次根变换
    
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return output
    # 将结果转换回0-255范围的整数并返回

def enhance(image, a=2):
    """实现对比度增强，使用公式 g(x,y) = a(f(x,y) - mean) + f(x,y)"""
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检查是否为彩色图像，如果是则转换为灰度图
    
    image_float = image.astype(float)
    mean = np.mean(image_float)
    # 将图像转换为浮点数并计算平均灰度值
    
    enhanced = a * (image_float - mean) + image_float
    # 应用对比度增强公式
    
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced
    # 将结果限制在0-255范围内并返回

def process_image(input_path, output_dir):
    """处理图像并保存所有变换结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    # 创建输出目录，如果已存在则不报错
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # 获取输入文件名（不含扩展名）
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")
    # 读取输入图像并检查是否成功
    
    results = {
        'a': gray_transform(img, 'a'),
        'b': gray_transform(img, 'b'),
        'log': gray_transform(img, 'log'),
        'c': gray_transform(img, 'c', n=2),
        'n': gray_transform(img, 'n', n=2),
        'enhance': enhance(img, a=1.5)
    }
    # 对图像应用所有变换类型
    
    for transform_type, result_img in results.items():
        output_path = os.path.join(output_dir, f"{base_name}_{transform_type}.jpg")
        cv2.imwrite(output_path, result_img)
    # 保存所有处理结果
        
    return results

if __name__ == "__main__":
    input_image = "C:/Users/admin/Desktop/test.png"
    output_directory = "C:/Users/admin/Desktop"
    # 设置输入输出路径
    
    try:
        results = process_image(input_image, output_directory)
        # 处理图像
        
        for name, img in results.items():
            cv2.imshow(name, img)
        # 显示所有处理结果
        
        print(f"处理完成！结果已保存到 {output_directory} 目录")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 等待按键并关闭所有窗口
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
    # 错误处理 