import cv2
import numpy as np

def big_variance(image, threshold):
    """
    计算类间方差
    """
    m, n = image.shape
    total_pixels = m * n
    
    histogram = np.bincount(image.ravel(), minlength=256)
    p_i = histogram / total_pixels
    
    w1 = np.sum(p_i[:threshold+1])
    w2 = np.sum(p_i[threshold+1:])
    
    if w1 == 0 or w2 == 0:
        return 0
    
    mu1 = np.sum(np.arange(threshold+1) * p_i[:threshold+1]) / w1
    mu2 = np.sum(np.arange(threshold+1, 256) * p_i[threshold+1:]) / w2
    
    variance = w1 * w2 * (mu1 - mu2)**2
    
    return variance

def otsu(image_path):
    """
    使用大津法计算最佳阈值
    """
    # 直接读取灰度图
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    max_variance = 0
    best_threshold = 0
    
    for t in range(256):
        variance = big_variance(gray, t)
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    _, segmented = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)
    
    return best_threshold, segmented

if __name__ == "__main__":
    input_image = "C:/Users/admin/Desktop/gaussian.jpg"
    output_dir = "C:/Users/admin/Desktop"
    
    try:
        threshold, segmented_image = otsu(input_image)
        cv2.imwrite(f"{output_dir}/segmented.jpg", segmented_image)
        print(f"最佳阈值: {threshold}")
        print(f"结果已保存到: {output_dir}/segmented.jpg")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}") 