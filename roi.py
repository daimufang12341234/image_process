import cv2
import numpy as np

def roi(image_path):
    """
    检测激光条纹的ROI区域
    
    参数:
        image_path: 输入图像路径
    返回:
        roi_coords: ROI坐标 [xmin, xmax, ymin, ymax]
    """
    # 直接读取为灰度图
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    height = gray.shape[0]
    
    # 计算每行的灰度值之和
    row_sums = np.sum(gray, axis=1)
    
    # 找到灰度值和最大的行索引
    vt = np.argmax(row_sums)
    
    # 初始化ROI边界
    roi_top = vt
    roi_bottom = vt
    
    # 向上搜索上边界
    for i in range(vt-1, -1, -1):
        if row_sums[i] < row_sums[i+1]:
            roi_top = i
            break
    
    # 向下搜索下边界
    for i in range(vt+1, height):
        if row_sums[i] < row_sums[i-1]:
            roi_bottom = i
            break
    
    # 扩展ROI区域（改为扩展50像素）
    roi_top = max(0, roi_top - 50)
    roi_bottom = min(height-1, roi_bottom + 50)
    
    # X轴范围设置为图像宽度范围
    width = gray.shape[1]
    roi_left = 0
    roi_right = width
    
    return [roi_left, roi_right, roi_top, roi_bottom]

def cat_roi(image_path, roi_coords):
    """
    裁切ROI区域
    
    参数:
        image_path: 输入图像路径
        roi_coords: ROI坐标 [xmin, xmax, ymin, ymax]
    返回:
        cropped_img: 裁切后的图像
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    x1, x2, y1, y2 = roi_coords
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img

if __name__ == "__main__":
    # 设置输入输出路径
    input_image = "C:/Users/admin/Desktop/test_c.jpg"
    output_dir = "C:/Users/admin/Desktop"
    
    try:
        # 检测ROI
        roi_coords = roi(input_image)
        print(f"检测到的ROI坐标: {roi_coords}")
        
        # 裁切ROI并保存
        cropped_image = cat_roi(input_image, roi_coords)
        cv2.imwrite(f"{output_dir}/roi.jpg", cropped_image)

        print(f"结果已保存到: {output_dir}/roi.jpg")

        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}") 