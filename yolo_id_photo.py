import cv2
import numpy as np
import os
import torch
import sys
from PIL import Image

def create_id_photo_yolo(input_image_path, output_image_path, background_color=(0, 0, 255)):
    """
    参数:
        input_image_path: 输入图片路径
        output_image_path: 输出图片路径
        background_color: 背景颜色，默认为红色 (BGR格式)
    """
    # 检查是否已安装YOLOv5
    # ... existing code ...
    
    # 检查是否已安装ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("正在安装ultralytics...")
        os.system("pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple")
        from ultralytics import YOLO
    
    # 读取图片
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"无法读取图片: {input_image_path}")
        return
    
    # 保存原始图像尺寸
    original_height, original_width = img.shape[:2]
    
    # 调整图像大小以便处理
    max_dimension = 800
    if max(original_height, original_width) > max_dimension:
        scale = max_dimension / max(original_height, original_width)
        img = cv2.resize(img, (int(original_width * scale), int(original_height * scale)))
    
    # 保存调整后的图像尺寸
    height, width = img.shape[:2]
    
    # 保存原始图像副本用于最终合成
    original_img = img.copy()
    
    # 加载YOLO模型
    print("加载YOLO模型...")
    try:
        model = YOLO("yolov8n-seg.pt")  # 使用YOLOv8分割模型
        # 使用模型进行检测和分割
        results = model(img[:, :, ::-1], conf=0.25, classes=[0])  # 只检测人类
        
        # 获取检测结果
        if len(results) > 0 and len(results[0].masks) > 0:
            # 获取第一个检测到的人体掩码
            mask_tensor = results[0].masks.data[0].cpu().numpy()
            # 调整掩码大小以匹配图像
            mask = cv2.resize(mask_tensor, (width, height))
            # 二值化掩码
            mask = (mask > 0.5).astype(np.uint8)
            
            # 获取边界框
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y = x1, y1
            w, h = x2 - x1, y2 - y1
            print(f"检测到人体: x={x}, y={y}, w={w}, h={h}")
            use_backup = False
        else:
            print("未检测到人体，尝试使用OpenCV的人脸检测...")
            use_backup = True
    except Exception as e:
        print(f"YOLO模型加载或检测失败: {e}")
        print("使用OpenCV的人脸检测作为备用...")
        use_backup = True
    
    if use_backup:
        # 使用OpenCV的人脸检测作为备用
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            print("未检测到人脸，无法创建证件照")
            return
        
        # 使用第一个检测到的人脸
        x, y, w, h = faces[0]
        
        # 创建简单的矩形掩码
        mask = np.zeros(img.shape[:2], np.uint8)
        # 扩大人脸区域以包含整个头部、脖子和肩膀
        head_x = max(0, x - w//2)
        head_y = max(0, y - h//2)
        head_w = min(w * 2, width - head_x)
        head_h = min(h * 3, height - head_y)
        mask[head_y:head_y+head_h, head_x:head_x+head_w] = 1
    
    # 使用形态学操作改善掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 保存原始掩码用于后续处理
    original_mask = mask.copy()
    
    # 增强对小异物的处理
    # 使用形态学操作改善掩码，增加闭运算的迭代次数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 增大核大小
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)  # 增加迭代次数
    
    # 添加额外的噪点清除步骤
    # 使用中值滤波去除小的噪点
    mask = cv2.medianBlur(mask, 5)
    
    # 使用形态学操作改善掩码，增加闭运算的迭代次数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 保存原始掩码用于后续处理
    original_mask = mask.copy()
    
    # 使用更精细的边缘处理技术，特别关注衣领区域
    # 1. 使用更小的结构元素创建精细边缘
    micro_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # 2. 创建非常窄的边缘掩码
    # 头发区域 - 窄边缘
    hair_dilated = cv2.dilate(original_mask, small_kernel, iterations=1)
    hair_edge = hair_dilated - original_mask
    
    # 脸部和其他区域 - 更窄的边缘
    face_dilated = cv2.dilate(original_mask, micro_kernel, iterations=1)
    face_edge = face_dilated - original_mask
    
    # 3. 创建区域掩码，根据图像位置划分不同区域，特别关注衣领区域
    height, width = mask.shape[:2]
    
    # 头发区域 (上部15%)
    hair_region = np.zeros_like(mask, dtype=np.float32)
    hair_height = int(height * 0.15)
    hair_region[:hair_height, :] = 1.0
    
    # 衣领区域 (下部30%)
    collar_region = np.zeros_like(mask, dtype=np.float32)
    collar_top = int(height * 0.7)  # 从70%的高度开始
    collar_region[collar_top:, :] = 1.0
    
    # 其他区域
    other_region = np.ones_like(mask, dtype=np.float32) - hair_region - collar_region
    
    # 平滑区域过渡
    hair_region = cv2.GaussianBlur(hair_region, (15, 15), 5)
    collar_region = cv2.GaussianBlur(collar_region, (15, 15), 5)
    
    # 4. 组合不同区域的边缘掩码，对衣领区域使用更强的处理
    edge_mask = (hair_edge * hair_region * 0.4 +
                face_edge * other_region * 0.3 +
                face_edge * collar_region * 0.5)  # 对衣领区域使用更强的边缘处理
    
    # 5. 应用高斯模糊，对衣领区域使用更强的模糊
    edge_mask = edge_mask.astype(np.float32)
    edge_mask = cv2.GaussianBlur(edge_mask, (9, 9), 3)  # 增加模糊程度
    
    # 6. 创建平滑的主体掩码，对衣领区域进行额外处理
    smooth_mask = cv2.GaussianBlur(original_mask.astype(float), (5, 5), 1.2)
    
    # 7. 对衣领区域进行额外的腐蚀处理，去除灰黑色线条
    # 创建衣领区域的腐蚀掩码
    # 增加眼镜区域的特殊处理
    # 使用边缘检测找到眼镜区域
    glasses_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glasses_edges = cv2.Canny(glasses_gray, 50, 150)
    glasses_dilated = cv2.dilate(glasses_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    glasses_region = np.zeros_like(mask, dtype=np.float32)
    
    # 假设眼镜在面部上半部分
    face_top = y
    face_bottom = y + h
    glasses_y = face_top + int(h * 0.2)  # 大约在脸部上方20%处
    glasses_height = int(h * 0.3)  # 大约占脸部高度的30%
    glasses_region[glasses_y:glasses_y+glasses_height, x:x+w] = 1.0
    glasses_region = glasses_region * cv2.GaussianBlur(glasses_dilated.astype(float) / 255.0, (5, 5), 1.0)
    
    # 创建衣领区域的腐蚀掩码 - 添加这一行来定义缺失的变量
    collar_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    collar_eroded = cv2.erode(original_mask, collar_erode_kernel, iterations=4)
    
    # 只在衣领区域应用腐蚀
    collar_mask = collar_eroded * collar_region + original_mask * (1 - collar_region)
    # 平滑过渡
    collar_mask = cv2.GaussianBlur(collar_mask, (9, 9), 2.0)  # 增强模糊效果
    # 更新主体掩码
    smooth_mask = smooth_mask * (1 - collar_region) + collar_mask * collar_region
    
    # 创建红色背景
    red_background = np.ones_like(img) * np.array(background_color, dtype=np.uint8)
    
    # 将掩码转换为3通道
    mask_3channel = np.stack([smooth_mask, smooth_mask, smooth_mask], axis=2)
    edge_mask_3channel = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
    
    # 8. 使用更多级别的透明度，特别关注衣领区域
    alpha_levels = [0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3]  # 增加更多级别
    edge_blend_total = np.zeros_like(img, dtype=np.float32)
    
    # 为每个透明度级别创建非常窄的边缘区域
    for i, alpha in enumerate(alpha_levels):
        # 创建逐渐扩大的结构元素
        level_size = 1 + i
        level_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (level_size, level_size))
        
        # 创建该级别的边缘掩码
        if i == 0:
            prev_dilated = np.zeros_like(edge_mask)
        else:
            prev_size = 1 + (i-1)
            prev_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prev_size, prev_size))
            prev_dilated = cv2.dilate(edge_mask.astype(np.uint8), prev_kernel, iterations=1)
            
        curr_dilated = cv2.dilate(edge_mask.astype(np.uint8), level_kernel, iterations=1)
        level_mask = (curr_dilated - prev_dilated).astype(np.float32) / 255.0
        
        # 应用不同区域的权重，对衣领区域使用更强的处理
        level_mask = level_mask * (hair_region * 0.5 + other_region * 0.4 + collar_region * 0.7)
        level_mask = np.clip(level_mask, 0, 1)
        
        # 转换为3通道
        level_mask_3ch = np.stack([level_mask, level_mask, level_mask], axis=2)
        
        # 创建该级别的混合，对衣领区域使用更高的背景比例
        collar_alpha = max(0.1, alpha - 0.2)  # 衣领区域使用更低的前景透明度
        region_alpha = alpha * (1 - collar_region) + collar_alpha * collar_region
        region_alpha_3ch = np.stack([region_alpha, region_alpha, region_alpha], axis=2)
        
        level_blend = (img.astype(np.float32) * region_alpha_3ch + 
                      red_background.astype(np.float32) * (1 - region_alpha_3ch)) * level_mask_3ch
        
        edge_blend_total += level_blend
    
    # 9. 合并所有层，主体区域保持原始图像不变
    foreground = img.astype(np.float32) * mask_3channel
    background = red_background.astype(np.float32) * (1 - mask_3channel - edge_mask_3channel)
    
    # 确保掩码不重叠
    edge_mask_3channel = np.clip(edge_mask_3channel, 0, 1 - np.clip(mask_3channel, 0, 1))
    
    # 合并结果
    result = foreground + background + edge_blend_total
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 10. 应用最终的高斯模糊，特别关注衣领区域
    # 创建衣领区域的掩码
    collar_blur_mask = np.zeros_like(mask, dtype=np.float32)
    collar_blur_mask[collar_top:, :] = 1.0
    collar_blur_mask = cv2.GaussianBlur(collar_blur_mask, (21, 21), 7)
    collar_blur_mask_3ch = np.stack([collar_blur_mask, collar_blur_mask, collar_blur_mask], axis=2)
    
    # 对衣领区域应用更强的模糊
    collar_blurred = cv2.GaussianBlur(result, (7, 7), 1.5)  # 增强模糊效果
    result = result * (1 - collar_blur_mask_3ch * 0.5) + collar_blurred * (collar_blur_mask_3ch * 0.5)  # 增加模糊权重
    
    # 额外的背景清理步骤
    # 创建更严格的掩码用于最终清理
    final_mask = cv2.erode(original_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    final_mask = cv2.GaussianBlur(final_mask.astype(float), (7, 7), 1.5)
    final_mask_3ch = np.stack([final_mask, final_mask, final_mask], axis=2)
    
    # 应用最终掩码，确保背景区域完全是纯色
    pure_background = np.ones_like(img) * np.array(background_color, dtype=np.uint8)
    result = result * final_mask_3ch + pure_background * (1 - final_mask_3ch)
    
    # 额外的边缘清理步骤
    # 创建边缘检测掩码
    # ... existing code ...
    
    # 修改边缘处理，特别关注问题区域
    # 增强边缘检测的参数
    edge_detect = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)  # 降低阈值以捕获更多边缘
    edge_detect = cv2.dilate(edge_detect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)  # 增加膨胀强度
    edge_detect = cv2.GaussianBlur(edge_detect.astype(float), (7, 7), 2.0) / 255.0  # 增强模糊效果
    
    # 将边缘检测掩码转换为3通道
    edge_detect_3ch = np.stack([edge_detect, edge_detect, edge_detect], axis=2)
    
    # 在边缘区域应用更强的背景混合
    edge_blend_factor = 0.85  # 增加边缘区域的背景混合因子
    edge_result = result * (1 - edge_detect_3ch * edge_blend_factor) + pure_background * (edge_detect_3ch * edge_blend_factor)
    
    # 只在原始掩码外部应用边缘清理
    outside_mask = 1 - cv2.dilate(original_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    outside_mask = cv2.GaussianBlur(outside_mask.astype(float), (5, 5), 1.0)
    outside_mask_3ch = np.stack([outside_mask, outside_mask, outside_mask], axis=2)
    
    result = result * (1 - outside_mask_3ch) + edge_result * outside_mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 调整为1寸证件照尺寸 (25mm x 35mm，约295 x 413像素，按照300dpi计算)
    id_photo_width = 295
    id_photo_height = 413
    
    # 计算裁剪区域，使人脸居中
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # 调整裁剪比例，确保包含整个头部、脖子和部分衣领
    # 头顶到下巴的距离约为照片高度的0.4
    crop_height = int(h / 0.4)
    crop_width = int(crop_height * id_photo_width / id_photo_height)
    
    # 计算裁剪区域的左上角
    crop_x = max(0, face_center_x - crop_width // 2)
    # 头顶应该在照片上方约1/5处
    crop_y = max(0, face_center_y - h - int(crop_height * 0.2))
    
    # 确保裁剪区域不超出图像边界
    if crop_x + crop_width > width:
        crop_width = width - crop_x
    if crop_y + crop_height > height:
        crop_height = height - crop_y
    
    # 裁剪图像
    cropped = result[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    
    # 调整为标准2寸照片尺寸
    id_photo = cv2.resize(cropped, (id_photo_width, id_photo_height))
    
    # 保存结果
    cv2.imwrite(output_image_path, id_photo)
    print(f"证件照已保存至: {output_image_path}")

if __name__ == "__main__":
    # 获取用户输入或使用默认路径
    input_path = "/Users/tigermei/Desktop/tigermei.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 文件 {input_path} 不存在")
    else:
        # 生成输出路径
        filename, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(os.path.dirname(input_path), f"{filename}_yolo_1inch_id_photo{ext}")
        
        # 创建证件照，使用蓝色背景
        create_id_photo_yolo(input_path, output_path, background_color=(255, 0, 0))
        