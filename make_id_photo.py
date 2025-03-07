import cv2
import numpy as np
import os

def create_id_photo(input_image_path, output_image_path, background_color=(0, 0, 255)):
    """
    将自拍照片转换为蓝底证件照
    
    参数:
        input_image_path: 输入图片路径
        output_image_path: 输出图片路径
        background_color: 背景颜色，默认为蓝色 (BGR格式)
    """
    # 读取图片
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"无法读取图片: {input_image_path}")
        return
    
    # 调整图像大小以便处理
    height, width = img.shape[:2]
    max_dimension = 800
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    # 保存原始图像副本用于最终合成
    original_img = img.copy()
    
    # 使用更强大的背景分割方法
    # 首先找到人脸区域
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 创建初始掩码
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    if len(faces) > 0:
        # 取第一个检测到的人脸
        x, y, w, h = faces[0]
        
        # 扩大人脸区域以包含整个头部、脖子和衣领
        # 使用更大的扩展范围确保包含所有人像部分
        expanded_rect = (
            max(0, x - w),                          # 左边界扩大一个人脸宽度
            max(0, y - h),                          # 上边界扩大一个人脸高度（确保包含头发）
            min(w * 3, img.shape[1] - x + w),       # 宽度扩大到3倍人脸宽度（确保包含耳朵）
            min(h * 4, img.shape[0] - y + h)        # 高度扩大到4倍人脸高度（确保包含脖子和肩膀）
        )
        
        # 使用GrabCut算法进行前景分割
        # 创建GrabCut的初始掩码，初始化为可能的背景
        grabcut_mask = np.zeros(img.shape[:2], np.uint8) + 2
        
        # 标记人脸区域为确定的前景
        grabcut_mask[y:y+h, x:x+w] = 1
        
        # 扩展区域标记为可能的前景
        rect_x, rect_y, rect_w, rect_h = expanded_rect
        grabcut_mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = 3  # 可能的前景
        
        # 标记图像边缘和更多区域为确定的背景
        border_width = 30  # 增加边缘宽度
        if border_width < img.shape[0] and border_width < img.shape[1]:
            # 标记所有边缘为确定的背景
            grabcut_mask[:border_width, :] = 0  # 上边缘
            grabcut_mask[-border_width:, :] = 0  # 下边缘
            grabcut_mask[:, :border_width] = 0  # 左边缘
            grabcut_mask[:, -border_width:] = 0  # 右边缘
            
            # 标记扩展区域外的所有区域为确定的背景
            # 上方区域
            if rect_y > 0:
                grabcut_mask[:rect_y, :] = 0
            
            # 左侧区域
            if rect_x > 0:
                grabcut_mask[:, :rect_x] = 0
            
            # 右侧区域
            if rect_x + rect_w < img.shape[1]:
                grabcut_mask[:, rect_x+rect_w:] = 0
            
            # 下方区域
            if rect_y + rect_h < img.shape[0]:
                grabcut_mask[rect_y+rect_h:, :] = 0
    
        # 背景和前景模型
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # 应用GrabCut算法
        try:
            # 首先使用矩形初始化
            cv2.grabCut(img, grabcut_mask, expanded_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            # 再次运行GrabCut，这次使用掩码初始化
            cv2.grabCut(img, grabcut_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
            
            # 修改掩码，0和2为背景，1和3为前景
            mask = np.where((grabcut_mask==2)|(grabcut_mask==0), 0, 255).astype('uint8')
        except:
            print("GrabCut算法执行失败，使用备用方法")
            # 备用方法：使用肤色检测和矩形区域
            mask = create_backup_mask(img, faces[0], expanded_rect)
    else:
        print("未检测到人脸，使用备用方法")
        # 备用方法：使用肤色检测
        mask = create_backup_mask(img)
    
    # 应用形态学操作改善掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 使用洪水填充算法填充掩码中的空洞
    mask_floodfill = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_floodfill, flood_mask, (0,0), 255)  # 填充背景
    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)
    mask_combined = mask | mask_floodfill_inv  # 结合原始掩码和填充后的反转掩码
    
    # 再次应用形态学操作
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_combined = cv2.GaussianBlur(mask_combined, (9, 9), 0)
    
    # 寻找最大连通区域
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 创建新掩码，只保留最大连通区域
        refined_mask = np.zeros_like(mask_combined)
        cv2.drawContours(refined_mask, [max_contour], 0, 255, -1)
        
        # 再次应用形态学操作填充空洞
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        
        # 使用改进的掩码
        mask = refined_mask
    else:
        mask = mask_combined
    
    # 创建蓝色背景 (修改为标准蓝色)
    blue_background = np.ones_like(img) * np.array([255, 0, 0], dtype=np.uint8)  # BGR格式
    
    # 将掩码转换为3通道
    mask_3channel = cv2.merge([mask, mask, mask])
    
    # 创建反向掩码来标识背景区域
    inverse_mask = cv2.bitwise_not(mask)
    
    # 在GrabCut处理后，进一步改进掩码
    # 应用形态学操作改善掩码
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 使用颜色聚类来帮助识别墙壁
    # 将图像转换为LAB颜色空间，更适合颜色聚类
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 重塑图像为一维数组
    pixels = img_lab.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # 定义聚类条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5  # 聚类数量
    
    # 应用K-means聚类
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 将结果转换回原始形状
    labels = labels.reshape(img.shape[:2])
    
    # 找出可能的背景标签（假设背景颜色较为一致）
    if len(faces) > 0:
        # 获取人脸周围区域的标签
        face_x, face_y, face_w, face_h = faces[0]
        face_margin = 20
        x_min = max(0, face_x - face_margin)
        y_min = max(0, face_y - face_margin)
        x_max = min(img.shape[1], face_x + face_w + face_margin)
        y_max = min(img.shape[0], face_y + face_h + face_margin)
        
        # 获取人脸周围区域的标签
        face_region_labels = labels[y_min:y_max, x_min:x_max].flatten()
        
        # 获取图像边缘区域的标签
        edge_labels = []
        edge_labels.extend(labels[:border_width, :].flatten())  # 上边缘
        edge_labels.extend(labels[-border_width:, :].flatten())  # 下边缘
        edge_labels.extend(labels[:, :border_width].flatten())  # 左边缘
        edge_labels.extend(labels[:, -border_width:].flatten())  # 右边缘
        
        # 找出在边缘区域常见但在人脸区域不常见的标签
        edge_hist = np.bincount(edge_labels)
        face_hist = np.bincount(face_region_labels)
        
        # 确保直方图长度一致
        if len(face_hist) < len(edge_hist):
            face_hist = np.pad(face_hist, (0, len(edge_hist) - len(face_hist)))
        
        # 计算比率
        ratio = np.zeros_like(edge_hist, dtype=float)
        for i in range(len(edge_hist)):
            if face_hist[i] > 0:
                ratio[i] = edge_hist[i] / face_hist[i]
            else:
                ratio[i] = float('inf') if edge_hist[i] > 0 else 0
        
        # 找出可能的背景标签
        background_labels = [i for i in range(len(ratio)) if ratio[i] > 2.0]
        
        # 创建基于颜色聚类的背景掩码
        color_mask = np.zeros_like(mask)
        for label in background_labels:
            color_mask[labels == label] = 255
        
        # 结合原始掩码和颜色掩码
        combined_mask = cv2.bitwise_or(inverse_mask, color_mask)
        
        # 应用形态学操作
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 更新反向掩码
        inverse_mask = combined_mask
        
    # 更新3通道反向掩码
    inverse_mask_3channel = cv2.merge([inverse_mask, inverse_mask, inverse_mask])
    
    # 简化背景替换逻辑，确保背景完全被替换
    # 创建一个全蓝色的图像
    result = blue_background.copy()
    
    # 只在人像区域使用原始图像
    foreground = cv2.bitwise_and(original_img, mask_3channel)
    
    # 将前景添加到蓝色背景上
    result = cv2.bitwise_and(result, inverse_mask_3channel)
    result = cv2.add(result, foreground)
    
    # 删除下面这些重复的代码，它们会覆盖上面的结果
    # foreground = cv2.multiply(alpha, original_img.astype(float))
    # background = cv2.multiply(1.0 - alpha, blue_background.astype(float))
    # result = cv2.add(foreground, background).astype(np.uint8)
    
    # 调整为2寸证件照尺寸 (35mm x 45mm，约413 x 531像素，按照300dpi计算)
    id_photo_width = 413
    id_photo_height = 531
    
    # 改进的裁剪逻辑，使人头像更好地居中
    if len(faces) > 0:
        # 取第一个检测到的人脸
        x, y, w, h = faces[0]
        
        # 计算裁剪区域，使人脸居中
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # 调整裁剪比例，确保包含整个头部、脖子和部分衣领
        # 头顶到下巴的距离约为照片高度的0.4（更小的比例意味着包含更多的身体部分）
        crop_height = int(h / 0.4)  # 进一步增大比例以包含更多下部区域
        crop_width = int(crop_height * id_photo_width / id_photo_height)
        
        # 计算裁剪区域的左上角
        crop_x = max(0, face_center_x - crop_width // 2)
        # 头顶应该在照片上方约1/5处
        crop_y = max(0, face_center_y - h - int(crop_height * 0.2))
        
        # 确保裁剪区域不超出图像边界
        if crop_x + crop_width > result.shape[1]:
            crop_width = result.shape[1] - crop_x
        if crop_y + crop_height > result.shape[0]:
            crop_height = result.shape[0] - crop_y
        
        # 裁剪图像
        cropped = result[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        
        # 调整为标准2寸照片尺寸
        id_photo = cv2.resize(cropped, (id_photo_width, id_photo_height))
    else:
        print("未检测到人脸，将直接调整图像大小")
        # 如果没有检测到人脸，直接调整图像大小
        id_photo = cv2.resize(result, (id_photo_width, id_photo_height))
    
    # 保存结果
    cv2.imwrite(output_image_path, id_photo)
    print(f"证件照已保存至: {output_image_path}")

def create_backup_mask(img, face=None, expanded_rect=None):
    """
    当GrabCut算法失败时，创建备用掩码
    
    参数:
        img: 输入图像
        face: 人脸区域 (x, y, w, h)
        expanded_rect: 扩展的矩形区域 (x, y, w, h)
    
    返回:
        mask: 生成的掩码
    """
    # 转换为HSV颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 使用更广泛的肤色范围
    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    # 创建肤色掩码
    mask1 = cv2.inRange(img_hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(img_hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # 应用形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 如果有人脸信息，使用它来改进掩码
    if face is not None:
        x, y, w, h = face
        
        # 创建人脸区域掩码
        face_mask = np.zeros_like(skin_mask)
        face_mask[y:y+h, x:x+w] = 255
        
        # 扩展人脸区域
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        face_mask = cv2.dilate(face_mask, kernel_large, iterations=3)
        
        # 结合肤色掩码和人脸掩码
        combined_mask = cv2.bitwise_or(skin_mask, face_mask)
    else:
        combined_mask = skin_mask
    
    # 如果有扩展矩形区域，使用它来限制掩码
    if expanded_rect is not None:
        rect_x, rect_y, rect_w, rect_h = expanded_rect
        
        # 创建矩形区域掩码
        rect_mask = np.zeros_like(combined_mask)
        rect_mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = 255
        
        # 结合之前的掩码和矩形掩码
        combined_mask = cv2.bitwise_and(combined_mask, rect_mask)
    
    return combined_mask

if __name__ == "__main__":
    # 获取用户输入
    #input_path = input("请输入自拍照片路径: ")
    # 把input_path的文件路径设置为/Users/tigermei/Desktop/tigermei.jpg
    input_path = "/Users/tigermei/Desktop/tigermei.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 文件 {input_path} 不存在")
    else:
        # 生成输出路径
        filename, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(os.path.dirname(input_path), f"{filename}_id_photo{ext}")
        
        # 创建证件照
        create_id_photo(input_path, output_path)