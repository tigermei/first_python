import os
import cv2
import numpy as np
import torch
import urllib.request
from PIL import Image
from torchvision import transforms
from skimage import transform

class U2NET:
    def __init__(self):
        self.model_dir = os.path.join(os.path.expanduser('~'), '.u2net')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'u2net_portrait.pth')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 下载模型（如果不存在）
        print(self.model_path)
        if not os.path.exists(self.model_path):
            print("正在下载 U²-Net 人像分割模型...")
            # 提供直接下载链接，而不是Google Drive链接
            url = "https://github.com/xuebinqin/U-2-Net/releases/download/1.0/u2net_portrait.pth"
            try:
                print(f"正在从 {url} 下载模型文件...")
                urllib.request.urlretrieve(url, self.model_path, 
                                          reporthook=lambda count, block_size, total_size: print(f"下载进度: {count*block_size/total_size*100:.1f}%", end="\r"))
                print("\n下载完成!")
            except Exception as e:
                print(f"下载失败: {e}")
                print("请手动下载模型文件并放置在以下位置：")
                print(self.model_path)
                print("下载链接：https://github.com/xuebinqin/U-2-Net/releases/download/1.0/u2net_portrait.pth")
                raise ImportError("无法自动下载模型文件，请手动下载")
            print(f"模型已下载到: {self.model_path}")
        
        # 加载模型
        print(f"正在加载模型到 {self.device} 设备...")
        from model import U2NET
        self.net = U2NET(3, 1)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
    
    def preprocess(self, image):
        # 将图像调整为 512x512
        image = image.resize((512, 512), Image.BILINEAR)
        # 转换为张量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image_tensor):
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = self.net(image_tensor)
        # 使用 d1 作为最终预测结果
        pred = d1[:, 0, :, :]
        pred = pred.cpu().numpy()
        return pred
    
    def postprocess(self, pred, original_size):
        # 将预测结果调整为原始图像大小
        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi)
        pred = transform.resize(pred[0], original_size, order=2)
        return pred

def download_model_files():
    """下载模型所需的文件"""
    # 创建模型目录
    model_dir = os.path.join(os.getcwd(), 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # 下载 model.py 文件
    model_py_url = "https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/model/u2net.py"
    model_py_path = os.path.join(model_dir, 'model.py')
    if not os.path.exists(model_py_path):
        print("正在下载 model.py...")
        urllib.request.urlretrieve(model_py_url, model_py_path)
        print(f"model.py 已下载到: {model_py_path}")
    
    # 创建 __init__.py 文件
    init_py_path = os.path.join(model_dir, '__init__.py')
    if not os.path.exists(init_py_path):
        with open(init_py_path, 'w') as f:
            f.write("from .model import U2NET, U2NETP\n")
        print(f"__init__.py 已创建: {init_py_path}")

def extract_portrait(input_path, output_path, background_color=(0, 0, 255)):
    """
    从自拍照片中抠出人物头像
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径
        background_color: 背景颜色，默认为蓝色 (RGB格式)
    """
    # 下载模型文件
    download_model_files()
    
    # 加载 U²-Net 模型
    u2net = U2NET()
    
    # 读取图像
    image = Image.open(input_path).convert('RGB')
    original_size = image.size
    
    # 预处理
    image_tensor = u2net.preprocess(image)
    
    # 预测
    print("正在进行人像分割...")
    pred = u2net.predict(image_tensor)
    
    # 后处理
    mask = u2net.postprocess(pred, (image.height, image.width))
    
    # 对掩码进行处理，增强对比度
    # 先进行高斯模糊平滑掩码边缘
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 将掩码值映射到0-1范围，并增强对比度
    mask = np.clip((mask - 0.2) * 1.5, 0, 1)
    
    # 将图像转换为 numpy 数组
    image_np = np.array(image)
    
    # 创建背景图像 (RGB格式)
    background = np.ones_like(image_np) * np.array(background_color)
    
    # 创建3通道掩码
    mask_3channel = np.stack([mask, mask, mask], axis=2)
    
    # 使用掩码合并前景和背景 (在RGB空间中)
    result = image_np * mask_3channel + background * (1 - mask_3channel)
    result = result.astype(np.uint8)
    
    # 转换为BGR格式保存
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"人像已提取并保存至: {output_path}")
    
    # 返回掩码，以便进一步处理
    return mask

def create_id_photo(input_path, output_path, background_color=(0, 0, 255), id_size=(413, 531)):
    """
    创建证件照
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径
        background_color: 背景颜色，默认为蓝色 (RGB格式)
        id_size: 证件照尺寸，默认为2寸 (413x531像素)
    """
    # 提取人像
    mask = extract_portrait(input_path, output_path, background_color)
    
    # 读取生成的图像
    img = cv2.imread(output_path)
    height, width = img.shape[:2]
    
    print("使用 U²-Net 分割结果估计人脸位置...")
    # 将掩码转换为二值图像
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # 找到掩码的非零区域
    y_indices, x_indices = np.where(mask_binary > 0)
    if len(y_indices) > 0 and len(x_indices) > 0:
        # 获取边界框
        top = np.min(y_indices)
        bottom = np.max(y_indices)
        left = np.min(x_indices)
        right = np.max(x_indices)
        
        # 计算人像高度和宽度
        mask_height = bottom - top
        mask_width = right - left
        
        # 估计人脸位置（通常在上部1/3处）
        face_height = mask_height // 3
        face_y = top + face_height // 2
        face_x = left + mask_width // 2 - face_height // 2  # 假设人脸宽高比接近1:1
        face_w = face_height
        face_h = face_height
        
        print(f"基于 U²-Net 掩码估计人脸位置: x={face_x}, y={face_y}, w={face_w}, h={face_h}")
    else:
        # 如果掩码为空，使用图像中心
        print("掩码为空，使用图像中心...")
        face_w = width // 3
        face_h = face_w
        face_x = width // 2 - face_w // 2
        face_y = height // 3 - face_h // 2
    
    # 计算裁剪区域，使人脸居中
    face_center_x = face_x + face_w // 2
    face_center_y = face_y + face_h // 2
    
    # 调整裁剪比例，确保包含整个头部、脖子和部分衣领
    # 头顶到下巴的距离约为照片高度的0.4
    crop_height = int(face_h / 0.4)
    crop_width = int(crop_height * id_size[0] / id_size[1])
    
    # 计算裁剪区域的左上角
    crop_x = max(0, face_center_x - crop_width // 2)
    # 头顶应该在照片上方约1/5处
    crop_y = max(0, face_center_y - face_h - int(crop_height * 0.2))
    
    # 确保裁剪区域不超出图像边界
    if crop_x + crop_width > width:
        crop_width = width - crop_x
    if crop_y + crop_height > height:
        crop_height = height - crop_y
    
    # 裁剪图像
    cropped = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    
    # 调整为标准证件照尺寸
    id_photo = cv2.resize(cropped, id_size)
    
    # 保存结果
    cv2.imwrite(output_path, id_photo)
    print(f"证件照已保存至: {output_path}")

if __name__ == "__main__":
    # 获取用户输入或使用默认路径
    input_path = "/Users/tigermei/Desktop/tigermei.jpg"
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 文件 {input_path} 不存在")
    else:
        # 生成输出路径
        filename, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(os.path.dirname(input_path), f"{filename}_blue_id_photo{ext}")
        
        # 创建证件照，使用蓝色背景 (RGB格式：蓝色为 0,0,255)
        create_id_photo(input_path, output_path, background_color=(0, 0, 255), id_size=(413, 531))