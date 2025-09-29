import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from pathlib import Path
import requests

# 解决中文字体问题 - 只使用SimHei字体
def setup_chinese_font():
    """设置中文字体，仅使用SimHei确保在Windows环境下正常显示"""
    # 只保留SimHei字体选项
    font_options = [
        {"name": "SimHei", "path": "C:/Windows/Fonts/simhei.ttf"},
    ]
    
    # 检查系统中已安装的SimHei字体
    system_fonts = fm.findSystemFonts()
    simhei_fonts = []
    for font_path in system_fonts:
        try:
            font_name = fm.get_font(font_path).family_name
            if 'simhei' in font_name.lower():
                simhei_fonts.append({"name": font_name, "path": font_path})
        except:
            continue
    
    if simhei_fonts:
        font_prop = FontProperties(fname=simhei_fonts[0]["path"])
        print(f"成功加载系统中文字体: {simhei_fonts[0]['name']}")
        
        # 只设置SimHei字体
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False
        return font_prop
    
    # 创建字体目录
    font_dir = os.path.expanduser("~/.fonts")
    Path(font_dir).mkdir(parents=True, exist_ok=True)
    
    # 尝试下载并加载SimHei字体
    for font in font_options:
        try:
            if not os.path.exists(font["path"]) and "url" in font and font["url"]:
                print(f"正在下载 {font['name']} 字体...")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(font["url"], headers=headers, timeout=30)
                response.raise_for_status()
                
                with open(font["path"], "wb") as f:
                    f.write(response.content)
                
                if os.path.getsize(font["path"]) < 1024 * 100:
                    raise Exception("字体文件不完整")
            
            if os.path.exists(font["path"]):
                font_prop = FontProperties(fname=font["path"])
                if fm.findfont(font_prop):
                    print(f"成功加载字体: {font['name']}")
                    break
                
        except Exception as e:
            print(f"加载 {font['name']} 失败: {str(e)}")
            continue
    else:
        print("警告: SimHei字体加载失败，中文可能无法正常显示")
        return FontProperties()
    
    # 应用字体设置 - 只使用SimHei
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False
    print("中文显示设置完成 (仅使用SimHei)")
    
    return font_prop

# 定义字符集 - 包含所有可能的车牌字符
provincelist = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "西", "陕", "甘", "青", "宁", "新"
]

wordlist = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", 
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

# 合并所有字符
all_chars = provincelist + wordlist
char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
num_classes = len(all_chars)

# 车牌固定长度为7个字符
PLATE_LENGTH = 7

class PlateRecognitionModel(nn.Module):
    """车牌识别模型，用于加载训练好的权重进行推理"""
    def __init__(self, num_classes):
        super(PlateRecognitionModel, self).__init__()
        # 使用MobileNetV2作为特征提取器
        self.cnn = mobilenet_v2(weights=None)  # 不加载预训练权重，我们将加载自己的
        self.features = self.cnn.features
        self.feature_channels = 1280
        
        # 减小通道数
        self.conv1x1 = nn.Conv2d(self.feature_channels, 64, kernel_size=1)
        self.bn = nn.BatchNorm2d(64)
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=64 * 4,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.6
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 输出层
        self.fc = nn.Linear(256, num_classes)
        self.seq_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, x):
        # CNN特征提取
        batch_size = x.size(0)
        x = self.features(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # 调整形状以适应LSTM
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # LSTM处理
        x, _ = self.lstm(x)
        
        # 应用注意力机制
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = x * attn_weights
        
        # 确保输出固定长度7
        if x.size(1) >= PLATE_LENGTH:
            x = x[:, -PLATE_LENGTH:, :]
        else:
            pad_length = PLATE_LENGTH - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_length))
        
        # 应用序列批归一化和Dropout
        x = x.permute(0, 2, 1)
        x = self.seq_bn(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        
        # 预测每个位置的字符
        x = self.fc(x)
        
        return x

def correct_skew(image, vertices):
    """矫正倾斜的车牌"""
    if len(vertices) != 4:
        rect = cv2.minAreaRect(vertices)
        vertices = cv2.boxPoints(rect)
    
    # 排序顶点
    center = np.mean(vertices, axis=0)
    
    def get_angle(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0]) * 180 / np.pi
    
    vertices = sorted(vertices, key=get_angle)
    
    # 计算宽度和高度
    width_top = np.linalg.norm(vertices[0] - vertices[1])
    width_bottom = np.linalg.norm(vertices[2] - vertices[3])
    width = int((width_top + width_bottom) / 2)
    
    height_left = np.linalg.norm(vertices[0] - vertices[3])
    height_right = np.linalg.norm(vertices[1] - vertices[2])
    height = int((height_left + height_right) / 2)
    
    # 确保宽大于高
    if width < height:
        width, height = height, width
        vertices = [vertices[1], vertices[2], vertices[3], vertices[0]]
    
    # 目标矩形
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype=np.float32)
    
    # 透视变换
    M = cv2.getPerspectiveTransform(np.array(vertices, dtype=np.float32), dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    # 调整到目标大小
    corrected_plate = cv2.resize(warped, (400, 100))
    
    return corrected_plate

class LicensePlateRecognizer:
    """车牌识别器，用于加载模型并进行推理"""
    def __init__(self, detection_model_path, recognition_model_path, chinese_font):
        """初始化识别器"""
        # 设置设备为CPU
        self.device = torch.device('cpu')
        print("使用设备: CPU")
        
        self.chinese_font = chinese_font
        self.img_size = (400, 100)
        
        # 加载检测模型
        print(f"加载车牌检测模型: {detection_model_path}")
        self.detection_model = YOLO(detection_model_path)
        
        # 初始化并加载识别模型
        print(f"加载车牌识别模型: {recognition_model_path}")
        self.recognition_model = PlateRecognitionModel(num_classes).to(self.device)
        self.recognition_model.load_state_dict(torch.load(
            recognition_model_path, 
            map_location=self.device,
            weights_only=True
        ))
        self.recognition_model.eval()
        
        # 测试时的数据变换
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def detect_plate(self, image_path):
        """检测并裁剪车牌"""
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        
        # 转换为RGB用于显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测车牌
        results = self.detection_model(image, verbose=False)
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None, image_rgb, None
        
        # 选择置信度最高的车牌
        best_idx = boxes.conf.argmax().item()
        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = boxes.conf[best_idx].item()
        
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制边界框
        marked_image = image_rgb.copy()
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 将中文"车牌"改为英文"License"
        cv2.putText(
            marked_image, 
            f"License: {conf:.2f}", 
            (x1, y1-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
        
        # 裁剪车牌区域
        plate_region = image[y1:y2, x1:x2]
        
        # 尝试使用顶点信息进行矫正
        try:
            # 如果有顶点信息，使用顶点进行矫正
            # 这里简化处理，实际应用中可以从标注文件读取顶点信息
            vertices = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            corrected_plate = correct_skew(image, vertices)
        except:
            # 矫正失败，使用原始裁剪并调整大小
            corrected_plate = cv2.resize(plate_region, self.img_size)
        
        # 转换为RGB格式
        corrected_plate_rgb = cv2.cvtColor(corrected_plate, cv2.COLOR_BGR2RGB)
        
        return corrected_plate_rgb, marked_image, conf
    
    def recognize_plate(self, plate_image):
        """识别车牌内容"""
        # 应用变换
        image = cv2.resize(plate_image, self.img_size)
        image_tensor = self.test_transform(image).unsqueeze(0)  # 添加批次维度
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.recognition_model(image_tensor)
            _, predicted = torch.max(output, 2)
        
        # 转换为字符
        predicted_chars = [idx_to_char[idx.item()] for idx in predicted[0]]
        return ''.join(predicted_chars)
    
    def process_image(self, image_path, save_result=True):
        """处理单张图片，检测并识别车牌"""
        # 检测车牌
        plate_image, marked_image, conf = self.detect_plate(image_path)
        
        if plate_image is None:
            print("未检测到车牌")
            # 显示原始图片
            plt.figure(figsize=(10, 6))
            plt.imshow(marked_image)
            plt.title("未检测到车牌", fontproperties=self.chinese_font)
            plt.axis('off')
            plt.show()
            return None
        
        # 识别车牌
        plate_number = self.recognize_plate(plate_image)
        print(f"识别结果: {plate_number} (检测置信度: {conf:.2f})")
        
        # 显示结果
        plt.figure(figsize=(12, 6))
        
        # 原始图片带检测框
        plt.subplot(1, 2, 1)
        plt.imshow(marked_image)
        plt.title("原始图片与检测结果", fontproperties=self.chinese_font)
        plt.axis('off')
        
        # 裁剪后的车牌与识别结果
        plt.subplot(1, 2, 2)
        plt.imshow(plate_image)
        plt.title(f"识别结果: {plate_number}", fontproperties=self.chinese_font)
        plt.axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        if save_result:
            result_dir = "recognition_results"
            os.makedirs(result_dir, exist_ok=True)
            img_name = os.path.basename(image_path)
            result_path = os.path.join(result_dir, f"result_{img_name}")
            plt.savefig(result_path, dpi=300, bbox_inches='tight')
            print(f"识别结果已保存至: {result_path}")
        
        plt.show()
        return plate_number

if __name__ == "__main__":
    # 设置中文字体
    chinese_font = setup_chinese_font()
    
    # 模型路径 - 请替换为你的模型实际路径
    detection_model_path = "./output/model/best.pt"       # 车牌检测模型
    recognition_model_path = "./output/model/best_mobilenet.pt"  # 车牌识别模型
    
    # 检查模型文件是否存在
    if not os.path.exists(detection_model_path):
        raise FileNotFoundError(f"未找到检测模型文件: {detection_model_path}")
    
    if not os.path.exists(recognition_model_path):
        raise FileNotFoundError(f"未找到识别模型文件: {recognition_model_path}")
    
    # 初始化识别器
    recognizer = LicensePlateRecognizer(
        detection_model_path=detection_model_path,
        recognition_model_path=recognition_model_path,
        chinese_font=chinese_font
    )
    
    # 输入图片路径 - 请替换为你的图片路径
    image_path = "./test/3.jpg"  # 测试图片
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图片文件: {image_path}")
    
    # 处理图片并识别车牌
    recognizer.process_image(image_path)
