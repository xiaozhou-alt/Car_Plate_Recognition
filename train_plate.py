import os
import cv2
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
import yaml
from ultralytics import YOLO
import torch.nn.functional as F
import json
from collections import defaultdict
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from pathlib import Path
import requests

# 解决中文字体问题 - 优化版
def setup_chinese_font():
    """改进的中文字体设置函数，确保在各种环境下正常显示中文"""
    font_options = [
        {"name": "SimHei", "path": "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"},
        {"name": "WenQuanYi Micro Hei", "path": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"},
        {"name": "Heiti TC", "path": ""},
        {"name": "Microsoft YaHei", "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/msyh.ttc",
         "path": os.path.expanduser("~/.fonts/msyh.ttc")},
        {"name": "SimHei", "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",
         "path": os.path.expanduser("~/.fonts/SimHei.ttf")},
        {"name": "WenQuanYi Micro Hei", "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/wqy-microhei.ttc",
         "path": os.path.expanduser("~/.fonts/wqy-microhei.ttc")},
        {"name": "SimSun", "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/simsun.ttc",
         "path": os.path.expanduser("~/.fonts/simsun.ttc")},
        {"name": "Arial Unicode MS", "path": ""}
    ]
    
    # 检查系统中已安装的中文字体
    system_fonts = fm.findSystemFonts()
    chinese_fonts = []
    for font_path in system_fonts:
        try:
            font_name = fm.get_font(font_path).family_name
            if any(name in font_name.lower() for name in 
                  ['heiti', 'simhei', 'microsoft yahei', 'simsun', 'wenquanyi', 'song', 'noto sans cjk']):
                chinese_fonts.append({"name": font_name, "path": font_path})
        except:
            continue
    
    if chinese_fonts:
        font_prop = FontProperties(fname=chinese_fonts[0]["path"])
        print(f"成功加载系统中文字体: {chinese_fonts[0]['name']}")
        
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Noto Sans CJK SC", chinese_fonts[0]["name"]]
        plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Noto Sans CJK SC", chinese_fonts[0]["name"]]
        plt.rcParams['axes.unicode_minus'] = False
        return font_prop
    
    # 创建字体目录
    font_dir = os.path.expanduser("~/.fonts")
    Path(font_dir).mkdir(parents=True, exist_ok=True)
    
    # 尝试下载并加载字体
    for font in font_options:
        try:
            if font["name"] == "Arial Unicode MS":
                arial_fonts = [f for f in fm.findSystemFonts() if "arial unicode ms" in f.lower()]
                if arial_fonts:
                    font["path"] = arial_fonts[0]
                    font_prop = FontProperties(fname=font["path"])
                    print(f"成功加载系统字体: {font['name']}")
                    break
            
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
        print("警告: 所有中文字体加载失败，中文可能无法正常显示")
        return FontProperties()
    
    # 应用字体设置
    plt.rcParams["font.family"] = [font_prop.get_name(), "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Noto Sans CJK SC"]
    plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Noto Sans CJK SC"]
    plt.rcParams['axes.unicode_minus'] = False
    print("中文显示设置完成")
    
    return font_prop

# 测试中文字体显示
def test_chinese_font_display(chinese_font):
    """测试中文字体是否能正常显示，接收字体属性作为参数"""
    plt.figure(figsize=(8, 4))
    plt.title("中文字体测试 - 车牌识别系统", fontproperties=chinese_font)
    plt.xlabel("X轴标签（中文测试）", fontproperties=chinese_font)
    plt.ylabel("Y轴标签（中文测试）", fontproperties=chinese_font)
    plt.text(0.5, 0.5, "测试文字：车牌识别 ABC123", 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=12, fontproperties=chinese_font)
    plt.plot([1, 2, 3], [4, 5, 6], label="示例曲线")
    
    plt.legend(
        title="图例（中文）", 
        prop=chinese_font,
        title_fontproperties=chinese_font
    )
    
    test_img_path = os.path.join('/kaggle/working/', 'chinese_font_test.png')
    plt.savefig(test_img_path, dpi=300)
    print(f"中文字体测试图像已保存至: {test_img_path}")
    plt.close()
    
    return test_img_path

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

class LicensePlateDataset(Dataset):
    """车牌数据集类"""
    def __init__(self, image_dir, annotations, transform=None):
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # 获取图片路径和标签
        img_name, plate_number = self.annotations[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"图片 {img_path} 未找到")
            
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 将车牌字符转换为索引
        label = []
        for char in plate_number:
            if char in char_to_idx:
                label.append(char_to_idx[char])
            else:
                # 未知字符用0填充
                label.append(0)
        
        # 确保标签长度为7
        if len(label) < PLATE_LENGTH:
            label += [0] * (PLATE_LENGTH - len(label))
        elif len(label) > PLATE_LENGTH:
            label = label[:PLATE_LENGTH]
            
        return image, torch.tensor(label, dtype=torch.long)

class PlateRecognitionModel(nn.Module):
    """简化的车牌识别模型，减轻过拟合风险"""
    def __init__(self, num_classes, pretrained=True):
        super(PlateRecognitionModel, self).__init__()
        # 使用更轻量的MobileNetV2作为特征提取器
        if pretrained:
            self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.cnn = mobilenet_v2()
        
        # 获取MobileNetV2的特征提取部分
        self.features = self.cnn.features
        
        # MobileNetV2的输出通道是1280
        self.feature_channels = 1280
        
        # 进一步减小通道数
        self.conv1x1 = nn.Conv2d(self.feature_channels, 64, kernel_size=1)
        
        # 添加批归一化层
        self.bn = nn.BatchNorm2d(64)
        
        # 简化的循环神经网络部分
        self.lstm = nn.LSTM(
            input_size=64 * 4,  # 相应调整输入大小
            hidden_size=128,    # 进一步减小隐藏层大小
            num_layers=1,       # 减少层数
            bidirectional=True,
            batch_first=True,
            dropout=0.6  # 增加dropout防止过拟合
        )
        
        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 输出层
        self.fc = nn.Linear(256, num_classes)
        
        # 序列批归一化层
        self.seq_bn = nn.BatchNorm1d(256)
        
        # 增加dropout强度
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, x):
        # CNN特征提取
        batch_size = x.size(0)
        x = self.features(x)  # [batch_size, 1280, H, W]
        x = self.conv1x1(x)   # [batch_size, 64, H, W]
        x = self.bn(x)        # 批归一化
        x = F.relu(x)         # 激活函数
        
        # 调整形状以适应LSTM
        x = x.permute(0, 3, 2, 1)  # [batch_size, W, H, 64]
        x = x.reshape(batch_size, x.size(1), -1)  # [batch_size, W, H*64]
        
        # LSTM处理
        x, _ = self.lstm(x)  # [batch_size, W, 256]
        
        # 应用注意力机制
        attn_weights = self.attention(x)  # [batch_size, W, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, W, 1]
        x = x * attn_weights  # [batch_size, W, 256]
        
        # 对于车牌识别，我们只需要固定的7个输出
        if x.size(1) >= PLATE_LENGTH:
            x = x[:, -PLATE_LENGTH:, :]  # [batch_size, 7, 256]
        else:
            # 如果长度不足，进行填充
            pad_length = PLATE_LENGTH - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_length))  # [batch_size, 7, 256]
        
        # 应用序列批归一化和Dropout
        x = x.permute(0, 2, 1)  # [batch_size, 256, 7]
        x = self.seq_bn(x)      # [batch_size, 256, 7]
        x = x.permute(0, 2, 1)  # [batch_size, 7, 256]
        x = self.dropout(x)     # [batch_size, 7, 256]
        
        # 预测每个位置的字符
        x = self.fc(x)  # [batch_size, 7, num_classes]
        
        return x

class PlateRecognizer:
    """车牌识别器类，整合模型训练和预测功能"""
    def __init__(self, detection_model_path, data_config, crop_dir=None, 
                 img_size=(400, 100), batch_size=16, epochs=30,
                 patience=6, learning_rate=5e-5, chinese_font=None,
                 position_weights=None):  # 新增位置权重参数
        """初始化识别器，使用更适合的参数"""
        self.detection_model_path = detection_model_path
        self.data_config = data_config
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.chinese_font = chinese_font  # 保存中文字体属性
        
        # 设置字符位置权重，默认给后几位更高的权重
        if position_weights is None:
            # 根据用户提供的准确率数据设置权重，准确率低的位置权重更高
            # 原始准确率: ['0.9695', '0.9680', '0.7981', '0.8321', '0.8704', '0.8699', '0.8445']
            self.position_weights = torch.tensor([1.0, 1.0, 1.8, 1.6, 1.4, 1.4, 1.5])
        else:
            self.position_weights = torch.tensor(position_weights)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 将权重移动到相应设备
        self.position_weights = self.position_weights.to(self.device)
        
        # 检查GPU显存
        if torch.cuda.is_available():
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 创建保存裁剪后车牌的目录
        if crop_dir is None:
            self.crop_dir = os.path.join('/kaggle/working/', 'cropped_plates')
        else:
            self.crop_dir = crop_dir
        
        # 创建训练集和验证集目录
        self.train_crop_dir = os.path.join(self.crop_dir, 'train')
        self.val_crop_dir = os.path.join(self.crop_dir, 'val')
        os.makedirs(self.train_crop_dir, exist_ok=True)
        os.makedirs(self.val_crop_dir, exist_ok=True)
        
        # 加载数据配置
        with open(data_config, 'r') as f:
            self.data_info = yaml.safe_load(f)
        
        # 加载车牌信息
        self.plate_info = pd.read_csv('/kaggle/input/plate-recognition/data/plate_info.csv')
        
        # 加载检测模型
        self.detection_model = YOLO(detection_model_path)
        print(f"已加载车牌检测模型: {detection_model_path}")
        
        # 初始化识别模型
        self.model = PlateRecognitionModel(num_classes).to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # 改为none以便应用权重
        # 使用AdamW优化器，更强的权重衰减
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.005  # 增加权重衰减，减少过拟合
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=3, verbose=True
        )
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            # 新增每个位置的准确率记录
            'train_char_acc': [[] for _ in range(PLATE_LENGTH)],
            'val_char_acc': [[] for _ in range(PLATE_LENGTH)]
        }
        
        # 增强的数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet均值
                std=[0.229, 0.224, 0.225]    # ImageNet标准差
            ),
            # 更强的数据增强
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # 测试时的数据变换（无增强）
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 数据集和数据加载器
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # 最佳模型路径
        self.best_model_path = os.path.join('/kaggle/working/', 'best_mobilenet_v2.pt')

    def crop_all_plates(self):
        """裁剪所有图片中的车牌并保存"""
        # 更新训练集和验证集的图片路径
        train_images_dir = '/kaggle/input/plate-recognition/data/images/train'
        val_images_dir = '/kaggle/input/plate-recognition/data/images/val'
        # 更新标签路径
        self.data_info['labels'] = '/kaggle/input/plate-recognition/data/labels'
        
        # 处理训练集图片
        print("开始裁剪训练集车牌...")
        self._process_images(
            images_dir=train_images_dir,
            output_dir=self.train_crop_dir
        )
        
        # 处理验证集图片
        print("开始裁剪验证集车牌...")
        self._process_images(
            images_dir=val_images_dir,
            output_dir=self.val_crop_dir
        )
        
        print(f"所有车牌裁剪完成，保存至: {self.crop_dir}")

    def _process_images(self, images_dir, output_dir):
        """处理指定目录下的图片，裁剪车牌并保存"""
        # 获取所有图片文件
        image_files = [f for f in os.listdir(images_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 过滤出在plate_info.csv中存在的图片
        valid_files = []
        for f in image_files:
            if not self.plate_info[self.plate_info['image_name'] == f].empty:
                valid_files.append(f)
        
        print(f"找到 {len(valid_files)} 个有效的图片文件")
        
        # 处理每个图片
        for filename in tqdm(valid_files, desc=f"处理 {os.path.basename(images_dir)}"):
            image_path = os.path.join(images_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # 如果已经裁剪过，跳过
            if os.path.exists(output_path):
                continue
            
            # 检测并裁剪车牌
            cropped_plate, _ = self._detect_and_crop(image_path, filename)
            
            if cropped_plate is not None:
                # 保存裁剪后的车牌
                cv2.imwrite(output_path, cropped_plate)
            else:
                print(f"警告: 无法从 {filename} 中检测到车牌")

    def _detect_and_crop(self, image_path, image_name):
        """检测并裁剪车牌"""
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        # 使用检测模型检测车牌
        results = self.detection_model(image, verbose=False)
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None, None
            
        # 选择置信度最高的车牌
        best_idx = boxes.conf.argmax().item()
        box = boxes.xyxy[best_idx].cpu().numpy()
        
        x1, y1, x2, y2 = map(int, box)
        plate_region = image[y1:y2, x1:x2]
        
        # 尝试使用顶点信息进行矫正
        plate_row = self.plate_info[self.plate_info['image_name'] == image_name]
        
        if not plate_row.empty:
            try:
                vertices_str = plate_row['vertices'].values[0]
                vertices = eval(vertices_str)
                vertices_np = np.array(vertices, dtype=np.float32)
                
                corrected_plate = self._correct_skew(image, vertices_np)
                return corrected_plate, True
            except Exception as e:
                pass  # 矫正失败，使用原始裁剪
        
        # 如果矫正失败，使用原始裁剪并调整大小
        corrected_plate = cv2.resize(plate_region, self.img_size)
        return corrected_plate, False

    def _correct_skew(self, image, vertices):
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
        corrected_plate = cv2.resize(warped, self.img_size)
        
        return corrected_plate

    def prepare_datasets(self):
        """准备训练集和验证集"""
        # 获取训练集标注
        train_files = os.listdir(self.train_crop_dir)
        train_annotations = []
        for filename in train_files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                plate_row = self.plate_info[self.plate_info['image_name'] == filename]
                if not plate_row.empty:
                    plate_number = plate_row['plate_number'].values[0]
                    train_annotations.append((filename, plate_number))
        
        # 获取验证集标注
        val_files = os.listdir(self.val_crop_dir)
        val_annotations = []
        for filename in val_files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                plate_row = self.plate_info[self.plate_info['image_name'] == filename]
                if not plate_row.empty:
                    plate_number = plate_row['plate_number'].values[0]
                    val_annotations.append((filename, plate_number))
        
        print(f"训练集样本数: {len(train_annotations)}")
        print(f"验证集样本数: {len(val_annotations)}")
        
        # 检查数据集平衡情况
        if len(train_annotations) < 100:
            print("警告: 训练样本数量过少，可能导致过拟合")
        
        # 创建数据集
        self.train_dataset = LicensePlateDataset(
            self.train_crop_dir, 
            train_annotations, 
            self.transform
        )
        self.val_dataset = LicensePlateDataset(
            self.val_crop_dir, 
            val_annotations, 
            self.test_transform  # 验证集不使用数据增强
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )

    def train(self):
        """训练模型"""
        if self.train_loader is None or self.val_loader is None:
            self.prepare_datasets()
        
        # 早停机制变量
        best_val_acc = 0.0
        no_improve_epochs = 0
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"字符位置权重: {self.position_weights.cpu().numpy()}")
        print(f"开始训练，共 {self.epochs} 个epochs")
        
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_char_correct = [0] * PLATE_LENGTH
            train_char_total = [0] * PLATE_LENGTH
            
            # 使用tqdm显示进度条
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} - 训练")
            
            for images, labels in train_pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(images)  # [batch_size, 7, num_classes]
                
                # 计算加权损失
                loss = 0.0
                for i in range(PLATE_LENGTH):
                    # 计算每个位置的损失并应用权重
                    pos_loss = self.criterion(outputs[:, i, :], labels[:, i])
                    loss += (pos_loss * self.position_weights[i]).mean()
                
                # 平均损失（考虑权重）
                loss /= self.position_weights.sum()
                
                # 反向传播和优化
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 统计训练数据
                train_loss += loss.item() * images.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 2)  # [batch_size, 7]
                batch_total = labels.size(0) * PLATE_LENGTH
                batch_correct = (predicted == labels).sum().item()
                
                train_total += batch_total
                train_correct += batch_correct
                
                # 统计每个位置的准确率
                for i in range(PLATE_LENGTH):
                    train_char_total[i] += labels.size(0)
                    train_char_correct[i] += (predicted[:, i] == labels[:, i]).sum().item()
                
                # 更新进度条
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{batch_correct / batch_total:.4f}"
                })
            
            # 计算平均训练损失和准确率
            avg_train_loss = train_loss / len(self.train_dataset)
            train_acc = train_correct / train_total
            
            # 计算每个位置的训练准确率
            train_char_acc = [train_char_correct[i] / train_char_total[i] 
                             for i in range(PLATE_LENGTH)]
            # 记录每个位置的准确率
            for i in range(PLATE_LENGTH):
                self.history['train_char_acc'][i].append(train_char_acc[i])
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_char_correct = [0] * PLATE_LENGTH
            val_char_total = [0] * PLATE_LENGTH
            
            with torch.no_grad():
                val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} - 验证")
                for images, labels in val_pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(images)
                    
                    # 计算加权损失
                    loss = 0.0
                    for i in range(PLATE_LENGTH):
                        pos_loss = self.criterion(outputs[:, i, :], labels[:, i])
                        loss += (pos_loss * self.position_weights[i]).mean()
                    loss /= self.position_weights.sum()
                    
                    val_loss += loss.item() * images.size(0)
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs, 2)
                    batch_total = labels.size(0) * PLATE_LENGTH
                    batch_correct = (predicted == labels).sum().item()
                    
                    val_total += batch_total
                    val_correct += batch_correct
                    
                    # 统计每个位置的准确率
                    for i in range(PLATE_LENGTH):
                        val_char_total[i] += labels.size(0)
                        val_char_correct[i] += (predicted[:, i] == labels[:, i]).sum().item()
                    
                    # 更新进度条
                    val_pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{batch_correct / batch_total:.4f}"
                    })
            
            # 计算平均验证损失和准确率
            avg_val_loss = val_loss / len(self.val_dataset)
            val_acc = val_correct / val_total
            
            # 计算每个位置的验证准确率
            val_char_acc = [val_char_correct[i] / val_char_total[i] 
                           for i in range(PLATE_LENGTH)]
            # 记录每个位置的准确率
            for i in range(PLATE_LENGTH):
                self.history['val_char_acc'][i].append(val_char_acc[i])
            
            # 记录训练历史
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 打印 epoch 结果
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print("每个位置的准确率:")
            print(f"位置 0-6 (省份+字符): {['%.4f' % acc for acc in val_char_acc]}")
            
            # 学习率调度
            self.scheduler.step(avg_val_loss)
            
            # 早停机制检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.history['best_val_acc'] = best_val_acc
                no_improve_epochs = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"保存最佳模型至 {self.best_model_path}")
            else:
                no_improve_epochs += 1
                print(f"没有改进的epochs: {no_improve_epochs}/{self.patience}")
                if no_improve_epochs >= self.patience:
                    print(f"早停机制触发，在第 {epoch+1} 个epoch停止训练")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")
        
        # 保存训练历史
        history_path = os.path.join('/kaggle/working/', 'training_history_mobilenet_v2.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)
        print(f"训练历史已保存至 {history_path}")
        
        # 绘制训练历史
        self.plot_training_history()

    def plot_training_history(self):
        """绘制训练历史曲线，使用中文字体"""
        plt.figure(figsize=(15, 15))
        
        # 绘制损失曲线
        plt.subplot(3, 1, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('损失曲线', fontproperties=self.chinese_font)
        plt.xlabel('轮次', fontproperties=self.chinese_font)
        plt.ylabel('损失值', fontproperties=self.chinese_font)
        plt.legend(
            prop=self.chinese_font,
            title_fontproperties=self.chinese_font
        )
        
        # 绘制整体准确率曲线
        plt.subplot(3, 1, 2)
        plt.plot([acc * 100 for acc in self.history['train_acc']], label='训练准确率')
        plt.plot([acc * 100 for acc in self.history['val_acc']], label='验证准确率')
        plt.title('整体准确率曲线', fontproperties=self.chinese_font)
        plt.xlabel('轮次', fontproperties=self.chinese_font)
        plt.ylabel('准确率 (%)', fontproperties=self.chinese_font)
        plt.legend(
            prop=self.chinese_font,
            title_fontproperties=self.chinese_font
        )
        
        # 绘制每个位置的准确率曲线
        plt.subplot(3, 1, 3)
        for i in range(PLATE_LENGTH):
            plt.plot([acc * 100 for acc in self.history['val_char_acc'][i]], 
                     label=f'位置 {i}')
        plt.title('各位置准确率曲线', fontproperties=self.chinese_font)
        plt.xlabel('轮次', fontproperties=self.chinese_font)
        plt.ylabel('准确率 (%)', fontproperties=self.chinese_font)
        plt.legend(
            prop=self.chinese_font,
            title_fontproperties=self.chinese_font
        )
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join('/kaggle/working/', 'training_history_mobilenet_v2.png')
        plt.savefig(plot_path, dpi=300)
        print(f"训练历史图表已保存至 {plot_path}")
        
        plt.show()

    def predict_plate(self, image_path):
        """预测单张车牌图片的内容"""
        self.model.eval()
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # 转换为RGB并调整大小
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        
        # 应用变换
        image_tensor = self.test_transform(image).unsqueeze(0)  # 添加批次维度
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)  # [1, 7, num_classes]
            _, predicted = torch.max(output, 2)  # [1, 7]
        
        # 转换为字符
        predicted_chars = [idx_to_char[idx.item()] for idx in predicted[0]]
        return ''.join(predicted_chars)

    def test_random_samples(self, num_samples=5):
        """从验证集中随机抽取样本进行测试并显示结果，字符级对比"""
        if self.val_dataset is None:
            self.prepare_datasets()
        
        # 随机选择样本
        sample_indices = random.sample(range(len(self.val_dataset)), min(num_samples, len(self.val_dataset)))
        
        # 创建显示结果的图表，增加高度以容纳更大的字体
        plt.figure(figsize=(15, 6 * num_samples))
        
        for i, idx in enumerate(sample_indices):
            # 获取样本
            image, label = self.val_dataset[idx]
            img_name, true_plate = self.val_dataset.annotations[idx]
            
            # 转换为字符
            true_chars = true_plate
            
            # 预测
            img_path = os.path.join(self.val_crop_dir, img_name)
            pred_chars = self.predict_plate(img_path)
            
            # 显示图片和结果
            plt.subplot(num_samples, 1, i + 1)
            # 转换为适合显示的格式
            img_np = image.numpy().transpose(1, 2, 0)
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = (img_np * 255).astype(np.uint8)
            
            plt.imshow(img_np)
            plt.title(f"样本 {i+1}", fontproperties=self.chinese_font, fontsize=24)
            plt.axis('off')
            
            # 在图片下方添加字符对比
            ax = plt.gca()
            
            # 增大字体大小并居中显示
            # 真实车牌标签和字符
            plt.text(0.35, -0.15, "真实车牌: ", 
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax.transAxes, fontproperties=self.chinese_font, fontsize=28)
            
            # 逐个显示真实字符，居中排列
            for j, char in enumerate(true_chars):
                plt.text(0.4 + (j+1)*0.08, -0.15, char, 
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes, fontproperties=self.chinese_font,
                         color='black', fontsize=28)
            
            # 预测车牌标签和字符
            plt.text(0.35, -0.3, "预测车牌: ", 
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax.transAxes, fontproperties=self.chinese_font, fontsize=28)
            
            # 逐个显示预测字符，居中排列，并根据是否正确设置颜色
            for j, (t_char, p_char) in enumerate(zip(true_chars, pred_chars)):
                color = 'green' if t_char == p_char else 'red'
                plt.text(0.4 + (j+1)*0.08, -0.3, p_char, 
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes, fontproperties=self.chinese_font,
                         color=color, fontsize=28)
        
        plt.tight_layout()
        
        # 保存测试结果，确保包含所有文本
        test_result_path = os.path.join('/kaggle/working/', 'test_samples_mobilenet_v2.png')
        plt.savefig(test_result_path, dpi=300, bbox_inches='tight')
        print(f"测试样本结果已保存至 {test_result_path}")
        
        plt.show()

# 初始化中文字体
chinese_font = setup_chinese_font()

if __name__ == "__main__":
    # 配置路径
    detection_model_path = "/kaggle/input/car_plate_recognition/pytorch/default/3/best.pt"
    data_config = "/kaggle/input/plate-recognition/data/plate_data.yaml"
    
    # 可以根据需要调整位置权重
    # 例如: [1.0, 1.0, 2.0, 1.8, 1.6, 1.6, 1.7]
    custom_weights = [1.0, 1.0, 2.2, 2.0, 1.8, 1.8, 1.8]
    
    # 初始化识别器，使用更适合的参数，并传入中文字体
    recognizer = PlateRecognizer(
        detection_model_path=detection_model_path,
        data_config=data_config,
        batch_size=64,
        epochs=50,
        patience=8,
        learning_rate=5e-5,
        chinese_font=chinese_font,
        position_weights=custom_weights  # 传递位置权重
    )
    
    # 确保中文字体设置成功并测试
    test_chinese_font_display(chinese_font)
    
    # 裁剪所有车牌
    recognizer.crop_all_plates()
    
    # 准备数据集
    recognizer.prepare_datasets()
    
    # 训练模型
    recognizer.train()
    
    # 测试随机样本
    recognizer.test_random_samples(num_samples=5)