import os
import cv2
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
import yaml
from ultralytics import YOLO
import torch
import matplotlib.font_manager as fm
import requests
from pathlib import Path

def setup_chinese_font():
    """改进的中文字体设置函数，确保在各种环境下正常显示中文"""
    font_options = [
        {
            "name": "SimHei",
            "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",
            "path": os.path.expanduser("~/.fonts/SimHei.ttf")
        },
        {
            "name": "WenQuanYi Micro Hei",
            "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/wqy-microhei.ttc",
            "path": os.path.expanduser("~/.fonts/wqy-microhei.ttc")
        },
        {
            "name": "Microsoft YaHei",
            "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/msyh.ttc",
            "path": os.path.expanduser("~/.fonts/msyh.ttc")
        },
        {
            "name": "SimSun",
            "url": "https://github.com/StellarCN/scp_zh/raw/master/fonts/simsun.ttc",
            "path": os.path.expanduser("~/.fonts/simsun.ttc")
        },
        {
            "name": "Arial Unicode MS",
            "url": "",
            "path": ""
        }
    ]
    
    # 首先检查系统中已安装的中文字体
    system_fonts = fm.findSystemFonts()
    chinese_fonts = []
    for font_path in system_fonts:
        try:
            font_name = fm.get_font(font_path).family_name
            if any(name in font_name.lower() for name in 
                  ['heiti', 'simhei', 'microsoft yahei', 'simsun', 'wenquanyi', 'song']):
                chinese_fonts.append({"name": font_name, "path": font_path})
        except:
            continue
    
    if chinese_fonts:
        font_prop = FontProperties(fname=chinese_fonts[0]["path"])
        print(f"成功加载系统中文字体: {chinese_fonts[0]['name']}")
        
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", chinese_fonts[0]["name"]]
        plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", chinese_fonts[0]["name"]]
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
            
            if not os.path.exists(font["path"]):
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
    plt.rcParams["font.family"] = [font_prop.get_name(), "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False
    print("中文显示设置完成")
    
    return font_prop

# 初始化中文字体
chinese_font = setup_chinese_font()

class LicensePlateTrainer:
    def __init__(self, data_config, model_name='yolov8m.pt', img_size=640, batch_size=16, epochs=50, class_name='license_plate'):
        """初始化训练器"""
        self.data_config = data_config
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_name = class_name
        
        # 保存中文字体属性
        self.chinese_font = chinese_font
        
        # 创建输出目录用于保存测试样本图片
        self.output_dir = "test_samples_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"测试样本图片将保存至: {os.path.abspath(self.output_dir)}")
        
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {'GPU' if self.device == '0' else 'CPU'}")
        
        self.model = YOLO(model_name)
        
        with open(data_config, 'r') as f:
            self.data_info = yaml.safe_load(f)
        
        if 'names' not in self.data_info:
            print(f"警告: 数据配置文件中未找到 'names' 定义")
        else:
            names = self.data_info['names']
            if isinstance(names, list) and self.class_name not in names:
                print(f"警告: 数据配置文件的names列表中未找到 {self.class_name} 类别")
            elif isinstance(names, dict) and self.class_name not in names.values():
                print(f"警告: 数据配置文件的names字典中未找到 {self.class_name} 类别")
            elif not isinstance(names, (list, dict)):
                print(f"警告: 数据配置文件的names类型不支持（应为list或dict）")
        
        dataset_dir = os.path.dirname(data_config)
        self.plate_info = pd.read_csv(os.path.join(dataset_dir, 'plate_info.csv'))
        
        self.best_map = 0
        self.patience = 5
        self.no_improve_epochs = 0
        self.best_model_path = None

    def train(self):
        """训练模型"""
        names = self.data_info['names']
        if isinstance(names, list):
            class_idx = names.index(self.class_name)
        elif isinstance(names, dict):
            class_idx = list(names.values()).index(self.class_name)
        else:
            raise ValueError(f"不支持的names类型: {type(names)}")
        
        results = self.model.train(
            data=self.data_config,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            patience=self.patience,
            save=True,
            verbose=True,
            project='license_plate_detection',
            name='exp',
            classes=[class_idx]
        )
        
        self.best_model_path = os.path.join(
            'license_plate_detection', 
            'exp', 
            'weights', 
            'best.pt'
        )
        print(f"最佳模型已保存至: {self.best_model_path}")
        
        return results

    def load_best_model(self):
        """加载最佳模型"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model = YOLO(self.best_model_path)
            print(f"已加载最佳模型: {self.best_model_path}")
            return True
        else:
            try:
                exp_dirs = sorted(
                    [d for d in Path('license_plate_detection').glob('exp*') if d.is_dir()],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                if exp_dirs:
                    latest_exp = exp_dirs[0]
                    self.best_model_path = os.path.join(latest_exp, 'weights', 'best.pt')
                    if os.path.exists(self.best_model_path):
                        self.model = YOLO(self.best_model_path)
                        print(f"已加载最新最佳模型: {self.best_model_path}")
                        return True
            except Exception as e:
                print(f"自动查找模型失败: {e}")
                
        print("未找到最佳模型，使用初始模型")
        return False

    def evaluate(self):
        """在验证集上评估模型"""
        metrics = self.model.val(
            data=self.data_config,
            imgsz=self.img_size,
            device=self.device,
            verbose=True
        )
        
        mAP50 = metrics.box.map50
        mAP50_95 = metrics.box.map
        
        print(f"验证集评估结果:")
        print(f"mAP@0.5: {mAP50:.4f}")
        print(f"mAP@0.5:0.95: {mAP50_95:.4f}")
        
        if hasattr(metrics.box, 'classes'):
            for i, c in enumerate(metrics.box.classes):
                print(f"类别 {c} - 精确率: {metrics.box.p[i]:.4f}, 召回率: {metrics.box.r[i]:.4f}, mAP50: {metrics.box.map50[i]:.4f}")
        
        return {
            'mAP50': mAP50,
            'mAP50_95': mAP50_95
        }

    def detect_and_crop(self, image_path):
        """检测车牌并裁剪矫正"""
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, imgsz=self.img_size)
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None, None
            
        best_idx = boxes.conf.argmax().item()
        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = boxes.conf[best_idx].item()
        
        x1, y1, x2, y2 = map(int, box)
        plate_region = image[y1:y2, x1:x2]
        
        image_name = os.path.basename(image_path)
        plate_row = self.plate_info[self.plate_info['image_name'] == image_name]
        
        if not plate_row.empty:
            try:
                vertices_str = plate_row['vertices'].values[0]
                vertices = eval(vertices_str)
                vertices_np = np.array(vertices, dtype=np.float32)
                
                corrected_plate = correct_skew(image, vertices_np)
                return corrected_plate, conf
            except Exception as e:
                print(f"使用顶点信息矫正失败: {e}")
        
        try:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                angles = []
                for line in lines:
                    x1_line, y1_line, x2_line, y2_line = line[0]
                    angle = np.arctan2(y2_line - y1_line, x2_line - x1_line) * 180.0 / np.pi
                    angles.append(angle)
                
                median_angle = np.median(angles)
                
                if abs(median_angle) > 5:
                    rows, cols = plate_region.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                    plate_region = cv2.warpAffine(plate_region, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            print(f"基于边界框的矫正失败: {e}")
        
        corrected_plate = cv2.resize(plate_region, (400, 100))
        return corrected_plate, conf

    def visualize_samples(self, num_samples=5, save_individual=True):
        """可视化验证集上的样本检测结果并保存图片
        Args:
            num_samples: 要展示的样本数量
            save_individual: 是否同时保存单个车牌图片
        """
        val_images_dir = self.data_info['val']
        if not os.path.isabs(val_images_dir):
            val_images_dir = os.path.join(os.path.dirname(self.data_config), val_images_dir)
            
        image_files = [f for f in os.listdir(val_images_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"警告: 在验证集目录 {val_images_dir} 中未找到图像文件")
            return
        
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        # 创建画布
        fig = plt.figure(figsize=(15, 3 * num_samples))
        gs = GridSpec(num_samples, 2, figure=fig)
        
        for i, filename in enumerate(samples):
            image_path = os.path.join(val_images_dir, filename)
            cropped_plate, conf = self.detect_and_crop(image_path)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            plate_row = self.plate_info[self.plate_info['image_name'] == filename]
            true_plate = plate_row['plate_number'].values[0] if not plate_row.empty else "未知"
            
            # 绘制原始图像
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(image_rgb)
            ax1.set_title(f"原始图像: {filename}", fontproperties=self.chinese_font)
            ax1.axis('off')
            
            # 绘制裁剪后的车牌
            ax2 = fig.add_subplot(gs[i, 1])
            if cropped_plate is not None:
                cropped_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
                ax2.imshow(cropped_rgb)
                ax2.set_title(
                    f"Plate (Confidence: {conf:.2f})", 
                    fontproperties=self.chinese_font
                )
                
                # 如果需要保存单个车牌图片
                if save_individual:
                    individual_filename = f"plate_{os.path.splitext(filename)[0]}.png"
                    individual_path = os.path.join(self.output_dir, individual_filename)
                    cv2.imwrite(individual_path, cropped_plate)
                    print(f"已保存单个车牌图片: {individual_path}")
            else:
                ax2.text(0.5, 0.5, "未检测到车牌", ha='center', va='center', fontproperties=self.chinese_font)
            ax2.axis('off')
        
        plt.tight_layout()
        
        # 生成带时间戳的文件名，避免覆盖
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_filename = f"test_samples_{num_samples}_{timestamp}.png"
        save_path = os.path.join(self.output_dir, save_filename)
        
        # 保存组合图片，设置高DPI确保清晰度
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存测试样本组合图片: {save_path}")
        
        # 显示图片
        plt.show()
        # 关闭图形释放资源
        plt.close(fig)

def correct_skew(image, vertices):
    """改进的车牌倾斜矫正算法"""
    if len(vertices) != 4:
        rect = cv2.minAreaRect(vertices)
        vertices = cv2.boxPoints(rect)
    
    center = np.mean(vertices, axis=0)
    
    def get_angle(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0]) * 180 / np.pi
    
    vertices = sorted(vertices, key=get_angle)
    
    width_top = np.linalg.norm(vertices[0] - vertices[1])
    width_bottom = np.linalg.norm(vertices[2] - vertices[3])
    width = int((width_top + width_bottom) / 2)
    
    height_left = np.linalg.norm(vertices[0] - vertices[3])
    height_right = np.linalg.norm(vertices[1] - vertices[2])
    height = int((height_left + height_right) / 2)
    
    # 确保宽大于高（车牌特征）
    if width < height:
        width, height = height, width
        vertices = [vertices[1], vertices[2], vertices[3], vertices[0]]
    
    # 强制设置宽高比为4:1（中国车牌标准）
    if width / height < 3.5:
        width = int(height * 4)
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(np.array(vertices, dtype=np.float32), dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    # 通过HSV颜色空间检测蓝色区域判断是否需要翻转
    if height > 0 and width > 0:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        # 蓝色的HSV范围
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_area = cv2.countNonZero(blue_mask)
        
        # 翻转后检测蓝色区域
        flipped = cv2.flip(warped, 0)
        flipped_hsv = cv2.cvtColor(flipped, cv2.COLOR_BGR2HSV)
        flipped_blue_mask = cv2.inRange(flipped_hsv, lower_blue, upper_blue)
        flipped_blue_area = cv2.countNonZero(flipped_blue_mask)
        
        # 如果翻转后的蓝色区域明显更大，说明需要翻转
        if flipped_blue_area > blue_area * 1.2:
            warped = flipped
    
    # 统一调整为目标尺寸
    target_size = (400, 100)
    resized = cv2.resize(warped, target_size)
    
    return resized

if __name__ == "__main__":
    data_config = "/kaggle/working/data/plate_data.yaml"  # 替换为你的数据配置文件路径
    img_size = 640
    batch_size = 16
    class_name = "license_plate"
    
    trainer = LicensePlateTrainer(
        data_config=data_config,
        model_name='yolov8m.pt',
        img_size=img_size,
        batch_size=batch_size,
        epochs=10,
        class_name=class_name
    )
    
    print("开始训练模型...")
    training_results = trainer.train()
    
    load_success = trainer.load_best_model()
    if not load_success:
        print("无法加载最佳模型，评估可能不准确")
    
    print("在验证集上评估模型...")
    eval_metrics = trainer.evaluate()
    
    print("展示并保存验证集样本检测结果...")
    # 保存组合图片和单个车牌图片
    trainer.visualize_samples(num_samples=5, save_individual=True)