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
import sys
import time
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                            QMessageBox, QFrame, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# 全局异常处理
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    print(f"未捕获的异常: {exc_type} {exc_value}")
    traceback.print_tb(exc_traceback)
    QMessageBox.critical(None, "程序崩溃", f"发生未预期的错误:\n{exc_value}\n\n请查看控制台获取详细信息")

sys.excepthook = handle_exception


class ModelLoadingThread(QThread):
    """模型加载线程，避免阻塞UI"""
    finished = pyqtSignal(bool, str)
    
    def __init__(self, detection_model_path, recognition_model_path, chinese_font):
        super().__init__()
        self.detection_model_path = detection_model_path
        self.recognition_model_path = recognition_model_path
        self.chinese_font = chinese_font
        self.recognizer = None
        
    def run(self):
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.detection_model_path):
                raise FileNotFoundError(f"未找到检测模型文件: {self.detection_model_path}")
            
            if not os.path.exists(self.recognition_model_path):
                raise FileNotFoundError(f"未找到识别模型文件: {self.recognition_model_path}")
            
            # 创建识别器实例
            self.recognizer = LicensePlateRecognizer(
                detection_model_path=self.detection_model_path,
                recognition_model_path=self.recognition_model_path,
                chinese_font=self.chinese_font
            )
            self.finished.emit(True, "模型加载成功")
        except Exception as e:
            self.finished.emit(False, str(e))


class RecognitionThread(QThread):
    """车牌识别线程，避免UI冻结"""
    finished = pyqtSignal(str, float)
    error = pyqtSignal(str)
    update_images = pyqtSignal(object, object)
    
    def __init__(self, recognizer, image_path):
        super().__init__()
        self.recognizer = recognizer
        self.image_path = image_path
        
    def run(self):
        try:
            # 调用识别器处理图片
            plate_number = self.recognizer.process_image(
                self.image_path, 
                save_result=False,
                show_plot=False
            )
            
            # 获取识别器处理的结果
            plate_image = self.recognizer.last_plate_image
            marked_image = self.recognizer.last_marked_image
            confidence = self.recognizer.last_confidence
            
            # 发送信号更新UI
            self.update_images.emit(marked_image, plate_image)
            self.finished.emit(plate_number, confidence)
            
        except Exception as e:
            self.error.emit(str(e))


class LicensePlateGUI(QMainWindow):
    """基于PyQt5的车牌识别系统界面"""
    def __init__(self):
        super().__init__()
        
        # 标志位，确保UI初始化完成
        self.ui_initialized = False
        
        # 设置中文字体
        self.setup_chinese_font()
        
        # 模型路径 - 请替换为你的模型实际路径
        self.detection_model_path = "./output/model/best.pt"       # 车牌检测模型
        self.recognition_model_path = "./output/model/best_mobilenet.pt"  # 车牌识别模型
        
        # 初始化识别器
        self.recognizer = None
        
        # 存储图片路径和数据
        self.image_path = None
        self.original_image = None
        self.plate_image = None
        self.marked_image = None
        self.plate_number = None
        self.confidence = None
        
        # 初始化UI
        self.init_ui()
        
        # 标记UI已初始化
        self.ui_initialized = True
        
        # 启动模型加载线程
        self.load_models()
    
    def setup_chinese_font(self):
        """设置中文字体，确保中文正常显示"""
        # 设置PyQt5字体
        font = QFont()
        font.setFamily("SimHei")
        QApplication.instance().setFont(font)
        
        # 设置matplotlib字体
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
            self.chinese_font = FontProperties(fname=simhei_fonts[0]["path"])
            print(f"成功加载系统中文字体: {simhei_fonts[0]['name']}")
            
            # 设置matplotlib字体
            plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
            plt.rcParams['axes.unicode_minus'] = False
            return
        
        # 创建字体目录
        font_dir = os.path.expanduser("~/.fonts")
        Path(font_dir).mkdir(parents=True, exist_ok=True)
        
        # 尝试下载并加载SimHei字体
        for font in font_options:
            try:
                if not os.path.exists(font["path"]):
                    print(f"正在下载 {font['name']} 字体...")
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    # SimHei字体下载链接
                    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
                    response = requests.get(font_url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    with open(font["path"], "wb") as f:
                        f.write(response.content)
                    
                    if os.path.getsize(font["path"]) < 1024 * 100:
                        raise Exception("字体文件不完整")
                
                if os.path.exists(font["path"]):
                    self.chinese_font = FontProperties(fname=font["path"])
                    if fm.findfont(self.chinese_font):
                        print(f"成功加载字体: {font['name']}")
                        break
                    
            except Exception as e:
                print(f"加载 {font['name']} 失败: {str(e)}")
                continue
        else:
            print("警告: SimHei字体加载失败，中文可能无法正常显示")
            self.chinese_font = FontProperties()
        
        # 应用字体设置
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams['axes.unicode_minus'] = False
        print("中文显示设置完成")
    
    def load_models(self):
        """加载模型（在单独线程中）"""
        # 更新UI显示加载状态
        self.result_label.setText("正在加载模型，请稍候...")
        
        # 创建并启动模型加载线程
        self.model_loading_thread = ModelLoadingThread(
            self.detection_model_path,
            self.recognition_model_path,
            self.chinese_font
        )
        self.model_loading_thread.finished.connect(self.on_model_loaded)
        self.model_loading_thread.start()
    
    def on_model_loaded(self, success, message):
        """模型加载完成回调"""
        if success:
            self.recognizer = self.model_loading_thread.recognizer
            self.result_label.setText("模型加载完成，请选择图片开始识别")
            QMessageBox.information(self, "初始化成功", "车牌识别模型加载成功，可以开始使用")
            # 启用选择图片按钮（修复按钮无法点击问题）
            self.select_btn.setEnabled(True)
        else:
            self.result_label.setText("模型加载失败")
            QMessageBox.critical(self, "初始化失败", f"模型加载出错: {message}")
            print(f"初始化错误: {message}")
    
    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("智能车牌识别系统")
        self.setGeometry(100, 100, 1000, 700)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部标题
        title_label = QLabel("智能车牌识别系统")
        title_font = QFont("SimHei", 18, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #165DFF; margin: 10px 0px;")
        main_layout.addWidget(title_label)
        
        # 操作按钮区域
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 0, 10, 10)
        
        self.select_btn = QPushButton("选择图片")
        self.select_btn.setFont(QFont("SimHei", 12))
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #165DFF;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0E42D2;
            }
            QPushButton:disabled {
                background-color: #A9CFC9;
            }
        """)
        self.select_btn.clicked.connect(self.select_image)
        self.select_btn.setEnabled(False)  # 初始禁用，模型加载后启用
        button_layout.addWidget(self.select_btn)
        button_layout.addSpacing(20)
        
        self.recognize_btn = QPushButton("开始识别")
        self.recognize_btn.setFont(QFont("SimHei", 12))
        self.recognize_btn.setStyleSheet("""
            QPushButton {
                background-color: #36CFC9;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2AAFA6;
            }
            QPushButton:disabled {
                background-color: #A9CFC9;
            }
        """)
        self.recognize_btn.clicked.connect(self.start_recognition)
        self.recognize_btn.setEnabled(False)
        button_layout.addWidget(self.recognize_btn)
        button_layout.addSpacing(20)
        
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setFont(QFont("SimHei", 12))
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #6B7280;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5A6270;
            }
            QPushButton:disabled {
                background-color: #A9ADB4;
            }
        """)
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # 文件路径显示
        self.file_path_label = QLabel("未选择图片文件")
        self.file_path_label.setFont(QFont("SimHei", 10))
        self.file_path_label.setStyleSheet("color: #6B7280; margin-bottom: 10px;")
        self.file_path_label.setWordWrap(True)
        main_layout.addWidget(self.file_path_label)
        
        # 主要内容区域 - 使用QSplitter实现可调整大小的分割窗口
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：原始图片展示
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        left_title = QLabel("原始图片")
        left_title.setFont(QFont("SimHei", 12, QFont.Bold))
        left_title.setStyleSheet("margin-bottom: 5px;")
        left_layout.addWidget(left_title)
        
        self.original_image_frame = QFrame()
        self.original_image_frame.setFrameShape(QFrame.StyledPanel)
        self.original_image_frame.setStyleSheet("border: 2px solid #D1D5DB; background-color: #F9FAFB;")
        self.original_image_layout = QVBoxLayout(self.original_image_frame)
        
        self.original_image_label = QLabel("请选择一张包含车牌的图片")
        self.original_image_label.setFont(QFont("SimHei", 10))
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setWordWrap(True)
        self.original_image_layout.addWidget(self.original_image_label)
        
        left_layout.addWidget(self.original_image_frame, 1)
        splitter.addWidget(left_widget)
        
        # 右侧：结果展示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 右侧上方：裁剪后的车牌
        top_right_layout = QVBoxLayout()
        
        top_right_title = QLabel("检测到的车牌")
        top_right_title.setFont(QFont("SimHei", 12, QFont.Bold))
        top_right_title.setStyleSheet("margin-bottom: 5px;")
        top_right_layout.addWidget(top_right_title)
        
        self.plate_image_frame = QFrame()
        self.plate_image_frame.setFrameShape(QFrame.StyledPanel)
        self.plate_image_frame.setStyleSheet("border: 2px solid #D1D5DB; background-color: #F9FAFB;")
        self.plate_image_layout = QVBoxLayout(self.plate_image_frame)
        
        self.plate_image_label = QLabel("车牌将显示在这里")
        self.plate_image_label.setFont(QFont("SimHei", 10))
        self.plate_image_label.setAlignment(Qt.AlignCenter)
        self.plate_image_layout.addWidget(self.plate_image_label)
        
        top_right_layout.addWidget(self.plate_image_frame, 1)
        right_layout.addLayout(top_right_layout, 1)
        
        # 右侧下方：识别结果
        bottom_right_layout = QVBoxLayout()
        bottom_right_layout.setContentsMargins(0, 10, 0, 0)
        
        bottom_right_title = QLabel("识别结果")
        bottom_right_title.setFont(QFont("SimHei", 12, QFont.Bold))
        bottom_right_title.setStyleSheet("margin-bottom: 5px;")
        bottom_right_layout.addWidget(bottom_right_title)
        
        self.result_frame = QFrame()
        self.result_frame.setFrameShape(QFrame.StyledPanel)
        self.result_frame.setStyleSheet("border: 2px solid #D1D5DB; background-color: #F5F7FA;")
        self.result_layout = QVBoxLayout(self.result_frame)
        
        self.result_label = QLabel("正在初始化程序...")
        self.result_label.setFont(QFont("SimHei", 10))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_layout.addWidget(self.result_label)
        
        bottom_right_layout.addWidget(self.result_frame, 1)
        
        # 识别结果显示框
        self.result_display_frame = QFrame()
        self.result_display_frame.setStyleSheet("background-color: #F0F4F8; margin-top: 10px;")
        self.result_display_layout = QVBoxLayout(self.result_display_frame)
        
        self.plate_result_label = QLabel("")
        self.plate_result_label.setFont(QFont("SimHei", 24, QFont.Bold))
        self.plate_result_label.setStyleSheet("color: #165DFF; margin: 10px 0px;")
        self.plate_result_label.setAlignment(Qt.AlignCenter)
        self.result_display_layout.addWidget(self.plate_result_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("SimHei", 10))
        self.confidence_label.setStyleSheet("color: #6B7280;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.result_display_layout.addWidget(self.confidence_label)
        
        bottom_right_layout.addWidget(self.result_display_frame)
        
        right_layout.addLayout(bottom_right_layout, 1)
        
        splitter.addWidget(right_widget)
        
        # 设置分割器初始大小
        splitter.setSizes([500, 500])
        main_layout.addWidget(splitter, 1)
        
        # 添加水印
        self.add_watermark()
        
        # 显示窗口
        self.show()
    
    def add_watermark(self):
        """添加水印"""
        self.watermark = QLabel("小周出品，必属精品")
        self.watermark.setFont(QFont("SimHei", 10))
        self.watermark.setStyleSheet("color: #D1D5DB;")
        self.watermark.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.watermark.setContentsMargins(0, 0, 20, 10)
        
        # 将水印添加到主布局
        main_layout = self.centralWidget().layout()
        main_layout.addWidget(self.watermark)
    
    def select_image(self):
        """选择图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        
        if file_path:
            self.image_path = file_path
            self.file_path_label.setText(f"已选择: {os.path.basename(file_path)}")
            
            # 显示原始图片
            self.display_image(file_path, self.original_image_label, self.original_image_frame)
            
            # 重置结果区域
            self.reset_result_area()
            
            # 启用识别按钮
            self.recognize_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
    
    def display_image(self, image_path, label, frame):
        """在指定标签中显示图片"""
        try:
            # 加载图片并调整大小
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                raise Exception("无法加载图片")
            
            # 保持比例缩放
            scaled_pixmap = pixmap.scaled(
                frame.width() - 20, frame.height() - 20, 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # 显示图片
            label.setPixmap(scaled_pixmap)
            label.setText("")
            
            # 保存原始图像（修复颜色问题：不转换颜色空间，保持BGR）
            if label == self.original_image_label:
                self.original_image = cv2.imread(image_path)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示图片: {str(e)}")
            print(f"显示图片错误: {str(e)}")
    
    def reset_result_area(self):
        """重置结果区域"""
        self.plate_image_label.setText("车牌将显示在这里")
        self.plate_image_label.setPixmap(QPixmap())
        self.result_label.setText("请点击开始识别按钮")
        self.plate_result_label.setText("")
        self.confidence_label.setText("")
        self.plate_image = None
        self.marked_image = None
        self.plate_number = None
        self.confidence = None
    
    def start_recognition(self):
        """开始识别车牌"""
        if not self.image_path or not self.recognizer:
            return
        
        # 显示加载状态
        self.result_label.setText("正在识别车牌，请稍候...")
        self.recognize_btn.setEnabled(False)
        
        # 创建并启动识别线程
        self.recognition_thread = RecognitionThread(self.recognizer, self.image_path)
        self.recognition_thread.finished.connect(self.on_recognition_finished)
        self.recognition_thread.error.connect(self.on_recognition_error)
        self.recognition_thread.update_images.connect(self.on_images_updated)
        self.recognition_thread.start()
    
    def on_images_updated(self, marked_image, plate_image):
        """更新图片显示"""
        self.marked_image = marked_image
        self.plate_image = plate_image
        
        # 显示带检测框的原图
        if self.marked_image is not None:
            self.display_cv2_image(self.marked_image, self.original_image_label, self.original_image_frame)
        
        # 显示裁剪后的车牌
        if self.plate_image is not None:
            self.display_cv2_image(self.plate_image, self.plate_image_label, self.plate_image_frame)
    
    def on_recognition_finished(self, plate_number, confidence):
        """识别完成回调"""
        self.plate_number = plate_number
        self.confidence = confidence
        
        if not self.plate_number:
            self.result_label.setText("未检测到车牌")
            self.recognize_btn.setEnabled(True)
            return
        
        # 更新结果显示
        self.result_label.setText("识别完成")
        self.plate_result_label.setText(self.plate_number)
        
        # 显示置信度
        if self.confidence is not None:
            self.confidence_label.setText(f"识别置信度: {self.confidence:.2f}")
        
        # 启用按钮
        self.recognize_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
    
    def on_recognition_error(self, error_msg):
        """识别错误回调"""
        QMessageBox.critical(self, "识别错误", f"识别过程出错: {error_msg}")
        self.result_label.setText("识别失败")
        self.recognize_btn.setEnabled(True)
        print(f"识别错误: {error_msg}")
    
    def display_cv2_image(self, cv2_image, label, frame):
        """显示OpenCV格式的图片（修复颜色反转问题）"""
        try:
            # 转换颜色空间：确保从BGR正确转换为RGB
            if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2_image  # 灰度图不需要转换
            
            # 转换为QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = channel * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 调整大小
            scaled_q_image = q_image.scaled(
                frame.width() - 20, frame.height() - 20, 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # 显示图片
            label.setPixmap(QPixmap.fromImage(scaled_q_image))
            label.setText("")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法显示处理后的图片: {str(e)}")
            print(f"显示处理后图片错误: {str(e)}")
    
    def save_result(self):
        """保存识别结果"""
        if not self.plate_number or self.marked_image is None or self.plate_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的识别结果")
            return
        
        try:
            # 额外检查图像数组是否为空
            if self.marked_image.size == 0 or self.plate_image.size == 0:
                QMessageBox.warning(self, "警告", "图像数据为空，无法保存")
                return
                
            # 创建保存目录
            result_dir = "recognition_results"
            os.makedirs(result_dir, exist_ok=True)
            
            # 生成文件名
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_filename = f"{base_name}_{timestamp}"
            
            # 保存带检测框的原图
            # 注意：self.marked_image是BGR格式，cv2.imwrite需要BGR格式，无需转换
            marked_image_path = os.path.join(result_dir, f"{result_filename}_marked.jpg")
            cv2.imwrite(marked_image_path, self.marked_image)
            
            # 保存裁剪的车牌
            # 注意：self.plate_image是BGR格式，cv2.imwrite需要BGR格式，无需转换
            plate_image_path = os.path.join(result_dir, f"{result_filename}_plate.jpg")
            cv2.imwrite(plate_image_path, self.plate_image)
            
            # 保存识别结果文本
            result_text_path = os.path.join(result_dir, f"{result_filename}_result.txt")
            with open(result_text_path, "w", encoding="utf-8") as f:
                f.write(f"车牌号码: {self.plate_number}\n")
                if self.confidence is not None:
                    f.write(f"识别置信度: {self.confidence:.2f}\n")
                f.write(f"识别时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"原始图片: {self.image_path}")
            
            QMessageBox.information(self, "保存成功", f"识别结果已保存至:\n{result_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存结果时出错: {str(e)}")
            print(f"保存结果错误: {str(e)}")


# 车牌识别相关代码
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
        self.cnn = mobilenet_v2(weights=None)  # 不加载预训练权重
        self.features = self.cnn.features
        self.feature_channels = 1280
        
        # 减小通道数
        self.conv1x1 = nn.Conv2d(self.feature_channels, 64, kernel_size=1)
        self.bn = nn.BatchNorm2d(64)
        
        # LSTM部分 - 修改dropout参数以消除警告
        self.lstm = nn.LSTM(
            input_size=64 * 4,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0  # 当num_layers=1时设置为0以消除警告
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
        
        # 存储最后处理的结果
        self.last_plate_image = None
        self.last_marked_image = None
        self.last_confidence = None
        
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
        """检测并裁剪车牌（修复颜色问题）"""
        # 读取图片（保持BGR格式）
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        
        # 检测车牌
        results = self.detection_model(image, verbose=False)
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            # 返回BGR格式的原始图像
            return None, image.copy(), None
        
        # 选择置信度最高的车牌
        best_idx = boxes.conf.argmax().item()
        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = boxes.conf[best_idx].item()
        
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制边界框（使用英文标注，避免中文显示问题）
        marked_image = image.copy()
        cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            marked_image, 
            f"Plate: {conf:.2f}",  # 将"车牌"改为"Plate"
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
            # 使用顶点进行矫正
            vertices = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            corrected_plate = correct_skew(image, vertices)
        except:
            # 矫正失败，使用原始裁剪并调整大小
            corrected_plate = cv2.resize(plate_region, self.img_size)
        
        # 保持BGR格式，不在此处转换为RGB
        return corrected_plate, marked_image, conf
    
    def recognize_plate(self, plate_image):
        """识别车牌内容"""
        # 应用变换
        image = cv2.resize(plate_image, self.img_size)
        # 转换为RGB用于模型输入
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.test_transform(image_rgb).unsqueeze(0)  # 添加批次维度
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.recognition_model(image_tensor)
            _, predicted = torch.max(output, 2)
        
        # 转换为字符
        predicted_chars = [idx_to_char[idx.item()] for idx in predicted[0]]
        return ''.join(predicted_chars)
    
    def process_image(self, image_path, save_result=True, show_plot=True):
        """处理单张图片，检测并识别车牌"""
        # 检测车牌（返回BGR格式）
        plate_image, marked_image, conf = self.detect_plate(image_path)
        
        # 保存结果供UI使用（保持BGR格式，由UI负责转换）
        self.last_plate_image = plate_image
        self.last_marked_image = marked_image
        self.last_confidence = conf
        
        if plate_image is None:
            print("未检测到车牌")
            if show_plot:
                # 显示时需要转换为RGB
                plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
                plt.title("No plate detected", fontproperties=self.chinese_font)  # 改为英文
                plt.axis('off')
                plt.show()
            return None
        
        # 识别车牌
        plate_number = self.recognize_plate(plate_image)
        print(f"识别结果: {plate_number} (检测置信度: {conf:.2f})")
        
        # 显示结果
        if show_plot:
            plt.figure(figsize=(12, 6))
            
            # 原始图片带检测框（转换为RGB显示）
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
            plt.title("Original image with detection result", fontproperties=self.chinese_font)  # 改为英文
            plt.axis('off')
            
            # 裁剪后的车牌与识别结果（转换为RGB显示）
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Recognition result: {plate_number}", fontproperties=self.chinese_font)  # 改为英文
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
    print("程序开始启动...")
    # 创建应用实例
    app = QApplication(sys.argv)
    
    # 确保中文显示正常
    font = QFont("SimHei")
    app.setFont(font)
    
    # 创建并显示主窗口
    print("初始化主窗口...")
    window = LicensePlateGUI()
    
    # 进入应用主循环
    print("进入事件循环...")
    sys.exit(app.exec_())
    