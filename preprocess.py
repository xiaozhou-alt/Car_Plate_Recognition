import os
import csv
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# 省份和字符映射表
provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

def parse_filename(filename):
    """解析CCPD文件名，提取相关信息"""
    parts = filename.split('-')
    
    # 提取边界框信息 (左上角和右下角)
    bbox_part = parts[2]
    bbox_coords = bbox_part.split('_')
    top_left = tuple(map(int, bbox_coords[0].split(',')))
    bottom_right = tuple(map(int, bbox_coords[1].split(',')))
    
    # 提取四个顶点坐标
    vertices_part = parts[3]
    vertices_coords = vertices_part.split('_')
    vertices = [tuple(map(int, coord.split(','))) for coord in vertices_coords]
    
    # 提取车牌号码信息
    plate_code = parts[4].split('_')
    province = provincelist[int(plate_code[0])]
    characters = [wordlist[int(code)] for code in plate_code[1:]]
    full_plate = province + ''.join(characters)
    
    return {
        'bbox': (top_left, bottom_right),
        'vertices': vertices,
        'plate_number': full_plate,
        'horizontal_angle': parts[1].split('_')[0],
        'vertical_angle': parts[1].split('_')[1]
    }

def convert_to_yolo_format(image_path, bbox, class_id=0):
    """将边界框转换为YOLO格式"""
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    height, width = image.shape[:2]
    
    # 边界框坐标
    (x1, y1), (x2, y2) = bbox
    
    # 计算中心点和宽高
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # 归一化
    center_x /= width
    center_y /= height
    bbox_width /= width
    bbox_height /= height
    
    return f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"

def create_yolo_dataset(original_image_dir, output_dir, sample_ratio=1.0):
    """创建YOLO格式的数据集和车牌信息CSV文件"""
    # 创建必要的目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 创建CSV文件存储车牌信息
    csv_path = os.path.join(output_dir, 'plate_info.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'plate_number', 'vertices']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(original_image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 按比例采样
        if sample_ratio < 1.0:
            sample_size = int(len(image_files) * sample_ratio)
            image_files = random.sample(image_files, sample_size)
        
        # 处理每个图片
        for filename in tqdm(image_files, desc="预处理数据"):
            try:
                # 解析文件名
                info = parse_filename(filename)
                
                # 复制图片到新目录
                src_path = os.path.join(original_image_dir, filename)
                dst_path = os.path.join(images_dir, filename)
                shutil.copy(src_path, dst_path)
                
                # 创建YOLO格式的标签文件
                yolo_label = convert_to_yolo_format(src_path, info['bbox'])
                if yolo_label:
                    label_filename = os.path.splitext(filename)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_filename)
                    with open(label_path, 'w') as f:
                        f.write(yolo_label)
                
                # 写入CSV文件
                writer.writerow({
                    'image_name': filename,
                    'plate_number': info['plate_number'],
                    'vertices': str(info['vertices'])
                })
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                continue
    
    return csv_path

def split_train_val(dataset_dir, train_ratio=0.8):
    """划分训练集和验证集"""
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    # 创建训练集和验证集目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)
    
    # 划分训练集和验证集
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # 移动文件
    for files, split in [(train_files, 'train'), (val_files, 'val')]:
        for filename in files:
            # 移动图片
            src_img = os.path.join(images_dir, filename)
            dst_img = os.path.join(images_dir, split, filename)
            shutil.move(src_img, dst_img)
            
            # 移动标签
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_filename)
            dst_label = os.path.join(labels_dir, split, label_filename)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
    
    # 创建YOLO所需的配置文件
    with open(os.path.join(dataset_dir, 'plate_data.yaml'), 'w') as f:
        f.write(f"train: {os.path.join(images_dir, 'train')}\n")
        f.write(f"val: {os.path.join(images_dir, 'val')}\n")
        f.write("nc: 1\n")  # 只有一个类别：车牌
        f.write("names: ['license_plate']\n")
    
    return os.path.join(dataset_dir, 'plate_data.yaml')

def correct_skew(image, vertices):
    """矫正倾斜的车牌，返回统一大小的车牌图像"""
    # 排序顶点以确保正确的顺序
    pts = np.array(vertices, dtype=np.float32)
    
    # 计算车牌的宽度和高度
    width1 = np.linalg.norm(pts[0] - pts[1])
    width2 = np.linalg.norm(pts[2] - pts[3])
    width = max(int(width1), int(width2))
    
    height1 = np.linalg.norm(pts[1] - pts[2])
    height2 = np.linalg.norm(pts[3] - pts[0])
    height = max(int(height1), int(height2))
    
    # 定义目标矩形
    dst = np.array([
        [0, height-1],
        [0, 0],
        [width-1, 0],
        [width-1, height-1]], dtype=np.float32)
    
    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    # 统一调整大小为 400x100
    target_size = (400, 100)
    resized = cv2.resize(warped, target_size)
    
    return resized

if __name__ == "__main__":
    # 配置路径
    original_dir = "/kaggle/input/ccpd-preprocess/CCPD2019/ccpd_base"  # 替换为你的CCPD_BASE目录
    output_dir = "/kaggle/working/data"
    
    # 设置样本比例（可以根据需要调整）
    sample_ratio = 0.2
    
    # 预处理数据
    csv_path = create_yolo_dataset(original_dir, output_dir, sample_ratio)
    print(f"数据预处理完成，车牌信息已保存至: {csv_path}")
    
    # 划分训练集和验证集
    data_yaml = split_train_val(output_dir)
    print(f"数据集划分完成，配置文件已保存至: {data_yaml}")