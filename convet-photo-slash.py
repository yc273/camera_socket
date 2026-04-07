import json
import cv2
import numpy as np
import os
import random

def process_single_jpg(jpg_file_path):
    """
    处理单个jpg文件：读取对应json标注、绘制斜线、覆盖原文件
    """
    # 1. 构造对应的json标注文件路径（默认和jpg文件同名，仅后缀不同）
    json_file_path = os.path.splitext(jpg_file_path)[0] + ".json"
    if not os.path.exists(json_file_path):
        print(f"警告：未找到{jpg_file_path}对应的标注文件{json_file_path}，跳过该文件")
        return
    
    # 2. 读取标注文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
    except Exception as e:
        print(f"错误：读取{json_file_path}失败 - {str(e)}，跳过该文件")
        return
    
    # 3. 提取交叉点坐标和目标区域
    cross_point = None
    target_area = None
    for shape in label_data['shapes']:
        if shape['label'] == 'CrossIntersection':
            cross_point = shape['points'][0]  # [x, y]
        elif shape['label'] == 'TargetArea':
            target_area = shape['points']     # [[x1,y1], [x2,y2]]
    
    if not cross_point or not target_area:
        print(f"警告：{json_file_path}中缺少CrossIntersection或TargetArea，跳过该文件")
        return
    
    # 4. 加载图片
    img = cv2.imread(jpg_file_path)
    if img is None:
        print(f"错误：无法加载图片{jpg_file_path}，跳过该文件")
        return
    
    # 5. 计算目标区域的边界
    x1, y1 = target_area[0]
    x2, y2 = target_area[1]
    cx, cy = cross_point

    # 获取交叉点附近小区域的平均颜色
    region_size = 3  # 定义小区域的大小，可以根据需要调整
    half_region = region_size // 2

    # 计算区域边界，确保不超出图像范围
    y_start = max(0, int(cy) - half_region)
    y_end = min(img.shape[0], int(cy) + half_region + 1)
    x_start = max(0, int(cx) - half_region)
    x_end = min(img.shape[1], int(cx) + half_region + 1)

    # 提取小区域并计算平均颜色
    region = img[y_start:y_end, x_start:x_end]
    center_color = np.mean(region, axis=(0, 1)).astype(int).tolist()  # BGR格式
    
    # 6. 计算斜线的起点和终点（45度，可调整）
    # 根据距离动态计算斜线长度系数和线宽
    distance = min(cx - x1, y2 - cy)
    line_coefficient = 0.5 + 0.2 * random.random()  # 随机系数在0.5-0.7之间
    dx = line_coefficient * distance

    # 根据距离调整线宽，距离越大线宽也相应增加
    thickness = max(2, int(distance * 0.05))  

    start_point = (int(cx - dx), int(cy + dx))
    end_point = (int(cx + dx), int(cy - dx))

    print(f"斜线起点：{start_point}，终点：{end_point}, dx:{dx}, 系数:{line_coefficient:.3f}, 线宽:{thickness}")
    # 7. 绘制斜线
    # 创建一个随机透明度因子
    visibility_factor = 0.5 + 0.5 * random.random()  # 可见度在0.5-1之间随机变化
    # 创建临时图像用于混合
    temp_img = img.copy()
    
    # 调整颜色的可见度
    adjusted_color = [int(color_val * visibility_factor) for color_val in center_color]
    adjusted_color = [max(0, min(255, val)) for val in adjusted_color]
    
    # 在临时图像上绘制线条
    cv2.line(temp_img, start_point, end_point, adjusted_color, thickness=thickness)
    
    # 将临时图像与原图按透明度混合
    alpha = visibility_factor  # 透明度参数
    img = cv2.addWeighted(img, 1 - alpha, temp_img, alpha, 0)

    
    # 8. 覆盖写入原jpg文件
    try:
        cv2.imwrite(jpg_file_path, img)
        print(f"成功处理并覆盖：{jpg_file_path}")
    except Exception as e:
        print(f"错误：无法覆盖{jpg_file_path} - {str(e)}，跳过该文件")

def batch_process_jpgs(target_dir):
    """
    批量处理指定目录下所有的jpg文件
    """
    # 校验目标目录是否存在
    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"目标目录不存在：{target_dir}")
    
    # 遍历目录下所有文件，筛选出jpg格式（支持.jpg和.JPG）
    for file_name in os.listdir(target_dir):
        if file_name.lower().endswith(".jpg"):
            jpg_file_full_path = os.path.join(target_dir, file_name)
            process_single_jpg(jpg_file_full_path)
    
    print("批量处理完成！")

# ----------------------
# 使用示例
# ----------------------
if __name__ == "__main__":
    # 请修改为你的目标目录（存放所有jpg和对应json文件的路径）
    TARGET_DIRECTORY = "E:/code/LabelMe/origin80-random-2/"
    
    try:
        batch_process_jpgs(TARGET_DIRECTORY)
    except Exception as e:
        print(f"批量处理失败：{str(e)}")