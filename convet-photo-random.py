import json
import cv2
import numpy as np
import os
import random
import shutil

# 全局参数（可按需调整，避免硬编码）
CONFIG = {
    "num_regions": (3, 6),       # 减少光照区域数量，避免叠加过暗
    "region_size_ratio": 0.5,   # 光照区域最大占比（更小，15%）
    "min_region_size": 50,       # 最小区域尺寸
    "brightness_range": (0.5, 1.5),  # 温和明暗范围，彻底杜绝极端暗
    "blur_size": (25, 50),       # 适中模糊核，边缘自然不生硬
    "hull_point_count": (5, 20),  # 凸包点数量，形状简单稳定
    "min_distance_from_boundary": 50,  # 缝隙距边界最小距离（像素）
    "min_distance_from_cross": 50,      # 缝隙距十字交叉点的最小距离（像素）
    "gap_thickness_range": (10, 20)       # 缝隙厚度范围（像素）
}

def generate_grid_points(img_w, img_h, num_regions):
    """
    在图片上生成相对均匀分布的点
    """
    # 计算网格大小
    grid_cols = int(np.ceil(np.sqrt(num_regions)))
    grid_rows = int(np.ceil(num_regions / grid_cols))
    
    # 计算每个网格的尺寸
    cell_width = img_w // grid_cols
    cell_height = img_h // grid_rows
    
    points = []
    for i in range(num_regions):
        # 确定网格位置
        col = i % grid_cols
        row = i // grid_cols
        
        # 在网格内随机选择一个点
        x_min = col * cell_width
        x_max = min((col + 1) * cell_width, img_w)
        y_min = row * cell_height
        y_max = min((row + 1) * cell_height, img_h)
        
        # 在网格单元内随机选择一个点
        cx = random.randint(x_min, x_max - 1)
        cy = random.randint(y_min, y_max - 1)
        
        points.append((cx, cy))
    
    return points

def get_cross_intersection(json_file_path):
    """
    从JSON文件中获取十字交叉点坐标
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        cross_point = None
        for shape in label_data['shapes']:
            if shape['label'] == 'CrossIntersection':
                cross_point = shape['points'][0]  # [x, y]
                break
        
        return cross_point
    except Exception as e:
        print(f"错误：读取{json_file_path}失败 - {str(e)}")
        return None

def draw_gap_line(img, gap_type, position, thickness, cross_point=None):
    """
    在图片上绘制缝隙线
    """
    img_h, img_w = img.shape[:2]
    
    # 检查是否与十字交叉点冲突
    if cross_point:
        cx, cy = cross_point
        if gap_type == 'horizontal':  # 水平线
            # 检查水平线是否与十字交叉点垂直距离太近
            if abs(position - cy) < CONFIG["min_distance_from_cross"]:
                return False
        else:  # 垂直线
            # 检查垂直线是否与十字交叉点水平距离太近
            if abs(position - cx) < CONFIG["min_distance_from_cross"]:
                return False
    
    # 绘制缝隙线
    if gap_type == 'horizontal':
        # 水平线
        start_point = (0, int(position))
        end_point = (img_w, int(position))
    else:
        # 垂直线
        start_point = (int(position), 0)
        end_point = (int(position), img_h)
    
    # 使用特定颜色绘制缝隙
    cv2.line(img, start_point, end_point, (45, 57, 59), thickness=thickness)
    return True

def process_single_jpg(jpg_file_path, output_index):
    """
    处理单个jpg文件：生成随机缝隙 + 自然不规则光照（光斑/阴影），生成新文件
    """
    # 1. 校验标注文件
    json_file_path = os.path.splitext(jpg_file_path)[0] + ".json"
    if not os.path.exists(json_file_path):
        print(f"警告：未找到标注文件{json_file_path}，跳过")
        return
    
    # 获取十字交叉点
    cross_point = get_cross_intersection(json_file_path)
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json.load(f)  # 仅校验可读取，无需解析内容
    except Exception as e:
        print(f"错误：读取标注文件失败 {e}，跳过")
        return

    # 2. 加载图片并校验
    img = cv2.imread(jpg_file_path)
    if img is None:
        print(f"错误：无法加载图片{jpg_file_path}，跳过")
        return
    img_h, img_w = img.shape[:2]
    working_img = img.copy().astype(np.float32)  # 使用副本进行操作

    # 3. 添加随机缝隙（水平和垂直各最多一条）- 先执行这一步
    min_dist_from_boundary_h = CONFIG["min_distance_from_boundary"]
    min_dist_from_boundary_w = CONFIG["min_distance_from_boundary"]
    
    # 随机决定是否添加水平缝隙
    # if random.choice([True, False]):
    #     # 计算水平缝隙的有效范围（避开边界）
    #     valid_y_range = (min_dist_from_boundary_h, img_h - min_dist_from_boundary_h)
    #     if valid_y_range[0] < valid_y_range[1]:
    #         horizontal_pos = random.randint(valid_y_range[0], valid_y_range[1])
            
    #         # 确定缝隙厚度
    #         thickness = random.randint(*CONFIG["gap_thickness_range"])
            
    #         # 尝试绘制水平缝隙
    #         if draw_gap_line(working_img, 'horizontal', horizontal_pos, thickness, cross_point):
    #             print(f"添加水平缝隙: Y={horizontal_pos}, 厚度={thickness}")
    
    # 随机决定是否添加垂直缝隙
    # if random.choice([True, False]):
    #     # 计算垂直缝隙的有效范围（避开边界）
    #     valid_x_range = (min_dist_from_boundary_w, img_w - min_dist_from_boundary_w)
    #     if valid_x_range[0] < valid_x_range[1]:
    #         vertical_pos = random.randint(valid_x_range[0], valid_x_range[1])
            
    #         # 确定缝隙厚度
    #         thickness = random.randint(*CONFIG["gap_thickness_range"])
            
    #         # 尝试绘制垂直缝隙
    #         if draw_gap_line(working_img, 'vertical', vertical_pos, thickness, cross_point):
    #             print(f"添加垂直缝隙: X={vertical_pos}, 厚度={thickness}")

    # 4. 初始化全局亮度掩膜（核心：叠加所有光照，避免逐区域修改导致像素溢出）
    global_light_mask = np.ones((img_h, img_w, 3), dtype=np.float32)

    # 5. 生成多个自然光照区域（核心优化：所有区域先叠加到掩膜，最后一次性应用）
    num_regions = random.randint(*CONFIG["num_regions"])
    
    # 生成均匀分布的中心点
    center_points = generate_grid_points(img_w, img_h, num_regions)
    
    for cx, cy in center_points:
        # 5.1 计算光照区域最大尺寸
        max_region_size = int(min(img_w, img_h) * CONFIG["region_size_ratio"])
        max_region_size = max(max_region_size, CONFIG["min_region_size"])

        # 5.2 限制中心点靠近边缘的距离，避免区域超出边界
        max_region_size_half = max_region_size // 2
        cx = max(max_region_size_half, min(img_w - max_region_size_half, cx))
        cy = max(max_region_size_half, min(img_h - max_region_size_half, cy))

        # 5.3 生成不规则凸包形状（替代矩形，自然光照形态）
        points = []
        for _ in range(random.randint(*CONFIG["hull_point_count"])):
            # 围绕中心随机生成点，限制在区域范围内
            px = cx + random.randint(-max_region_size//2, max_region_size//2)
            py = cy + random.randint(-max_region_size//2, max_region_size//2)
            px = np.clip(px, 0, img_w-1)
            py = np.clip(py, 0, img_h-1)
            points.append([px, py])
        hull = cv2.convexHull(np.array(points, dtype=np.int32))

        # 5.4 生成单区域光照掩膜（径向渐变+边缘模糊，自然过渡）
        # 第一步：绘制不规则形状基础掩膜
        single_mask = np.zeros((img_h, img_w), dtype=np.float32)
        cv2.fillConvexPoly(single_mask, hull, 1.0)

        # 第二步：生成径向渐变（中心强→边缘弱，模拟真实光照）
        y, x = np.ogrid[:img_h, :img_w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist = np.clip(dist, 0, max_region_size//2)
        grad = (max_region_size//2 - dist) / (max_region_size//2)  # 0→1渐变，稳定无异常

        # 第三步：设置单区域亮度因子（温和范围，无极端暗）
        brightness = random.uniform(*CONFIG["brightness_range"])
        single_light = 1.0 + (brightness - 1.0) * grad * single_mask

        # 第四步：高斯模糊，消除形状轮廓，边缘平滑
        blur = random.randint(*CONFIG["blur_size"])
        single_light = cv2.GaussianBlur(single_light, (blur*2+1, blur*2+1), 0)

        # 5.5 叠加到全局掩膜（关键：保留3通道，避免逐区域修改像素）
        global_light_mask[:, :, 0] *= single_light
        global_light_mask[:, :, 1] *= single_light
        global_light_mask[:, :, 2] *= single_light

    # 6. 应用全局光照掩膜到已添加缝隙的图片（核心：强制约束亮度下限，彻底杜绝全黑）
    global_light_mask = np.clip(global_light_mask, 0.6, 2.0)  # 最低亮度0.6，无全黑
    result = np.clip(working_img * global_light_mask, 0, 255).astype(np.uint8)

    # 7. 生成新的文件名并保存
    base_name = os.path.splitext(os.path.basename(jpg_file_path))[0]
    dir_name = os.path.dirname(jpg_file_path)
    
    new_jpg_path = os.path.join(dir_name, f"{base_name}-{output_index}.jpg")
    new_json_path = os.path.join(dir_name, f"{base_name}-{output_index}.json")
    
    # 保存处理后的图片
    try:
        cv2.imwrite(new_jpg_path, result)
        print(f"成功生成：{new_jpg_path}")
    except Exception as e:
        print(f"错误：写入文件失败 {e}，跳过")

    # 复制JSON文件
    try:
        shutil.copy2(json_file_path, new_json_path)
        print(f"成功复制JSON文件：{new_json_path}")
    except Exception as e:
        print(f"错误：复制JSON文件失败 {e}，跳过")

def batch_process_jpgs(target_dir, copies_per_image=1):
    """批量处理目录下所有jpg文件（含大小写），为每张图片生成多份副本"""
    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"目录不存在：{target_dir}")
    
    for file in os.listdir(target_dir):
        if file.lower().endswith(".jpg"):
            jpg_file_path = os.path.join(target_dir, file)
            
            # 为每张图片生成指定数量的副本
            for i in range(1, copies_per_image + 1):
                process_single_jpg(jpg_file_path, i)
                
    print("批量处理完成！")

if __name__ == "__main__":
    # 替换为你的目标目录
    TARGET_DIRECTORY = "E:/code/LabelMe/origin-add-02041423/"
    
    # 每张图片生成多少份副本（可以根据需要调整）
    COPIES_PER_IMAGE = 4
    
    try:
        batch_process_jpgs(TARGET_DIRECTORY, COPIES_PER_IMAGE)
    except Exception as e:
        print(f"批量处理失败：{e}")