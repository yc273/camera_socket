import json
import os
import glob

# 配置参数（移除KEYPOINT_VISIBILITY，保持其他配置不变）
CLASS_NAME_TO_ID = {"CrossIntersection": 0,"StudCenter":1}  # 关键点类别→ID映射
BOX_LABEL_NAME = "TargetArea"  # 矩形框的标注标签名（必须和LabelMe中标注的一致）
DEFAULT_GROUP_KEY = "no_group"  # 用于存放group_id为null的默认分组标识

def labelme_box_and_point2yolov8_pose(json_path):
    # 1. 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 获取图片基本信息
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    json_basename = os.path.basename(json_path)  # 获取json文件名（如 origin-1.json）
    txt_basename = os.path.splitext(json_basename)[0] + '.txt'  # 转换为 origin-1.txt
    txt_path = os.path.join(os.path.dirname(json_path), txt_basename)  # 生成txt完整路径
    
    # 3. 初始化分组字典：
    # key=group_id（或DEFAULT_GROUP_KEY），value={box_info: ..., kpt_infos: [...], kpt_label: ...}
    # 其中kpt_infos是列表，存储该组所有关键点的归一化坐标
    group_data = {}
    
    # 4. 遍历标注形状，按规则归类矩形框和多个关键点
    for shape in data['shapes']:
        # 获取当前形状的分组ID，为null则使用默认分组标识
        group_id = shape.get('group_id')
        current_group_key = group_id if group_id is not None else DEFAULT_GROUP_KEY
        
        # 4.1 提取矩形框（TargetArea），按当前分组key存入分组字典
        if shape['shape_type'] == 'rectangle' and shape['label'] == BOX_LABEL_NAME:
            # 获取矩形框的两个对角点（LabelMe的rectangle存储两个点）
            point1_x, point1_y = shape['points'][0]
            point2_x, point2_y = shape['points'][1]
            
            # 计算矩形框的左上角、右下角坐标（确保坐标有序，避免负数尺寸）
            x_min = min(point1_x, point2_x)
            y_min = min(point1_y, point2_y)
            x_max = max(point1_x, point2_x)
            y_max = max(point1_y, point2_y)
            
            # 边界检查（防止超出图片范围）
            x_min = max(0, min(img_width, x_min))
            y_min = max(0, min(img_height, y_min))
            x_max = max(0, min(img_width, x_max))
            y_max = max(0, min(img_height, y_max))
            
            # 计算YOLO格式的框参数（中心坐标、宽高）
            box_width_pixel = x_max - x_min
            box_height_pixel = y_max - y_min
            box_x_center_pixel = x_min + box_width_pixel / 2
            box_y_center_pixel = y_min + box_height_pixel / 2
            
            # 归一化（转换为0~1之间，保留6位小数）
            x_center = box_x_center_pixel / img_width
            y_center = box_y_center_pixel / img_height
            width = box_width_pixel / img_width
            height = box_height_pixel / img_height
            
            # 按当前分组key存入分组字典（若分组不存在则创建）
            if current_group_key not in group_data:
                group_data[current_group_key] = {}
            group_data[current_group_key]['box_info'] = [x_center, y_center, width, height]
            continue
        
        # 4.2 提取多个关键点（CrossIntersection），按当前分组key存入分组字典（列表形式）
        if shape['shape_type'] == 'point' and shape['label'] in CLASS_NAME_TO_ID:
            # 获取点坐标，并做边界检查
            point_x, point_y = shape['points'][0]
            point_x = max(0, min(img_width, point_x))
            point_y = max(0, min(img_height, point_y))
            
            # 归一化坐标
            x_normalized = point_x / img_width
            y_normalized = point_y / img_height
            
            # 按当前分组key存入分组字典（列表形式，追加多个点，不覆盖）
            if current_group_key not in group_data:
                group_data[current_group_key] = {}
            # 初始化关键点列表（若不存在），然后追加当前关键点
            if 'kpt_infos' not in group_data[current_group_key]:
                group_data[current_group_key]['kpt_infos'] = []
            group_data[current_group_key]['kpt_infos'].append([x_normalized, y_normalized])
            # 存储关键点标签（只需存储一次即可，所有点标签一致）
            group_data[current_group_key]['kpt_label'] = shape['label']
            continue
    
    # 5. 校验并生成YOLOv8 Pose标注行（每组对应一行，多个关键点按顺序拼接）
    yolo_lines = []  # 存储所有合法分组的标注行
    for group_key, group_info in group_data.items():
        # 区分分组类型，用于提示信息
        group_tip = f"group_id={group_key}" if group_key != DEFAULT_GROUP_KEY else "无group_id（默认组）"
        
        # 校验：该分组必须同时存在矩形框和至少1个关键点
        if 'box_info' not in group_info:
            print(f"警告：{json_path} 中{group_tip} 未找到 {BOX_LABEL_NAME} 矩形框，跳过该分组")
            continue
        if 'kpt_infos' not in group_info or len(group_info['kpt_infos']) == 0:
            print(f"警告：{json_path} 中{group_tip} 未找到有效关键点（至少1个），跳过该分组")
            continue
        
        # 提取该分组的有效数据
        box_info = group_info['box_info']
        kpt_infos = group_info['kpt_infos']  # 多个关键点的列表（[[x1,y1], [x2,y2], ...]）
        kpt_label = group_info['kpt_label']
        class_id = CLASS_NAME_TO_ID[kpt_label]
        
        # 5.1 拼接框参数（保留6位小数）
        box_str = " ".join([f"{param:.6f}" for param in box_info])
        
        # 5.2 拼接多个关键点参数（按顺序平铺x1 y1 x2 y2 ...，保留6位小数）
        kpt_str = ""
        for kpt in kpt_infos:
            kpt_str += f"{kpt[0]:.6f} {kpt[1]:.6f} "
        # 去除末尾多余的空格（可选，不影响YOLO解析，仅让格式更整洁）
        kpt_str = kpt_str.strip()
        
        # 5.3 生成单组标注行（格式：class_id 框参数 多个关键点参数，移除可见性）
        line = f"{class_id} {box_str} {kpt_str}\n"
        yolo_lines.append(line)
    
    # 6. 若没有合法标注行，直接返回
    if not yolo_lines:
        print(f"警告：{json_path} 中无有效分组标注，跳过该文件")
        return None
    
    # 7. 写入txt文件（所有合法分组的标注行写入同一个txt）
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(yolo_lines)
    
    print(f"转换完成：{json_path} → {txt_path}（共{len(yolo_lines)}个有效分组）")
    return txt_path

# 批量转换（可选：单文件/批量）
if __name__ == "__main__":
    # 方式1：转换单个JSON文件（注释解除后使用）
    # json_file = "你的标注文件.json"
    # labelme_box_and_point2yolov8_pose(json_file)
    
    # 方式2：批量转换指定目录下所有JSON文件
    json_dir = "E:/code/LabelMe/add-十字线-1/"  # 替换为你的JSON文件所在目录
    # 检查目录是否存在
    if not os.path.exists(json_dir):
        print(f"错误：目录{json_dir}不存在，请检查路径")
    else:
        for json_file in glob.glob(os.path.join(json_dir, "*.json")):
            labelme_box_and_point2yolov8_pose(json_file)