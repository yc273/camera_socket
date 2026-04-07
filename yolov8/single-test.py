# 导入YOLOv8模型
from ultralytics import YOLO
from PIL import Image
import os

def convert_jpg_to_black_white(target_path):
    """
    将指定路径下所有jpg格式图片转换为黑白灰度图
    :param target_path: 目标文件夹路径
    """
    # 1. 检查指定路径是否存在，不存在则提示并退出
    if not os.path.exists(target_path):
        print(f"错误：指定的路径 {target_path} 不存在！")
        return
    
    # 2. 遍历目标路径下的所有文件
    for file_name in os.listdir(target_path):
        # 3. 筛选出jpg格式文件（兼容小写.jpg和大写.JPG）
        if file_name.lower().endswith(".jpg"):
            # 拼接完整的文件路径（避免路径分隔符问题，使用os.path.join）
            file_full_path = os.path.join(target_path, file_name)
            
            try:
                # 4. 打开图片文件
                with Image.open(file_full_path) as img:
                    # 5. 转换为黑白灰度图（L模式：8位灰度，0=黑，255=白）
                    gray_img = img.convert("L")
                    
                    # 6. 保存转换后的图片（两种方式可选，注释其中一种即可）
                    # 方式1：覆盖原图片（谨慎使用，会丢失原彩色图片）
                    # gray_img.save(file_full_path)
                    
                    # 方式2：不覆盖原图，生成新文件（文件名后加"_gray"区分）
                    new_file_name = os.path.splitext(file_name)[0] + "_gray.jpg"
                    new_file_full_path = os.path.join(target_path, new_file_name)
                    gray_img.save(new_file_full_path)
                
                print(f"成功转换：{file_name}")
            
            except Exception as e:
                print(f"处理失败：{file_name}，错误信息：{str(e)}")

# 1. 加载训练好的Pose关键点模型（注意：这里是pose训练后的best.pt，不是detect的）
model = YOLO(r"E:\code\ImageRecognition\runs\pose\train-cross\weights\best-1280-3.pt")  # 建议修改为你的pose模型路径

# (可选)转换为黑白照片
# target_folder = r"E:\code\ImageRecognition\test\images"
# convert_jpg_to_black_white(target_folder)

# 2. 执行关键点检测（支持多种数据源，按需选择一种即可）
# 可选数据源：单张图片、图片文件夹、视频、摄像头
# 方式A：检测单张图片
# results = model.predict(
#     source=r"E:\code\ImageRecognition\test.jpg",  # 待检测图片路径
#     imgsz=640,  # 输入图片尺寸
#     conf=0.25,  # 置信度阈值
#     save=True,  # 保存检测结果到runs/pose/predict
#     show=False,  # 是否实时显示检测结果（本地运行可改为True）
#     verbose=False  # 关闭详细日志输出
# )

# 方式B：检测文件夹下所有图片（推荐，保留你原有选择）
# results = model.predict(source=r"E:\code\ImageRecognition\target_images", imgsz=640, conf=0.25, save=True,device="cpu",show=False)
# results = model.predict(source=r"E:\code\ImageRecognition\test\images", imgsz=1280, conf=0.25, save=True,device="cpu",show=False,
#                         project=r"E:\code\ImageRecognition\test",name="result")
results = model.predict(source=r"E:\code\ImageRecognition\test\images3", imgsz=1280, conf=0.25, save=True,show=False,
                        project=r"E:\code\ImageRecognition\test",name="result")
# 3. 解析关键点检测结果（提取十字线交叉点坐标、置信度等信息）
for result in results:
    # 提取关键点信息（替代原有的box信息，核心修改）
    keypoints = result.keypoints  # 获取所有关键点信息
    if keypoints is not None and keypoints.xy is not None:
        # 遍历每张图片的关键点（xy.shape: [目标数量, 关键点数量, 2(x,y)]）
        for obj_idx, kpt in enumerate(keypoints.xy):
            # kpt：当前目标的所有关键点坐标（你的场景只有1个十字交叉点，对应kpt[0]）
            for point_idx, (x, y) in enumerate(kpt):
                # 转换为像素坐标（保留2位小数，更易读取）
                pixel_x = round(x.item(), 2)
                pixel_y = round(y.item(), 2)
                
                # 提取该关键点的置信度（可选，反映模型对该点的判断可信度）
                kpt_conf = keypoints.conf[obj_idx][point_idx].item() if keypoints.conf is not None else 1.0
                kpt_conf = round(kpt_conf, 4)
                
                # 提取类别信息（和你的训练数据集类别一致）
                cls = result.boxes.cls[obj_idx].item() if result.boxes is not None else 0.0
                cls_name = model.names[int(cls)] if hasattr(model, 'names') else "CrossIntersection"
                
                # 打印解析结果（清晰展示十字线交叉点信息）
                print(f"类别：{cls_name}，关键点置信度：{kpt_conf}，交叉点像素坐标：(x={pixel_x}, y={pixel_y})")

