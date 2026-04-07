# 导入YOLOv8模型
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
from PIL import ImageDraw
import time

def crop_image_with_offset(img, crop_w, crop_h, offsets):
    """
    裁切图片并记录偏移值（核心：只记录裁切的x1/y1偏移）
    """
    img_w, img_h = img.size
    crop_imgs = []
    crop_offsets = []  # 仅记录裁切图在原图的左上角偏移 (x1, y1)

    for idx, (dx, dy) in enumerate(offsets):
        # 居中裁切 + 轻微偏移
        x1 = (img_w - crop_w) // 2 + dx
        y1 = (img_h - crop_h) // 2 + dy
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        crop_img = img.crop((x1, y1, x2, y2))
        crop_imgs.append(crop_img)
        crop_offsets.append((x1, y1))  # 只保留偏移值，用于后续补位

    return crop_imgs, crop_offsets

def save_detection_result(result, save_path):
    """保存带关键点的识别结果图"""
    img_array = result.plot()
    img = Image.fromarray(img_array)
    img.save(save_path, quality=95)
    return save_path

# ---------------------- 主流程 ----------------------
if __name__ == "__main__":
    # 1. 加载模型
    model = YOLO(r"E:\code\ImageRecognition\runs\pose\train-cross\weights\best-1920-new-11.pt")

    # 2. 配置（极简）
    test_folder       = r"E:\code\ImageRecognition\test\images8"
    detect_save_dir   = r"E:\code\ImageRecognition\test\crop_detect"    # 裁切图识别结果
    final_save_dir    = r"E:\code\ImageRecognition\test\final_result"   # 整合后的原图
    os.makedirs(detect_save_dir, exist_ok=True)
    os.makedirs(final_save_dir, exist_ok=True)

    # 核心尺寸（只裁20px）
    ORIG_W, ORIG_H = 3072, 2048
    CROP_W, CROP_H = 3032, 2008  # 3072-40=3032, 2048-40=2008
    offsets        = [(0, 0), (-10, -10), (10, 10)]  # 3种裁切偏移
    timestamp      = str(int(time.time() * 1000))

    # 3. 处理每张图
    for fname in os.listdir(test_folder):
        if not fname.lower().endswith(".jpg"):
            continue
        
        img_base = os.path.splitext(fname)[0]
        img_path = os.path.join(test_folder, fname)
        print(f"\n===== 处理：{fname} =====")

        try:
            # 打开原图并固定尺寸
            orig_img = Image.open(img_path).convert("RGB")
            orig_img = orig_img.resize((ORIG_W, ORIG_H), Image.Resampling.LANCZOS)
            final_img = orig_img.copy()

            # 4. 裁切3张图 + 获取偏移值
            crop_imgs, crop_offsets = crop_image_with_offset(orig_img, CROP_W, CROP_H, offsets)
            
            # 5. 识别 + 收集坐标（核心简化）
            orig_coords = []  # 存储映射回原图的坐标
            for crop_idx, (crop_img, (x1_offset, y1_offset)) in enumerate(zip(crop_imgs, crop_offsets)):
                # 生成唯一文件名
                unique_suffix = f"{img_base}_crop{crop_idx+1}_dx{offsets[crop_idx][0]}_dy{offsets[crop_idx][1]}"
                
                # 推理裁切图
                results = model.predict(
                    source=crop_img,
                    imgsz=1920,
                    conf=0.6,
                    save=False,
                    show=False
                )
                res = results[0]

                # 保存识别结果图
                detect_path = os.path.join(detect_save_dir, f"{unique_suffix}_detect.jpg")
                save_detection_result(res, detect_path)
                print(f"  保存裁切{crop_idx+1}识别图：{detect_path}")

                # 6. 核心映射逻辑（你说的：识别坐标 + 裁切偏移 = 原图坐标）
                if res.keypoints is not None and len(res.keypoints.xy) > 0:
                    # YOLO输出的是裁切图的相对坐标（已自动适配比例）
                    kpt = res.keypoints.xy[0][0]
                    x_crop = kpt[0].item()  # 裁切图内的x
                    y_crop = kpt[1].item()  # 裁切图内的y

                    # 最简单的映射：直接补裁切偏移值
                    x_orig = x_crop + x1_offset
                    y_orig = y_crop + y1_offset

                    orig_coords.append((x_orig, y_orig))
                    print(f"  裁切{crop_idx+1}：识别坐标({x_crop:.2f},{y_crop:.2f}) + 偏移({x1_offset},{y1_offset}) = 原图坐标({x_orig:.2f},{y_orig:.2f})")
                else:
                    print(f"  裁切{crop_idx+1}：未检测到关键点")

            # 7. 平均坐标 + 绘制到原图
            if len(orig_coords) > 0:
                avg_x = np.mean([x for x, y in orig_coords])
                avg_y = np.mean([y for x, y in orig_coords])
                print(f"\n  最终平均坐标：({avg_x:.2f}, {avg_y:.2f}) (原图3072×2048)")

                # 绘制标记
                draw = ImageDraw.Draw(final_img)
                x, y = round(avg_x), round(avg_y)
                draw.ellipse([x-7, y-7, x+7, y+7], fill=(255,0,0), outline=(0,0,0), width=3)

                # 保存最终图
                final_path = os.path.join(final_save_dir, f"{img_base}_final.jpg")
                final_img.save(final_path, quality=95)
                print(f"  保存最终图：{final_path}")
            else:
                print(f"  无有效坐标，跳过整合")

        except Exception as e:
            print(f"  出错：{str(e)}")