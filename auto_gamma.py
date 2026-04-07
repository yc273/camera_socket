import os
import random
import cv2
import numpy as np
from pathlib import Path

# ===================== 【智能可调参数区】 =====================
INPUT_FOLDER = r"E:/code/ImageRecognition/test/test"       # 输入文件夹
OUTPUT_FOLDER = r"E:/code/ImageRecognition/test/test_result"     # 输出文件夹
GENERATE_NUM = 1                # 每张图生成数量

# 🔴 智能亮度检测阈值（核心）
# 平均值 < 该值 → 判断为暗图，自动提亮
DARK_THRESHOLD = 80  # 推荐：70~100，数值越小越严格

# 暗图使用的伽马值（gamma越大越亮）
DARK_GAMMA_RANGE = (1.8, 2.8)
# 亮图使用的伽马值（=1.0 不处理）
BRIGHT_GAMMA = 1.0

# 对比度（统一）
CONTRAST_RANGE = (1.0, 1.2)
USE_RANDOM_PARAMS = True
# =================================================================

def calculate_brightness(image):
    """计算图片平均亮度（0~255，数值越小越暗）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def adjust_gamma(image, gamma=2.0):
    """伽马提亮：gamma越大越亮，1.0=原图"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_contrast(img, alpha=1.0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def process_images():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    input_path = Path(INPUT_FOLDER)
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))

    if not image_files:
        print("❌ 未找到 JPG 图片")
        return

    print(f"✅ 找到 {len(image_files)} 张图片，开始【智能明暗检测】处理...\n")

    for img_path in image_files:
        file_stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 读取失败：{img_path.name}")
            continue

        # 🔥 核心：自动检测亮度
        avg_bright = calculate_brightness(img)
        is_dark = avg_bright < DARK_THRESHOLD

        status = "🔴 暗图 → 自动提亮" if is_dark else "🟢 亮图 → 不处理"
        print(f"处理：{img_path.name} | 平均亮度：{avg_bright:.1f} | {status}")

        for idx in range(1, GENERATE_NUM + 1):
            # 智能选择伽马值
            if is_dark:
                gamma = random.uniform(*DARK_GAMMA_RANGE) if USE_RANDOM_PARAMS else 2.2
            else:
                gamma = BRIGHT_GAMMA  # 亮图=1.0，不提亮

            # 对比度
            contrast = random.uniform(*CONTRAST_RANGE) if USE_RANDOM_PARAMS else 1.0

            # 处理图片
            img_gamma = adjust_gamma(img, gamma=gamma)
            new_img = adjust_contrast(img_gamma, alpha=contrast)

            # 保存
            new_name = f"{file_stem}_copy_{idx}{img_path.suffix}"
            save_path = os.path.join(OUTPUT_FOLDER, new_name)
            cv2.imwrite(save_path, new_img)

            print(f"   生成：{new_name} | γ:{gamma:.2f}")

            # 复制 json
            json_src = input_path / f"{file_stem}.json"
            if json_src.exists():
                json_dst = os.path.join(OUTPUT_FOLDER, f"{file_stem}_copy_{idx}.json")
                with open(json_src, 'rb') as f1, open(json_dst, 'wb') as f2:
                    f2.write(f1.read())

            # 复制 txt
            txt_src = input_path / f"{file_stem}.txt"
            if txt_src.exists():
                txt_dst = os.path.join(OUTPUT_FOLDER, f"{file_stem}_copy_{idx}.txt")
                with open(txt_src, 'rb') as f1, open(txt_dst, 'wb') as f2:
                    f2.write(f1.read())

    print("\n🎉 【智能处理完成】暗图自动提亮，亮图保持原样！")

if __name__ == "__main__":
    process_images()