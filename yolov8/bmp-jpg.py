import os
import cv2

# 配置
input_dir  = "E:\海康MVS\images"          # 图片所在文件夹（. 表示当前文件夹）
output_dir = "E:\海康MVS\images" # 输出文件夹
quality    = 85           # JPG 画质 0~100，越大越清晰、文件越大

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".bmp"):
        # 读 BMP
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        
        # 保存为 JPG
        name_no_ext = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{name_no_ext}.jpg")
        
        # 压缩保存
        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        print(f"已转换：{fname} -> {out_path}")