# -*- coding:utf-8 -*-
import os
import random
import shutil

def data_split(full_list, ratio):
    """
    按比例分割数据集
    :param full_list: 完整的数据列表
    :param ratio: 训练集占比
    :return: 分割后的两个子列表
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

# 目标目录结构
train_p = "train"
val_p = "val"
imgs_p = "images"
labels_p = "labels"

# 使用完整的路径定义
base_dir = "E:/code/LabelMe/"
train_p = os.path.join(base_dir, "origin80-random-3/train")
val_p = os.path.join(base_dir, "origin80-random-3/val")

# 创建训练集目录
if not os.path.exists(train_p):
    os.makedirs(train_p)
tp1 = os.path.join(train_p, imgs_p)
tp2 = os.path.join(train_p, labels_p)
if not os.path.exists(tp1):
    os.makedirs(tp1)
if not os.path.exists(tp2):
    os.makedirs(tp2)

# 创建验证集目录
if not os.path.exists(val_p):
    os.makedirs(val_p)
vp1 = os.path.join(val_p, imgs_p)
vp2 = os.path.join(val_p, labels_p)
if not os.path.exists(vp1):
    os.makedirs(vp1)
if not os.path.exists(vp2):
    os.makedirs(vp2)

# 数据集路径（图片和标签分开）
images_dir = "E:/code/LabelMe/origin80-random-3"  # 图片目录
labels_dir = "E:/code/LabelMe/origin80-random-3"  # 标签目录

# 划分数据集，设置训练集占比
proportion_ = 0.7  # 训练集占比

# 获取所有标签文件名（不含扩展名）
label_files = [f.split('.')[0] for f in os.listdir(labels_dir) if f.endswith('.txt')]
num = len(label_files)  # 统计标签文件数量
print(f"总标签文件数量: {num}")

# 按比例分割数据集
list1, list2 = data_split(label_files, proportion_)
print(f"训练集数量: {len(list1)}, 验证集数量: {len(list2)}")

# 复制训练集文件
for name in list1:
    jpg_src = os.path.join(images_dir, name + '.jpg')
    txt_src = os.path.join(labels_dir, name + '.txt')
    jpg_dst = os.path.join(tp1, name + '.jpg')
    txt_dst = os.path.join(tp2, name + '.txt')

    if os.path.exists(jpg_src) and os.path.exists(txt_src):
        shutil.copyfile(jpg_src, jpg_dst)
        shutil.copyfile(txt_src, txt_dst)
    else:
        print(f"缺失文件: {jpg_src} 或 {txt_src}")

# 复制验证集文件
for name in list2:
    jpg_src = os.path.join(images_dir, name + '.jpg')
    txt_src = os.path.join(labels_dir, name + '.txt')
    jpg_dst = os.path.join(vp1, name + '.jpg')
    txt_dst = os.path.join(vp2, name + '.txt')

    if os.path.exists(jpg_src) and os.path.exists(txt_src):
        shutil.copyfile(jpg_src, jpg_dst)
        shutil.copyfile(txt_src, txt_dst)
    else:
        print(f"缺失文件: {jpg_src} 或 {txt_src}")

print("数据集划分完成！")