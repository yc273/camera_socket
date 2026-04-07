# 视觉相机识别系统

基于 YOLOv8 的工业相机实时检测系统，支持螺钉和十字交叉点识别，通过 Socket 服务器提供检测结果。

## 功能特性

- 🎯 **实时检测**：使用工业相机进行实时图像采集和目标检测
- 🔩 **螺钉识别**：检测螺钉位置和姿态
- ✚ **十字交叉点识别**：检测十字交叉点坐标
- 📡 **Socket 服务器**：提供 TCP Socket 接口获取检测结果
- 🔄 **自动重连**：相机断连自动重连机制
- 📊 **多裁切策略**：使用多种偏移裁切提高检测精度

## 环境要求

- Python 3.8+
- Windows 操作系统
- 海康威视工业相机及驱动

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/yc273/camera_socket.git
cd camera_socket
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置相机

确保已安装海康威视相机驱动，相机相关文件位于 `lib/` 目录。

### 4. 准备模型文件

训练权重文件已包含在 `runs/pose/` 目录中：
- `train-stud/weights/best-1280-1.pt` - 螺钉检测模型
- `train-cross/weights/best-1920-1.pt` - 十字交叉点检测模型

## 使用方法

### 启动 Socket 服务器

```bash
python server_socket.py
```

服务器将在 `0.0.0.0:9999` 启动，等待客户端连接。

### 客户端通信协议

**连接**
```
TCP 连接到服务器 IP:9999
```

**请求格式 (JSON)**
```json
{
  "action": "detect"
}
```

**响应格式 (JSON)**
```json
{
  "status": "success",
  "stud_position": {"x": 100.5, "y": 200.3},
  "cross_position": {"x": 150.2, "y": 180.7},
  "confidence": 0.95,
  "timestamp": "2024-01-01 12:00:00"
}
```

## 项目结构

```
camera_socket/
├── camera.py              # 相机控制类
├── realtime_test.py       # 实时检测核心逻辑
├── realtime_test_origin.py # 全局检测器实例
├── server_socket.py       # Socket 服务器
├── interface.py           # 用户界面
├── auto_gamma.py          # 自动伽马校正
├── requirements.txt       # Python 依赖
├── lib/                   # 相机 SDK 和库文件
├── runs/                  # 训练模型和权重
│   └── pose/
│       ├── train-stud/    # 螺钉检测模型
│       └── train-cross/   # 十字交叉点检测模型
└── yolov8/                # YOLOv8 相关工具
```

## 配置说明

### 相机参数

相机参数配置位于 `lib/CommonParameters.ini`，根据实际相机型号调整。

### 检测参数

在 `realtime_test.py` 中可调整：
- `detection_conf`: 检测置信度阈值（默认 0.25）
- `disconnect_threshold`: 断连判定时间（默认 2.0 秒）
- 裁切参数 `CROP_W`, `CROP_H` 等

## 常见问题

### 1. 相机无法连接

- 检查相机驱动是否正确安装
- 确认相机 IP 和网络配置
- 查看防火墙设置

### 2. 模型加载失败

- 确认 `runs/` 目录下存在对应的 `.pt` 权重文件
- 检查 Git LFS 是否正确下载大文件

### 3. Socket 连接失败

- 确认服务器已启动
- 检查端口 9999 是否被占用
- 查看防火墙设置

## 技术栈

- **深度学习框架**: PyTorch 2.7.1
- **目标检测**: YOLOv8 (Ultralytics)
- **图像处理**: OpenCV
- **相机 SDK**: 海康威视 MvCameraControl
- **通信**: Python Socket

## 许可证

MIT License

## 联系方式

GitHub: https://github.com/yc273/camera_socket
