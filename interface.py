import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QWidget, QHBoxLayout, QSpinBox, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from camera import Camera
import os
import time
import cv2
import numpy as np
from ctypes import *
from threading import Thread, Event
from MvCameraControl_class import *
# 添加缺失的像素类型常量引用
from PixelType_header import *
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QWidget, QHBoxLayout, QSpinBox, QGroupBox,
                            QSizePolicy)
# ... existing code ...
class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能相机控制系统")
        self.setGeometry(100, 100, 900, 700)

        # 初始化相机
        self.camera = Camera()
        if not self.camera.connect():
            print("无法连接相机！")
            sys.exit()

        # 自动采集控制变量
        self.is_auto_capturing = False
        self.current_capture_count = 0
        self.total_captures_needed = 0
        self.captures_per_second = 5  # 默认每秒5张
        
        # 创建UI
        self.setup_ui()
        
        # 启动视频流
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video)
        self.camera.start_video_stream(callback=self.update_frame)
        self.video_timer.start(30)  # 每30ms更新画面

        # 自动采集定时器
        self.auto_capture_timer = QTimer(self)
        self.auto_capture_timer.timeout.connect(self.perform_auto_capture)

# ... existing code ...
    def setup_ui(self):
        """设置用户界面"""
        # 主要显示区域
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setText("等待视频流...")

        # 状态显示
        self.status_label = QLabel("状态：就绪", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        # 控制面板
        control_group = self.create_control_panel()
        
        # 布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addWidget(self.status_label, stretch=0)
        main_layout.addWidget(control_group, stretch=0)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
# ... existing code ...

    def create_control_panel(self):
        """创建控制面板"""
        group_box = QGroupBox("相机控制")
        layout = QVBoxLayout()

        # 参数设置区域
        param_layout = QHBoxLayout()
        
        # 每秒采集张数设置
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 30)  # 1-30张/秒
        self.fps_spinbox.setValue(5)
        self.fps_spinbox.valueChanged.connect(self.on_fps_changed)
        
        fps_label = QLabel("每秒采集张数:")
        fps_label.setFixedWidth(100)
        
        param_layout.addWidget(fps_label)
        param_layout.addWidget(self.fps_spinbox)
        param_layout.addStretch()

        # 总采集数量设置
        self.count_spinbox = QSpinBox()
        self.count_spinbox.setRange(1, 1000)
        self.count_spinbox.setValue(10)
        self.count_spinbox.valueChanged.connect(self.on_count_changed)
        
        count_label = QLabel("总采集数量:")
        count_label.setFixedWidth(100)
        
        param_layout.addWidget(count_label)
        param_layout.addWidget(self.count_spinbox)

        layout.addLayout(param_layout)

        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.start_auto_btn = QPushButton("开始自动采集")
        self.start_auto_btn.clicked.connect(self.start_auto_capture)
        self.start_auto_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        
        self.stop_auto_btn = QPushButton("停止采集")
        self.stop_auto_btn.clicked.connect(self.stop_auto_capture)
        self.stop_auto_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.stop_auto_btn.setEnabled(False)
        
        self.manual_save_btn = QPushButton("手动保存图片")
        self.manual_save_btn.clicked.connect(self.manual_save_image)
        
        button_layout.addWidget(self.start_auto_btn)
        button_layout.addWidget(self.stop_auto_btn)
        button_layout.addWidget(self.manual_save_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        group_box.setLayout(layout)
        return group_box

    def on_fps_changed(self, value):
        """FPS参数改变时的回调"""
        self.captures_per_second = value
        self.update_status()

    def on_count_changed(self, value):
        """采集数量参数改变时的回调"""
        self.total_captures_needed = value
        self.update_status()

    def update_frame(self, frame):
        """更新帧数据"""
        self.current_frame = frame

    def update_video(self):
        """更新视频显示"""
        try:
            if not hasattr(self, 'current_frame') or self.current_frame is None:
                return
                
            frame = self.current_frame.copy()
            
            # 确保是 3 通道图像
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            # 创建 QImage 时使用拷贝的数据
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
            
        except Exception as e:
            print(f"视频更新错误：{e}")
            # 发生错误时显示黑色背景
            self.video_label.setStyleSheet("background-color: black; color: red;")
            self.video_label.setText(f"视频流错误\n{str(e)}")

    def update_status(self):
        """更新状态显示"""
        if self.is_auto_capturing:
            status_text = f"状态: 自动采集中 ({self.current_capture_count}/{self.total_captures_needed})"
            if self.total_captures_needed > 0:
                progress = (self.current_capture_count / self.total_captures_needed) * 100
                status_text += f" - 进度: {progress:.1f}%"
        else:
            status_text = f"状态: 就绪 (参数: {self.captures_per_second}张/秒, 共{self.total_captures_needed}张)"
        
        self.status_label.setText(status_text)

    def start_auto_capture(self):
        """开始自动采集"""
        if self.is_auto_capturing:
            return
            
        self.total_captures_needed = self.count_spinbox.value()
        self.captures_per_second = self.fps_spinbox.value()
        self.current_capture_count = 0
        
        if self.total_captures_needed <= 0:
            print("请设置有效的采集数量！")
            return
            
        self.is_auto_capturing = True
        self.start_auto_btn.setEnabled(False)
        self.stop_auto_btn.setEnabled(True)
        self.fps_spinbox.setEnabled(False)
        self.count_spinbox.setEnabled(False)
        
        # 计算采集间隔（毫秒）
        interval_ms = int(1000 / self.captures_per_second)
        
        # 启动采集定时器
        self.auto_capture_timer.start(interval_ms)
        print(f"开始自动采集: {self.captures_per_second}张/秒, 共{self.total_captures_needed}张")
        self.update_status()

    def perform_auto_capture(self):
        """执行单次自动采集"""
        if not self.is_auto_capturing or not hasattr(self, 'current_frame'):
            return
            
        if self.current_capture_count >= self.total_captures_needed:
            self.stop_auto_capture()
            return
            
        # 保存图片
        self.save_auto_image()
        self.current_capture_count += 1
        self.update_status()
        
        # 检查是否完成
        if self.current_capture_count >= self.total_captures_needed:
            self.stop_auto_capture()

    def save_auto_image(self):
        """保存自动采集的图片"""
        if not hasattr(self, 'current_frame'):
            return
            
        # 创建保存目录
        save_dir = "data/auto"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = int(time.time() * 1000)
        filename = f"auto_{self.current_capture_count+1:04d}_{timestamp}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # 保存图片（40%质量压缩）
        success = self.camera.save_frame(save_path, quality=40)
        if success:
            print(f"自动采集 [{self.current_capture_count+1}/{self.total_captures_needed}]: {save_path}")

    def stop_auto_capture(self):
        """停止自动采集"""
        self.is_auto_capturing = False
        self.auto_capture_timer.stop()
        
        self.start_auto_btn.setEnabled(True)
        self.stop_auto_btn.setEnabled(False)
        self.fps_spinbox.setEnabled(True)
        self.count_spinbox.setEnabled(True)
        
        print("自动采集已停止")
        self.update_status()

    def manual_save_image(self):
        """手动保存图片"""
        if not hasattr(self, 'current_frame'):
            print("无可用帧！")
            return
            
        # 创建保存目录
        save_dir = "data/manual"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = int(time.time() * 1000)
        filename = f"manual_{timestamp}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # 保存图片（40%质量压缩）
        success = self.camera.save_frame(save_path, quality=40)
        if success:
            print(f"手动保存: {save_path}")

    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        # 停止所有操作
        self.stop_auto_capture()
        self.camera.stop_video_stream()
        self.camera.disconnect()
        self.video_timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())