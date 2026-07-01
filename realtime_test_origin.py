import cv2
from dateutil.rrule import MO
import numpy as np
from sympy import deg, degree
from torch import obj, res
from camera import Camera  # 导入相机类
from ultralytics import YOLO
import os
from threading import Lock, Thread, Event
import time
import random
from datetime import datetime

# ===================== 全局共享帧 ======================
# 任何模块都可以直接导入这个变量来访问相机帧
_global_shared_frame = None
_global_frame_lock = Lock()

def get_shared_frame():
    """获取共享的相机帧（线程安全）"""
    with _global_frame_lock:
        if _global_shared_frame is not None:
            return _global_shared_frame.copy()
        return None

def set_shared_frame(frame):
    """设置共享的相机帧（线程安全）"""
    with _global_frame_lock:
        global _global_shared_frame
        _global_shared_frame = frame.copy() if frame is not None else None
# ========================================================

# ===================== 【伽马调整参数】=====================
# 智能亮度检测阈值（核心）
DARK_THRESHOLD = 80  # 平均值 < 该值 → 判断为暗图，自动提亮
# 暗图使用的伽马值（gamma越大越亮）
DARK_GAMMA_RANGE = (1.8, 2.8)
# 亮图使用的伽马值（=1.0 不处理）
BRIGHT_GAMMA = 1.0
# 对比度（统一）
CONTRAST_RANGE = (1.0, 1.2)
USE_RANDOM_PARAMS = False  # 实时检测不使用随机参数，保证结果稳定
# ==========================================================


class RealTimePoseDetector:
    _instance = None
    _instance_lock = Lock()  # 线程安全的单例锁

    def __new__(cls, *args, **kwargs):
        """线程安全的单例实现"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, camera_index=0, model_path_stud=None, model_path_cross=None,model_path_u=None, model_path_slot=None, model_path_detect=None, save_frame_path="data/frame.jpg"):
        """防止重复初始化"""
        if hasattr(self, '_initialized') and self._initialized:
            return

        import gc

        # 初始化核心属性
        self.camera_index = camera_index
        self.camera = Camera(device_index=camera_index)
        self.camera_connected = False

        # 模型路径配置（保存路径以便动态加载）
        self.model_path_stud = model_path_stud
        self.model_path_cross = model_path_cross
        self.model_path_u = model_path_u
        self.model_path_slot = model_path_slot
        self.model_path_detect = model_path_detect

        # 模型对象（初始为None，按需加载）
        self.model_stud = None
        self.model_cross = None
        self.model_u = None
        self.model_slot = None
        self.model_detect = None

        # 模型加载状态（是否已加载到GPU）
        self.model_stud_loaded = False
        self.model_cross_loaded = False
        self.model_u_loaded = False
        self.model_slot_loaded = False
        self.model_detect_loaded = False

        # 模型启用开关（是否执行检测）
        self.model_stud_switch = False
        self.model_cross_switch = False
        self.model_u_switch = False
        self.model_slot_switch = False
        self.model_detect_switch = False
        self.model_board_switch = False  # 实际上与cross共用

        # 模型操作锁（保证线程安全）
        self.model_lock = Lock()

        self.save_frame_path = save_frame_path
        self.frame_lock = Lock()
        self.current_frame = None
        self.detection_result_stud = None
        self.detection_result_cross = None
        self.detection_result_board = None
        self.detection_result_u = None
        self.detection_result_slot = None
        self.detection_result_slot_origin = None
        self.detection_result_board_angle = None
        self.detection_result_board_position = None
        self.detection_result_detect = None
        self.running = False  # 初始为 False，避免自动运行

        # 检测线程相关
        self.detection_thread = None
        self.detection_event = Event()
        self.new_frame_event = Event()
        self.detection_conf = 0.25

        # 相机断连检测相关
        self.last_frame_time = 0  # 最后一次接收到帧的时间
        self.disconnect_threshold = 2.0  # 超过2秒无帧则判定为断开
        self.reconnect_interval = 1.0  # 重连间隔（秒）
        self.reconnect_attempts = 0  # 当前重连尝试次数
        self.is_reconnecting = False  # 是否正在重连中

        # 验证模型路径是否存在
        self._validate_model_paths()

        self._initialized = True  # 标记已初始化
        print("检测器初始化完成（模型已配置，初始全部加载，按需开关）")
        self.enable_model_detect()
        self.enable_model_stud()
        self.enable_model_u()
        self.enable_model_slot()
        self.enable_model_cross()
        self.enable_model_board()


    def _validate_model_paths(self):
        """验证所有模型路径是否存在"""
        paths = {
            'stud': self.model_path_stud,
            'cross': self.model_path_cross,
            'u': self.model_path_u,
            'slot': self.model_path_slot,
            'detect': self.model_path_detect
        }

        for model_name, path in paths.items():
            if not path or not os.path.exists(path):
                raise ValueError(f"{model_name} 模型路径不存在：{path}")
        print("所有模型路径验证通过")

    def _load_model_to_gpu(self, model_path):
        """将模型从磁盘加载到GPU"""
        try:
            model = YOLO(model_path)
            # 预热模型（执行一次推理，确保完全加载到GPU）
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model.predict(source=dummy_input, imgsz=640, conf=0.5, verbose=False, save=False, show=False)
            return model
        except Exception as e:
            print(f"加载模型失败 {model_path}: {e}")
            raise

    def _unload_model_from_gpu(self, model):
        """从GPU卸载模型并释放显存"""
        import torch
        import gc

        try:
            if model is not None:
                # 删除模型对象
                del model

            # 清空CUDA缓存
            torch.cuda.empty_cache()

            # 强制垃圾回收
            gc.collect()

        except Exception as e:
            print(f"卸载模型时出错: {e}")

    # ===================== 【模型启用/禁用接口】=====================

    def enable_model_stud(self):
        """启用螺钉检测模型（加载到GPU并启用开关）"""
        with self.model_lock:
            if not self.model_stud_loaded:
                print("正在加载螺钉检测模型...")
                self.model_stud = self._load_model_to_gpu(self.model_path_stud)
                self.model_stud_loaded = True
                print(f"螺钉检测模型加载成功: {self.model_path_stud}")
            self.model_stud_switch = True
            return {"success": True, "message": "螺钉检测模型已启用"}

    def disable_model_stud(self):
        """禁用螺钉检测模型（卸载GPU并关闭开关）"""
        with self.model_lock:
            self.model_stud_switch = False
            if self.model_stud_loaded:
                print("正在卸载螺钉检测模型...")
                self._unload_model_from_gpu(self.model_stud)
                self.model_stud = None
                self.model_stud_loaded = False
                self.detection_result_stud = None  # 清空检测结果
                print("螺钉检测模型已卸载，显存已释放")
            return {"success": True, "message": "螺钉检测模型已禁用"}

    def enable_model_cross(self):
        """启用十字交叉点检测模型"""
        with self.model_lock:
            if not self.model_cross_loaded:
                print("正在加载十字交叉点检测模型...")
                self.model_cross = self._load_model_to_gpu(self.model_path_cross)
                self.model_cross_loaded = True
                print(f"十字交叉点检测模型加载成功: {self.model_path_cross}")
            self.model_cross_switch = True
            return {"success": True, "message": "十字交叉点检测模型已启用"}

    def disable_model_cross(self):
        """禁用十字交叉点检测模型"""
        with self.model_lock:
            self.model_cross_switch = False
            # 检查board模型是否也需要
            if not self.model_board_switch:
                # board也关闭时才卸载模型
                if self.model_cross_loaded:
                    print("正在卸载十字交叉点检测模型...")
                    self._unload_model_from_gpu(self.model_cross)
                    self.model_cross = None
                    self.model_cross_loaded = False
                    self.detection_result_cross = None
                    self.detection_result_board = None
                    self.detection_result_board_angle = None
                    self.detection_result_board_position = None
                    print("十字交叉点检测模型已卸载，显存已释放")
            else:
                print("十字交叉点检测开关已关闭（装板检测仍使用该模型，模型保持加载）")
            return {"success": True, "message": "十字交叉点检测模型已禁用"}

    def enable_model_board(self):
        """启用装板检测模型（与cross共用模型）"""
        with self.model_lock:
            if not self.model_cross_loaded:
                print("正在加载装板检测模型...")
                self.model_cross = self._load_model_to_gpu(self.model_path_cross)
                self.model_cross_loaded = True
                print(f"装板检测模型加载成功: {self.model_path_cross}")
            self.model_board_switch = True
            return {"success": True, "message": "装板检测模型已启用"}

    def disable_model_board(self):
        """禁用装板检测模型"""
        with self.model_lock:
            self.model_board_switch = False
            # 检查cross模型是否也需要
            if not self.model_cross_switch:
                # cross也关闭时才卸载模型
                if self.model_cross_loaded:
                    print("正在卸载装板检测模型...")
                    self._unload_model_from_gpu(self.model_cross)
                    self.model_cross = None
                    self.model_cross_loaded = False
                    self.detection_result_cross = None
                    self.detection_result_board = None
                    self.detection_result_board_angle = None
                    self.detection_result_board_position = None
                    print("装板检测模型已卸载，显存已释放")
            else:
                print("装板检测开关已关闭（十字交叉点检测仍使用该模型，模型保持加载）")
            return {"success": True, "message": "装板检测模型已禁用"}

    def enable_model_u(self):
        """启用U型件检测模型"""
        with self.model_lock:
            if not self.model_u_loaded:
                print("正在加载U型件检测模型...")
                self.model_u = self._load_model_to_gpu(self.model_path_u)
                self.model_u_loaded = True
                print(f"U型件检测模型加载成功: {self.model_path_u}")
            self.model_u_switch = True
            return {"success": True, "message": "U型件检测模型已启用"}

    def disable_model_u(self):
        """禁用U型件检测模型"""
        with self.model_lock:
            self.model_u_switch = False
            if self.model_u_loaded:
                print("正在卸载U型件检测模型...")
                self._unload_model_from_gpu(self.model_u)
                self.model_u = None
                self.model_u_loaded = False
                self.detection_result_u = None
                print("U型件检测模型已卸载，显存已释放")
            return {"success": True, "message": "U型件检测模型已禁用"}

    def enable_model_slot(self):
        """启用十字接缝检测模型"""
        with self.model_lock:
            if not self.model_slot_loaded:
                print("正在加载十字接缝检测模型...")
                self.model_slot = self._load_model_to_gpu(self.model_path_slot)
                self.model_slot_loaded = True
                print(f"十字接缝检测模型加载成功: {self.model_path_slot}")
            self.model_slot_switch = True
            return {"success": True, "message": "十字接缝检测模型已启用"}

    def disable_model_slot(self):
        """禁用十字接缝检测模型"""
        with self.model_lock:
            self.model_slot_switch = False
            if self.model_slot_loaded:
                print("正在卸载十字接缝检测模型...")
                self._unload_model_from_gpu(self.model_slot)
                self.model_slot = None
                self.model_slot_loaded = False
                self.detection_result_slot = None
                self.detection_result_slot_origin = None
                print("十字接缝检测模型已卸载，显存已释放")
            return {"success": True, "message": "十字接缝检测模型已禁用"}

    def enable_model_detect(self):
        """启用螺钉反馈检测模型"""
        with self.model_lock:
            if not self.model_detect_loaded:
                print("正在加载螺钉反馈检测模型...")
                self.model_detect = self._load_model_to_gpu(self.model_path_detect)
                self.model_detect_loaded = True
                print(f"螺钉反馈检测模型加载成功: {self.model_path_detect}")
            self.model_detect_switch = True
            return {"success": True, "message": "螺钉反馈检测模型已启用"}

    def disable_model_detect(self):
        """禁用螺钉反馈检测模型"""
        with self.model_lock:
            self.model_detect_switch = False
            if self.model_detect_loaded:
                print("正在卸载螺钉反馈检测模型...")
                self._unload_model_from_gpu(self.model_detect)
                self.model_detect = None
                self.model_detect_loaded = False
                self.detection_result_detect = None
                print("螺钉反馈检测模型已卸载，显存已释放")
            return {"success": True, "message": "螺钉反馈检测模型已禁用"}

    def get_model_status(self):
        """获取所有模型的状态"""
        return {
            "stud": {
                "loaded": self.model_stud_loaded,
                "enabled": self.model_stud_switch,
                "path": self.model_path_stud
            },
            "cross": {
                "loaded": self.model_cross_loaded,
                "enabled": self.model_cross_switch,
                "path": self.model_path_cross
            },
            "board": {
                "loaded": self.model_cross_loaded,
                "enabled": self.model_board_switch,
                "path": self.model_path_cross
            },
            "u": {
                "loaded": self.model_u_loaded,
                "enabled": self.model_u_switch,
                "path": self.model_path_u
            },
            "slot": {
                "loaded": self.model_slot_loaded,
                "enabled": self.model_slot_switch,
                "path": self.model_path_slot
            },
            "detect": {
                "loaded": self.model_detect_loaded,
                "enabled": self.model_detect_switch,
                "path": self.model_path_detect
            }
        }
    # =====================================================================

    # ===================== 【新增：伽马调整相关方法】=====================
    def calculate_brightness(self, image):
        """计算图片平均亮度（0~255，数值越小越暗）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def adjust_gamma(self, image, gamma=2.0):
        """伽马提亮：gamma越大越亮，1.0=原图"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        # print(f"[{datetime.now().strftime('%H:%M:%S')}]已使用gamma校正增强图像")
        return cv2.LUT(image, table)

    def adjust_contrast(self, img, alpha=1.0):
        """调整对比度"""
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    def process_frame(self, frame):
        """处理帧（伽马+对比度）"""
        # 计算亮度
        avg_bright = self.calculate_brightness(frame)
        is_dark = avg_bright < DARK_THRESHOLD
        
        # 选择伽马值
        if is_dark:
            gamma = random.uniform(*DARK_GAMMA_RANGE) if USE_RANDOM_PARAMS else 2.2
        else:
            gamma = BRIGHT_GAMMA
        
        # 选择对比度
        contrast = random.uniform(*CONTRAST_RANGE) if USE_RANDOM_PARAMS else 1.0
        
        # 应用伽马和对比度调整
        frame_gamma = self.adjust_gamma(frame, gamma=gamma)
        frame_processed = self.adjust_contrast(frame_gamma, alpha=contrast)
        
        return frame_processed, avg_bright, gamma
    # =====================================================================

    def connect_camera(self):
        """连接相机（防止重复连接）"""
        if self.camera_connected:
            return True
        if self.camera.connect():
            self.camera_connected = True
            self.last_frame_time = time.time()  # 初始化最后帧时间
            print("相机连接成功！")
            self.camera.start_video_stream(callback=self._frame_callback)
            return True
        else:
            print("相机连接失败！")
            return False

    def disconnect_camera(self):
        """断开相机连接"""
        self.camera_connected = False
        self.camera.stop_video_stream()
        self.camera.disconnect()
        print("相机已断开连接")

    def reconnect_camera(self):
        """尝试重新连接相机"""
        if self.is_reconnecting:
            return False  # 已经在重连中，避免重复

        self.is_reconnecting = True
        print(f"尝试重新连接相机... (第 {self.reconnect_attempts + 1} 次)")

        # 清理旧连接
        try:
            self.camera.stop_video_stream()
            self.camera.disconnect()
        except Exception as e:
            print(f"清理旧连接时出错（可忽略）: {e}")

        # 等待资源释放
        time.sleep(0.5)

        # 重新创建相机对象
        self.camera = Camera(device_index=self.camera_index)

        # 尝试连接
        if self.camera.connect():
            self.camera_connected = True
            self.reconnect_attempts = 0  # 重置计数
            self.is_reconnecting = False

            print("相机重连成功！启动视频流...")
            self.camera.start_video_stream(callback=self._frame_callback)

            # 等待第一帧
            print("等待第一帧...")
            wait_start = time.time()
            while time.time() - wait_start < 2.0:
                with self.frame_lock:
                    if self.current_frame is not None:
                        # 重置最后帧时间
                        self.last_frame_time = time.time()
                        print("已接收到视频流，重连完成！")
                        return True
                time.sleep(0.1)

            print("警告：相机已连接但未收到视频帧")
            return True
        else:
            self.reconnect_attempts += 1
            self.is_reconnecting = False
            print(f"相机重连失败 (尝试 {self.reconnect_attempts} 次)")
            return False

    def _frame_callback(self, frame):
        """帧回调函数"""
        with self.frame_lock:
            self.current_frame = frame.copy()

        # 更新全局共享帧（其他模块可以直接访问）
        set_shared_frame(frame)

        # 记录最后接收到帧的时间
        self.last_frame_time = time.time()

        # 通知检测线程有新帧
        self.new_frame_event.set()

    def _detection_loop(self):
        """独立的检测线程循环，在后台持续运行"""
        while self.running:
            try:
                # 等待新帧到达
                self.new_frame_event.wait(timeout=0.1)
                self.new_frame_event.clear()

                if not self.running:
                    break

                # 执行检测
                with self.frame_lock:
                    if self.current_frame is None:
                        continue
                    frame_to_detect = self.current_frame.copy()

                # 检测信息置零
                detection_info_stud = None
                detection_info_cross_all = None
                detection_info_cross_single = None
                detection_info_board_position = None
                detection_info_board_angle = None
                detection_info_u = None
                detection_info_slot = None
                detection_info_detect = None

                intensive_frame, avg_bright, gamma = self.process_frame(frame_to_detect)

                # 螺钉检测 - 先处理帧（伽马调整）
                if self.model_stud_switch:
                    results_stud = self.model_stud.predict(
                        source=intensive_frame,
                        imgsz=1280,
                        conf=0,
                        save=False,
                        show=False,
                        verbose=False,
                        stream=True,
                        max_det=1
                        
                    )
                    detection_info_stud = self._parse_detection_results(results_stud, self.model_stud)

                # 十字交叉点检测，碰钉十字线和装板十字线定位公用同一模型，检测结果中区分
                if self.model_cross_switch or self.model_board_switch:
                    results_cross = self.model_cross.predict(
                        source=intensive_frame,
                        imgsz=1280,
                        conf=0,
                        save=False,
                        show=False,
                        verbose=False,
                        stream=True,
                        max_det=2,#若后续装板十字线设置3个检测点，则这里需要改为3，请注意！！！
                        # save_txt=True,
                        # save_conf=True,
                    )
                    detection_info_cross_all, detection_info_cross_single = self._parse_detection_results_with_confidence(results_cross, self.model_cross)
                    detection_info_board_angle = self.calculate_board_angle(detection_info_cross_all)
                    detection_info_board_position = self.calculate_board_position(detection_info_cross_all)
    
                if self.model_u_switch:
                    # U型件六角螺套检测
                    results_u = self.model_u.predict(
                        source=intensive_frame,
                        imgsz=1280,
                        conf=0.25,
                        save=False,
                        show=False,
                        verbose=False,
                        stream=True,
                        max_det=1
                    )
                    detection_info_u = self._parse_detection_results(results_u, self.model_u)
                if self.model_slot_switch:
                    results_slot = self.model_slot.predict(
                        source=intensive_frame,
                        imgsz=1280,
                        conf=0.25,
                        save=False,
                        show=False,
                        verbose=False,
                        stream=True,
                        max_det=1
                    )
                    detection_info_slot_origin = self._parse_detection_results(results_slot, self.model_slot)#保存原始检测点位置信息，供后续计算使用
                    detection_info_slot = self.calculate_slot_position(detection_info_slot_origin)#计算十字接缝位置（使用所有点的几何中心）

                if self.model_detect_switch:
                    results_detect = self.model_detect.predict(
                        source=intensive_frame,
                        imgsz=1280,
                        conf=0.1,
                        save=False,
                        show=False,
                        verbose=False,
                        stream=True
                    )
                    detection_info_detect = self._parse_detection_results_box(results_detect, self.model_detect)

                # 更新检测结果（使用锁保证线程安全）
                with self.frame_lock:
                    self.detection_result_stud = detection_info_stud
                    self.detection_result_cross = detection_info_cross_single
                    self.detection_result_board = detection_info_cross_all
                    self.detection_result_u = detection_info_u
                    self.detection_result_board_angle = detection_info_board_angle
                    self.detection_result_board_position = detection_info_board_position
                    self.detection_result_slot = detection_info_slot
                    self.detection_result_detect = detection_info_detect

            except Exception as e:
                print(f"检测线程出错: {e}")
                time.sleep(0.1)

        print("检测线程已停止")

    def calculate_slot_position(self, slot_points):
        """计算十字接缝位置（使用所有点的几何中心）"""
        if not slot_points or len(slot_points) < 2:
            return None
        
        # 转换为 numpy 数组以便计算
        points_array = np.array(slot_points)
        
        # 计算所有点的几何中心（质心）
        center_x = round(np.mean(points_array[:, 0]), 2)
        center_y = round(np.mean(points_array[:, 1]), 2)
        
        return (center_x, center_y)

    def calculate_board_angle(self, cross_points):
        """计算装板角度（优化版本）"""
        if len(cross_points) == 2:
            # 按 X 坐标排序，保证方向一致
            sorted_points = sorted(cross_points, key=lambda p: p[0])
            x1, y1 = sorted_points[0]
            x2, y2 = sorted_points[1]
            
            # 处理垂直线
            if abs(x2 - x1) < 1e-6:
                return round(np.pi / 2 if y2 > y1 else -np.pi / 2, 5)
            
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            return round(angle_rad, 5)
            
        elif len(cross_points) >= 3:
            # 使用更多点进行拟合
            x_coords = np.array([pt[0] for pt in cross_points])
            y_coords = np.array([pt[1] for pt in cross_points])
            
            # 检查垂直线
            if np.ptp(x_coords) < 1e-6:  # ptp = peak to peak (max - min)
                return round(np.pi / 2, 5)
            
            # 最小二乘拟合
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
            angle_rad = np.arctan(m)
            return round(angle_rad, 5)
        
        return None
    
    def calculate_board_position(self, cross_points):
        """计算装板位置（使用所有点的几何中心）"""
        # print(f"计算装板位置，输入十字交叉点: {cross_points}")
        if not cross_points or len(cross_points) < 2:
            return None
        
        # 转换为 numpy 数组以便计算
        points_array = np.array(cross_points)
        
        # 计算所有点的几何中心（质心）
        center_x = round(np.mean(points_array[:, 0]), 2)
        center_y = round(np.mean(points_array[:, 1]), 2)
        # print(f"装板位置: ({center_x}, {center_y})")
        return (center_x, center_y)
         
    


    def _parse_detection_results(self, results, model):
        """解析检测结果"""
        detection_info = []
        for result in results:
            keypoints = result.keypoints
            # print(keypoints)
            if keypoints is not None and keypoints.xy is not None:
                for obj_idx, kpt in enumerate(keypoints.xy):
                    for point_idx, (x, y) in enumerate(kpt):
                        pixel_x = round(x.item(), 2)
                        pixel_y = round(y.item(), 2)
                        detection_info.append([pixel_x,pixel_y])
        return detection_info
    
    def _parse_detection_results_box(self, results, model):
        """解析检测结果 - 返回检测框的坐标"""
        box_info = []
        for result in results:
            if result.boxes is not None:
                # print(result.boxes)
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    box_info.append([round(x1.item(), 2), round(y1.item(), 2), round(x2.item(), 2), round(y2.item(), 2)])
        return box_info


    def _parse_detection_results_with_confidence(self, results, model):
        """解析检测结果 - 返回所有点和置信度最大的点"""
        all_points = []
        # 根据试验，置信度最大的点为检测结果中的第一个点（即使模型设置了max_det>1），因此直接取第一个点的置信度进行比较即可，无需遍历所有点的置信度
        best_point = None
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None and keypoints.xy is not None:
                for obj_idx, kpt in enumerate(keypoints.xy):
                    try:
                        x, y = kpt[0]
                        point = [round(float(x), 2), round(float(y), 2)]
                        all_points.append(point)
                        
                        if obj_idx == 0 or best_point is None:  # 直接取第一个点作为最佳点
                            best_point = point
                    except Exception as e:
                        print(f"解析关键点出错: {e}")
                        continue
        
        single_point = [best_point] if best_point is not None else []
        return all_points, single_point



    def draw_detection_result(self):
        """绘制检测结果"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # 绘制螺钉（蓝色）
        if self.detection_result_stud:
            for info in self.detection_result_stud:
                x, y = int(info[0]), int(info[1])
                cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

        # 绘制十字交叉点（红色）
        if self.detection_result_cross:
            for info in self.detection_result_cross:
                x, y = int(info[0]), int(info[1])
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        # 绘制装板（紫色）
        if self.detection_result_board:
            for info in self.detection_result_board:
                x, y = int(info[0]), int(info[1])
                cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)

        # 绘制装板角度信息
        if self.detection_result_board_angle is not None:
            degree = round(deg(self.detection_result_board_angle), 2)
            cv2.putText(frame, f"Board Angle: {degree} degree", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Board Angle: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 绘制检测目标数量和检测框
        detect_count = len(self.detection_result_detect) if self.detection_result_detect else 0
        cv2.putText(frame, f"Detect Count: {detect_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if self.detection_result_detect:
            for box in self.detection_result_detect:
                try:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                except Exception as e:
                    print(f"绘制检测框出错: {e}, box={box}")
    
        # 绘制装板中心点位置信息(绿色)
        if self.detection_result_board_position is not None:
            pos_x, pos_y = self.detection_result_board_position
            cv2.circle(frame, (int(pos_x), int(pos_y)), 6, (0, 255, 0), -1)

        # 绘制U型件六角螺套中心（黄色）
        if self.detection_result_u is not None:
            for info in self.detection_result_u:
                x, y = int(info[0]), int(info[1])
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

        # 绘制十字接缝中心（紫色）
        if self.detection_result_slot is not None:
            x, y = int(self.detection_result_slot[0]), int(self.detection_result_slot[1])
            cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
        
        # 绘制十字接缝4个检测点（紫色）
        if self.detection_result_slot_origin is not None:
            for info in self.detection_result_slot_origin:
                x, y = int(info[0]), int(info[1])
                cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)

        return frame

    def run(self, window_name="Real-Time Dual Model Detection"):
        """启动检测主循环"""

        # 初始化最后帧时间（即使没有相机也设置）
        if self.last_frame_time == 0:
            self.last_frame_time = time.time()

        # 尝试连接相机，如果失败也不退出，而是进入主循环持续重连
        self.connect_camera()

        self.running = True  # 标记为运行中

        # 启动检测线程
        self.detection_thread = Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("检测线程已启动")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.moveWindow(window_name, 100, 100)

        print("实时模型检测已启动，按 'q' 或 'ESC' 退出...")
        print("显示和检测已分离，帧率已提升")
        print("相机断连自动重连已启用")
        try:
            while self.running:
                # 检查相机状态
                current_time = time.time()
                time_since_last_frame = current_time - self.last_frame_time

                # 如果超过阈值没有收到帧，尝试连接/重连
                if time_since_last_frame > self.disconnect_threshold and not self.is_reconnecting:
                    if not self.camera_connected:
                        print(f"未检测到相机，尝试连接... ({time_since_last_frame:.1f}秒无帧)")
                    else:
                        print(f"检测到相机可能已断开 ({time_since_last_frame:.1f}秒无帧)")
                        self.disconnect_camera()

                    # 尝试连接/重连
                    while self.running and not self.is_reconnecting:
                        if self.reconnect_camera():
                            with self.frame_lock:
                                self.current_frame = None
                            break
                        else:
                            print(f"等待 {self.reconnect_interval} 秒后重试...")
                            time.sleep(self.reconnect_interval)

                # 直接显示当前帧（不等待检测完成）
                display_frame = self.draw_detection_result()

                if display_frame is not None:
                    cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF  # 改为1ms以获得更高帧率
                if key in [ord('q'), ord('Q'), 27]:
                    self.running = False
                    print("\n检测退出中...")
                    break

        except Exception as e:
            print(f"\n检测过程出错：{str(e)}")
            import traceback
            traceback.print_exc()
            self.running = False
        finally:
            self.running = False

            # 等待检测线程结束
            if self.detection_thread and self.detection_thread.is_alive():
                print("等待检测线程结束...")
                self.detection_thread.join(timeout=2.0)

            self.camera.stop_video_stream()
            self.camera.disconnect()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("实时检测已停止，所有资源已释放")

    def get_result(self):
        """修复：无论是否运行都返回当前最新结果"""
        result = {
            "stud": self.detection_result_stud if self.detection_result_stud else [],
            "cross": self.detection_result_cross if self.detection_result_cross else [],
            "board_angle": self.detection_result_board_angle if self.detection_result_board_angle is not None else None,
            "board_position": self.detection_result_board_position if self.detection_result_board_position is not None else None,
            "u": self.detection_result_u if self.detection_result_u else [],
            "slot": self.detection_result_slot if self.detection_result_slot else None,
            "detect_count": len(self.detection_result_detect) if self.detection_result_detect is not None else 0,
        }
        print(f"获取检测结果 - 螺钉：{result['stud']} | 十字交叉点：{result['cross']} | 板角度：{result['board_angle']} | 板位置：{result['board_position']} | U型件：{result['u']} | 十字接缝：{result['slot']} | 检测到螺钉数量：{result['detect_count']}")
        return result

# 全局配置
CAMERA_INDEX = 0
MODEL_PATH_STUD = r"D:\_Project\视觉相机\ImageRecognition\runs\pose\train-stud\weights\stud-best-1280-15.pt"
MODEL_PATH_CROSS = r"D:\_Project\视觉相机\ImageRecognition\runs\pose\train-cross\weights\yolo11-cross-x-1280-9.pt"
MODEL_PATH_U = r"D:\_Project\视觉相机\ImageRecognition\runs\pose\train-u\weights\u-l-1280-3.pt"
MODEL_PATH_SLOT = r"D:\_Project\视觉相机\ImageRecognition\runs\pose\train-slot\weights\slot-1280-2.pt"
MODEL_PATH_DETECT = r"D:\_Project\视觉相机\ImageRecognition\runs\detect\train-nail\weights\detect-1280-8.pt"
SAVE_FRAME_PATH = "data/frame.jpg"
# DETECT_CONF = 0.05

# 创建全局唯一实例（关键：确保所有地方导入的是同一个）
detector = RealTimePoseDetector(
    camera_index=CAMERA_INDEX,
    model_path_stud=MODEL_PATH_STUD,
    model_path_cross=MODEL_PATH_CROSS,
    model_path_u=MODEL_PATH_U,
    model_path_slot=MODEL_PATH_SLOT,
    model_path_detect=MODEL_PATH_DETECT,
    save_frame_path=SAVE_FRAME_PATH
)

if __name__ == "__main__":
    # 启动检测（单独运行时执行）
    detector.run()