import cv2
import numpy as np
from camera import Camera  # 导入相机类
from ultralytics import YOLO
import os
from threading import Lock, Thread, Event
import time
from PIL import Image

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
    
    def __init__(self, camera_index=0, model_path_stud=None, model_path_cross=None, save_frame_path="data/frame.jpg"):
        """防止重复初始化"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # 初始化核心属性
        self.camera_index = camera_index
        self.camera = Camera(device_index=camera_index)
        self.camera_connected = False
        self.model_stud = None
        self.model_cross = None
        self.save_frame_path = save_frame_path
        self.frame_lock = Lock()
        self.current_frame = None
        self.detection_result_stud = None
        self.detection_result_cross = None
        self.running = False  # 初始为False，避免自动运行

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
        
        # 裁切相关配置（移植自single-test-average.py）
        self.ORIG_W, self.ORIG_H = 3072, 2048
        self.CROP_W, self.CROP_H = 3032, 2008  # 3072-40=3032, 2048-40=2008
        self.offsets = [(0, 0), (-10, -10), (10, 10),(-10, 10),(10, -10)]  # 5种裁切偏移
        self.avg_result_stud = None  # 螺钉平均坐标
        self.avg_result_cross = None  # 十字交叉点平均坐标
        
        # 加载模型
        if model_path_stud and os.path.exists(model_path_stud):
            self.model_stud = YOLO(model_path_stud)
            print(f"成功加载螺钉检测模型：{model_path_stud}")
        else:
            raise ValueError(f"螺钉模型路径不存在：{model_path_stud}")
        
        if model_path_cross and os.path.exists(model_path_cross):
            self.model_cross = YOLO(model_path_cross)
            print(f"成功加载十字交叉点检测模型：{model_path_cross}")
        else:
            raise ValueError(f"十字交叉点模型路径不存在：{model_path_cross}")
        
        self._initialized = True  # 标记已初始化

    def crop_image_with_offset(self, img, crop_w, crop_h, offsets):
        """
        裁切图片并记录偏移值（移植自single-test-average.py）
        img: PIL Image对象
        return: 裁切后的图片列表，偏移值列表
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

        # 记录最后接收到帧的时间
        self.last_frame_time = time.time()

        # 通知检测线程有新帧
        self.new_frame_event.set()

    def _detect_with_crop(self, frame, model, imgsz, conf):
        """
        带裁切的检测逻辑（核心修改，修复NoneType迭代错误）
        return: 平均后的坐标列表，原始检测结果
        """
        # 将OpenCV帧转换为PIL Image并调整尺寸
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_img = Image.fromarray(frame_rgb)
        orig_img = orig_img.resize((self.ORIG_W, self.ORIG_H), Image.Resampling.LANCZOS)
        
        # 裁切图片 + 获取偏移值
        crop_imgs, crop_offsets = self.crop_image_with_offset(
            orig_img, self.CROP_W, self.CROP_H, self.offsets
        )
        
        all_valid_coords = []  # 存储所有有效坐标（映射回原图）
        
        # 遍历所有裁切图进行检测
        for crop_idx, (crop_img, (x1_offset, y1_offset)) in enumerate(zip(crop_imgs, crop_offsets)):
            # 将PIL Image转回numpy数组用于检测
            crop_img_np = np.array(crop_img)
            
            # 推理裁切图
            results = model.predict(
                source=crop_img_np,
                imgsz=imgsz,
                conf=conf,
                save=False,
                show=False,
                verbose=False,
                max_det=1,
            )
            res = results[0]
            
            # 关键修复：增加对keypoints的空值判断
            if res.keypoints is None:
                continue  # 没有检测到关键点，跳过
            
            # 解析检测结果并映射回原图
            # 修复：先判断xy是否有数据
            if len(res.keypoints.xy) > 0:
                for obj_idx, kpt in enumerate(res.keypoints.xy):
                    # 遍历每个关键点
                    for point_idx, (x, y) in enumerate(kpt):
                        # 过滤无效坐标（0值或负数）
                        if x.item() > 0 and y.item() > 0:
                            # 映射回原图坐标
                            x_orig = x.item() + x1_offset
                            y_orig = y.item() + y1_offset
                            all_valid_coords.append([round(x_orig, 2), round(y_orig, 2)])
        
        # 计算平均坐标（过滤无效点）
        avg_coords = []
        if len(all_valid_coords) > 0:
            # 按点索引分组计算平均（支持多个关键点）
            coords_np = np.array(all_valid_coords)
            avg_np = np.mean(coords_np, axis=0)
            avg_coords = [[round(avg_np[0], 2), round(avg_np[1], 2)]]
        
        return avg_coords, all_valid_coords

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

                # 螺钉检测（带裁切和平均）
                avg_stud, raw_stud = self._detect_with_crop(
                    frame_to_detect, self.model_stud, 1280, self.detection_conf
                )
                
                # 十字交叉点检测（带裁切和平均）
                avg_cross, raw_cross = self._detect_with_crop(
                    frame_to_detect, self.model_cross, 1920, self.detection_conf
                )

                # 更新检测结果（使用锁保证线程安全）
                with self.frame_lock:
                    self.detection_result_stud = raw_stud  # 原始检测结果
                    self.detection_result_cross = raw_cross  # 原始检测结果
                    self.avg_result_stud = avg_stud  # 平均后的结果
                    self.avg_result_cross = avg_cross  # 平均后的结果

            except Exception as e:
                print(f"检测线程出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        print("检测线程已停止")

    def detect_frame(self, conf=0.5):
        """执行检测（兼容原有接口）"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
        
        # 带裁切的检测
        avg_stud, raw_stud = self._detect_with_crop(
            self.current_frame, self.model_stud, 1280, conf
        )
        avg_cross, raw_cross = self._detect_with_crop(
            self.current_frame, self.model_cross, 1920, conf
        )

        # 更新结果
        with self.frame_lock:
            self.detection_result_stud = raw_stud
            self.detection_result_cross = raw_cross
            self.avg_result_stud = avg_stud
            self.avg_result_cross = avg_cross

        return {
            "stud": {"raw": raw_stud, "average": avg_stud},
            "cross": {"raw": raw_cross, "average": avg_cross}
        }

    def draw_detection_result(self):
        """绘制检测结果（增加平均点绘制）"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        # 调整帧尺寸以匹配检测坐标
        h, w = frame.shape[:2]
        scale_x = w / self.ORIG_W
        scale_y = h / self.ORIG_H
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # 绘制平均点（螺钉-青色，十字-黄色，更大更醒目）
        if self.avg_result_stud:
            for info in self.avg_result_stud:
                x = int(info[0] * scale_x)
                y = int(info[1] * scale_y)
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)  # 黄色大点
                cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)  # 黑色边框
        
        if self.avg_result_cross:
            for info in self.avg_result_cross:
                x = int(info[0] * scale_x)
                y = int(info[1] * scale_y)
                cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)  # 青色大点
                cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)  # 黑色边框

        return frame

    def run(self, conf=0.5, window_name="Real-Time Dual Model Detection"):
        """启动检测主循环"""
        self.detection_conf = conf  # 保存置信度参数

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
        print("已启用裁切平均检测，无效点将自动过滤")
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
        """获取平均后的检测结果"""
        with self.frame_lock:
            result = {
                "stud": self.avg_result_stud if self.avg_result_stud else [],
                "cross": 
                    # "raw": self.detection_result_cross if self.detection_result_cross else [],
                    self.avg_result_cross if self.avg_result_cross else []
                
            }
        print(f"获取检测结果 - 螺钉：{result['stud']} | 十字交叉点：{result['cross']}")
        print(f"原始检测点数量 - 螺钉：{len(result['stud'])} | 十字交叉点：{len(result['cross'])}")
        return result

# 全局配置
CAMERA_INDEX = 0
MODEL_PATH_STUD = r"D:\_Project\视觉相机\ImageRecognition\runs\pose\train-stud\weights\stud-x-1280-2.pt"
MODEL_PATH_CROSS = r"D:\_Project\视觉相机\ImageRecognition\runs\pose\train-cross\weights\cross-x-1920-1.pt"
SAVE_FRAME_PATH = "data/frame.jpg"
DETECT_CONF = 0.6

# 创建全局唯一实例（关键：确保所有地方导入的是同一个）
detector = RealTimePoseDetector(
    camera_index=CAMERA_INDEX,
    model_path_stud=MODEL_PATH_STUD,
    model_path_cross=MODEL_PATH_CROSS,
    save_frame_path=SAVE_FRAME_PATH
)

if __name__ == "__main__":
    # 启动检测（单独运行时执行）
    detector.run(conf=DETECT_CONF)