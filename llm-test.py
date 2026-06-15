# ... existing code ...
from tracemalloc import start
import requests
import base64
import json
import socket
import time
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageGrab
import datetime
import cv2
import numpy as np
from camera import Camera  # 导入相机类

# ===================== 配置区（请修改这里）=====================
# API配置
USE_LOCAL_MODEL = True  # True=使用本地模型, False=使用阿里云API
LOCAL_API_URL = "http://192.168.0.150:8000/v1/chat/completions"  # 本地模型API地址
LOCAL_API_KEY = ""  # 本地模型如果需要密钥则填写，否则留空

ALIYUN_API_KEY = "sk-726d3c6af38e49b7b4cb7fe51f7ef178"  # 阿里云API Key
ALIYUN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # 阿里云API地址

# 图片源配置
USE_CAMERA = True  # False=屏幕截图, True=使用相机输入
USE_SHARED_CAMERA = False  # True=共享realtime_test_origin.py中的相机, False=创建新的相机实例
USE_SOCKET_CAMERA = True  # True=从server_socket.py获取视频流, False=直接导入相机实例
SOCKET_HOST = 'localhost'  # Socket服务器地址
SOCKET_PORT = 65432  # Socket服务器端口
CAMERA_INDEX = 0  # 相机索引（当USE_CAMERA=True且USE_SHARED_CAMERA=False且USE_SOCKET_CAMERA=False时使用）

IMAGE_PATH = "E:/code/ImageRecognition/test/images8/manual_1774254735101.jpg"
IMAGE_FOLDER = "/home/smartrobot/LLM/test/images-cross/"  # 图片文件夹路径（用于批量处理）
OUTPUT_FOLDER = "marked_images"  # 标记图片输出文件夹
MAX_WORKERS = 5 # 并发请求数（建议3-10，根据API限流调整）
MODEL_NAME = "local"  # 模型名称（本地模型请填写实际模型名）

# 任务模式选择
TASK_MODE = "coordinate"  # "coordinate"=坐标识别模式, "description"=描述模式

# 截图循环模式配置
SCREENSHOT_INTERVAL = 0.05  # 截图间隔时间（秒）
MAX_SCREENSHOT_LOOPS = 0  # 最大循环次数（0表示无限循环，直到手动停止）
SAVE_SCREENSHOTS = False  # 是否保存截图（True=保存到screenshots文件夹，False=处理后删除）

# 相机配置（当USE_CAMERA=True时使用）
CAMERA_FRAME_RETRIES = 5  # 获取相机帧时的最大重试次数
CAMERA_RETRY_INTERVAL = 0.1  # 获取相机帧失败时的重试间隔（秒）

# 日志配置
LOG_DATE = datetime.datetime.now().strftime("%Y%m%d")
LOG_FILE = f"screenshot_descriptions_{LOG_DATE}.log"  # 日志文件路径（包含日期）
ENABLE_STREAMING = True  # 是否启用流式输出
SHOW_RECENT_LOGS = True  # 启动时显示最近的日志条目
RECENT_LOG_COUNT = 5  # 显示最近的日志条目数量

# 方框识别任务的问题
COORDINATE_QUESTION = """请找到图中黑色十字线，并给出选定方框。请严格按照以下JSON格式返回结果，像素坐标采用0-1000的归一化坐标：
```json
[
    {"point_1": [x, y], "point_2": [x, y]}
]```"""

# 描述任务的问题
DESCRIPTION_QUESTION = """定义：图片为绑定在机械臂上的摄像头拍摄，尺寸为2048*3072，其中x轴范围为0-3072，y轴范围为0-2048。任务：描述图中金属圆盘的位置（用归一化坐标表示），并推断机械臂的移动方向和大概距离（用归一化坐标表示）使零件在图像中央，限制在50字以内。示例：
金属圆盘在图像右下方，坐标为(0.8, 0.5)，机械臂需要向左移动，移动距离约为(0.3, 0)。或者没有识别到金属圆盘"""

# 根据模式选择问题
QUESTION = COORDINATE_QUESTION if TASK_MODE == "coordinate" else DESCRIPTION_QUESTION
# ==============================================================

def encode_image_to_base64(image_path):
    """将本地图片转换为base64编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_and_convert_coordinates(answer):
    """从AI返回的结果中提取矩形框两个点的坐标并转换为像素坐标"""
    try:
        # 尝试从回答中提取JSON部分
        json_start = answer.find('```json')
        if json_start != -1:
            json_end = answer.find('```', json_start + 7)
            if json_end != -1:
                json_str = answer[json_start + 7:json_end].strip()
            else:
                json_str = answer[json_start + 7:].strip()
        else:
            json_str = answer.strip()

        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0:
            point = data[0]
            point_1 = point.get('point_1', [])
            point_2 = point.get('point_2', [])

            if len(point_1) == 2 and len(point_2) == 2:
                # 提取两个点的归一化坐标
                x1_norm = point_1[0]
                y1_norm = point_1[1]
                x2_norm = point_2[0]
                y2_norm = point_2[1]

                # 转换为像素坐标
                pixel_x1 = round(x1_norm / 1000 * 3072, 2)
                pixel_y1 = round(y1_norm / 1000 * 2048, 2)
                pixel_x2 = round(x2_norm / 1000 * 3072, 2)
                pixel_y2 = round(y2_norm / 1000 * 2048, 2)

                return {
                    "point_1": {"pixel_x": pixel_x1, "pixel_y": pixel_y1, "norm_x": x1_norm, "norm_y": y1_norm},
                    "point_2": {"pixel_x": pixel_x2, "pixel_y": pixel_y2, "norm_x": x2_norm, "norm_y": y2_norm}
                }
            else:
                return None
        else:
            return None

    except json.JSONDecodeError:
        return None
    except Exception:
        return None 

def mark_rectangle_on_image(image_path, point1, point2, output_path):
    """在图片上标记矩形框并保存"""
    try:
        # 打开图片
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # 提取两个点的坐标
        x1, y1 = point1['pixel_x'], point1['pixel_y']
        x2, y2 = point2['pixel_x'], point2['pixel_y']

        # 定义矩形框的边界（确保坐标从小到大）
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        # 定义标记样式
        line_width = 5  # 线条宽度

        # 绘制矩形框（红色）
        draw.rectangle([left, top, right, bottom], outline='red', width=line_width)

        # 在两个角点绘制圆点标记
        point_radius = 8
        for px, py in [(x1, y1), (x2, y2)]:
            bbox = [
                px - point_radius,
                py - point_radius,
                px + point_radius,
                py + point_radius
            ]
            draw.ellipse(bbox, fill='blue', outline='blue')

        # 保存图片
        img.save(output_path)

        return True
    except Exception as e:
        print(f"   ⚠️ 标记图片失败: {str(e)}")
        return False

def log_description(timestamp, description, elapsed_time, loop_count):
    """记录描述到日志文件"""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"序号: #{loop_count}\n")
            f.write(f"耗时: {elapsed_time:.2f}秒\n")
            f.write(f"描述:\n{description}\n")
            f.write(f"{'='*70}\n")
    except Exception as e:
        print(f"⚠️  日志记录失败: {str(e)}")

class SocketCamera:
    """从Socket服务器获取视频流的相机类"""
    def __init__(self, host='localhost', port=65432):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False

    def connect(self):
        """连接到Socket服务器"""
        if self.connected and self.sock is not None:
            return True

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.connected = True
            print(f"✅ 已连接到Socket服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接Socket服务器失败: {str(e)}")
            self.connected = False
            self.sock = None
            return False

    def get_frame(self):
        """从Socket服务器获取一帧图像"""
        # 如果未连接，尝试连接
        if not self.connected:
            if not self.connect():
                return None

        try:
            # 发送获取帧请求
            request = 'get_frame\n'
            self.sock.sendall(request.encode('utf-8'))

            # 先接收4字节的数据长度
            length_bytes = self._recv_exact(4)
            if not length_bytes:
                print(f"⚠️  无法接收数据长度")
                self.connected = False
                self.sock = None
                return None

            data_length = int.from_bytes(length_bytes, byteorder='big')

            # 再接收完整的数据
            response_data = self._recv_exact(data_length)
            if not response_data:
                print(f"⚠️  无法接收完整数据（需要{data_length}字节）")
                self.connected = False
                self.sock = None
                return None

            # 解析JSON响应
            response = json.loads(response_data.decode('utf-8'))

            # 解析响应
            if response.get('success'):
                # 解码base64图像数据
                frame_base64 = response.get('frame')
                if not frame_base64:
                    print(f"⚠️  响应中没有frame数据")
                    return None

                frame_data = base64.b64decode(frame_base64)
                # 转换为OpenCV格式
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"⚠️  图像解码失败")
                    return None
                return frame
            else:
                error = response.get('error', 'Unknown error')
                print(f"⚠️  服务器返回错误: {error}")
                return None

        except socket.timeout:
            print(f"⚠️  Socket接收超时")
            self.connected = False
            self.sock = None
            return None
        except Exception as e:
            print(f"⚠️  获取帧失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.connected = False
            self.sock = None
            return None

    def _recv_exact(self, n):
        """精确接收n字节数据"""
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def disconnect(self):
        """断开连接"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False

def view_logs(lines_count=20):
    """查看日志文件"""
    if not os.path.exists(LOG_FILE):
        print(f"📝 日志文件不存在: {LOG_FILE}")
        return

    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)

        print(f"\n📝 日志文件: {LOG_FILE} (共{total_lines}行)")
        print(f"{'='*70}")

        # 显示最后N行
        start_line = max(0, total_lines - lines_count)
        for i, line in enumerate(lines[start_line:], start=start_line + 1):
            print(f"{i:4d}: {line}", end='')

        print(f"{'='*70}\n")
    except Exception as e:
        print(f"❌ 读取日志失败: {str(e)}")

def clear_logs():
    """清空日志文件"""
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            print(f"✅ 日志文件已清空: {LOG_FILE}")
        else:
            print(f"📝 日志文件不存在: {LOG_FILE}")
    except Exception as e:
        print(f"❌ 清空日志失败: {str(e)}")

def capture_screen_description():
    """循环截图并使用AI描述内容（支持流式输出和日志记录）"""
    print(f"\n{'='*70}")
    image_source = "相机输入" if USE_CAMERA else "屏幕截图"
    print(f"🖼️  图片描述模式启动 - {image_source}")
    print(f"{'='*70}")
    print(f"⚙️  配置信息:")
    print(f"   • 图片源: {'相机' if USE_CAMERA else '屏幕截图'}")
    if USE_CAMERA:
        if USE_SOCKET_CAMERA:
            print(f"   • 相机模式: Socket客户端（从server_socket.py获取视频流）")
            print(f"   • Socket地址: {SOCKET_HOST}:{SOCKET_PORT}")
        elif USE_SHARED_CAMERA:
            print(f"   • 相机模式: 全局共享帧（realtime_test_origin.py）")
        else:
            print(f"   • 相机模式: 独立相机实例")
            print(f"   • 相机索引: {CAMERA_INDEX}")
        print(f"   • 相机重试间隔: {CAMERA_RETRY_INTERVAL}秒")
        print(f"   • 相机重试次数: {CAMERA_FRAME_RETRIES}次")
    print(f"   • 截图间隔: {SCREENSHOT_INTERVAL}秒")
    print(f"   • 最大循环: {'无限' if MAX_SCREENSHOT_LOOPS == 0 else MAX_SCREENSHOT_LOOPS}次")
    print(f"   • 流式输出: {'启用' if ENABLE_STREAMING else '禁用'}")
    print(f"   • 保存截图: {'是' if SAVE_SCREENSHOTS else '否'}")
    print(f"   • 日志文件: {LOG_FILE}")
    print(f"   • 模型: {MODEL_NAME}")
    print(f"{'='*70}")

    # 显示最近的日志记录（如果存在）
    if SHOW_RECENT_LOGS and os.path.exists(LOG_FILE):
        print(f"\n📋 最近{RECENT_LOG_COUNT}条日志记录:")
        print(f"{'='*70}")
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                # 按分隔符分割日志条目
                log_entries = content.split(f"\n{'='*70}\n")
                # 显示最后N条
                recent_entries = log_entries[-(RECENT_LOG_COUNT + 1):-1]  # 排除最后的空条目
                for entry in recent_entries:
                    lines = entry.strip().split('\n')
                    for line in lines:
                        print(f"  {line}")
                    print()
        except Exception as e:
            print(f"  ⚠️  读取日志失败: {str(e)}")
        print(f"{'='*70}\n")

    print(f"💡 提示: 按 Ctrl+C 停止循环\n")

    # 创建截图文件夹
    if SAVE_SCREENSHOTS:
        screenshot_dir = f"screenshots_{LOG_DATE}"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        print(f"📁 截图将保存到: {screenshot_dir}\n")
    else:
        # 创建临时文件夹保存截图
        temp_dir = "temp_screenshots"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    # 初始化相机（如果使用相机模式）
    camera_instance = None
    socket_camera = None
    current_frame = None
    frame_lock = None
    shared_detector = None

    if USE_CAMERA:
        try:
            from threading import Lock

            if USE_SOCKET_CAMERA:
                # 使用Socket相机（从server_socket.py获取视频流）
                print(f"📷 正在初始化Socket相机...")
                socket_camera = SocketCamera(host=SOCKET_HOST, port=SOCKET_PORT)

                # 测试获取一帧
                print(f"⏳ 测试获取帧...")
                test_frame = socket_camera.get_frame()
                if test_frame is not None:
                    print(f"✅ Socket相机准备就绪")
                else:
                    print(f"❌ 无法从Socket获取帧")
                    print(f"💡 建议：确保server_socket.py正在运行")
                    return

            elif USE_SHARED_CAMERA:
                # 使用共享相机（从realtime_test_origin.py导入detector）
                print(f"📷 正在导入共享相机...")
                try:
                    from realtime_test_origin import detector

                    # 检查相机状态
                    print(f"📊 相机状态检查:")
                    print(f"   • camera_connected: {detector.camera_connected}")
                    print(f"   • running: {detector.running}")

                    if not detector.camera_connected:
                        print(f"⚠️  相机未连接，尝试连接...")
                        if not detector.connect_camera():
                            print(f"❌ 相机连接失败")
                            print(f"💡 建议：检查相机是否被其他程序占用")
                            return
                        print(f"✅ 相机连接成功")

                    # 尝试直接从 camera.get_frame() 获取一帧来测试
                    print(f"⏳ 测试获取帧...")
                    test_frame = detector.camera.get_frame()
                    if test_frame is not None:
                        print(f"✅ 可以成功获取帧")
                    else:
                        print(f"❌ 无法获取帧")
                        print(f"💡 建议：检查相机是否正常工作")
                        return

                    print(f"✅ 共享相机准备就绪")

                except ImportError as e:
                    print(f"❌ 无法导入realtime_test_origin.py: {e}")
                    return
                except Exception as e:
                    print(f"❌ 使用共享相机失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
            else:
                # 创建独立相机实例
                frame_lock = Lock()

                print(f"📷 正在初始化相机 (索引: {CAMERA_INDEX})...")
                camera_instance = Camera(device_index=CAMERA_INDEX)

                if camera_instance.connect():
                    print(f"✅ 相机连接成功！")

                    # 定义帧回调函数
                    def frame_callback(frame):
                        nonlocal current_frame
                        with frame_lock:
                            current_frame = frame.copy()

                    # 启动视频流
                    camera_instance.start_video_stream(callback=frame_callback)
                    print(f"✅ 视频流已启动")

                    # 等待第一帧
                    print(f"⏳ 等待第一帧...")
                    wait_start = time.time()
                    while time.time() - wait_start < 3.0:
                        with frame_lock:
                            if current_frame is not None:
                                print(f"✅ 已接收到视频流")
                                break
                        time.sleep(0.1)
                    else:
                        print(f"⚠️  警告：未在3秒内接收到视频帧，将继续尝试")
                else:
                    print(f"❌ 相机连接失败！请检查相机索引和连接。")
                    return
        except Exception as e:
            print(f"❌ 相机初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return

    loop_count = 0
    start_time = time.time()

    try:
        while True:
            # 检查是否达到最大循环次数
            if MAX_SCREENSHOT_LOOPS > 0 and loop_count >= MAX_SCREENSHOT_LOOPS:
                print(f"\n✅ 已达到最大循环次数 ({MAX_SCREENSHOT_LOOPS})，停止运行")
                break

            loop_count += 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp = datetime.datetime.now().strftime("%H%M%S")

            # 获取图片（根据配置选择屏幕截图或相机）
            if USE_CAMERA:
                print(f"[{loop_count}] {datetime.datetime.now().strftime('%H:%M:%S')} - 正在从相机获取图片...", end="", flush=True)
                # 尝试多次获取相机帧
                screenshot = None
                for retry in range(CAMERA_FRAME_RETRIES):
                    if USE_SOCKET_CAMERA:
                        # 从Socket获取帧
                        frame = socket_camera.get_frame()
                        if frame is not None:
                            # 将OpenCV的BGR格式转换为RGB格式（PIL需要RGB）
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            screenshot = Image.fromarray(frame_rgb)
                            break
                    elif USE_SHARED_CAMERA:
                        # 从全局共享帧获取
                        from realtime_test_origin import get_shared_frame
                        frame = get_shared_frame()
                        if frame is not None:
                            # 将OpenCV的BGR格式转换为RGB格式（PIL需要RGB）
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            screenshot = Image.fromarray(frame_rgb)
                            break
                    else:
                        # 使用独立相机的回调帧
                        with frame_lock:
                            if current_frame is not None:
                                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                                screenshot = Image.fromarray(frame_rgb)
                                break

                    if screenshot is None:
                        time.sleep(CAMERA_RETRY_INTERVAL)

                if screenshot is None:
                    print(f" ❌ (获取失败，跳过本次循环)\n")
                    time.sleep(SCREENSHOT_INTERVAL)
                    continue
            else:
                print(f"[{loop_count}] {datetime.datetime.now().strftime('%H:%M:%S')} - 正在截图...", end="", flush=True)
                screenshot = ImageGrab.grab()

            if SAVE_SCREENSHOTS:
                screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}_{loop_count}.png")
            else:
                screenshot_path = os.path.join(temp_dir, f"screenshot_{loop_count}.png")

            screenshot.save(screenshot_path)
            print(f" ✓")

            # 调用AI描述（流式或非流式）
            if ENABLE_STREAMING:
                result = call_qwen_vision_streaming(screenshot_path)
                if result["success"]:
                    description = result["answer"].strip()
                    # 清理描述（移除可能的markdown标记）
                    if description.startswith("```"):
                        lines = description.split('\n')
                        if len(lines) > 1:
                            description = '\n'.join(lines[1:-1])
                        else:
                            description = description.replace('```', '').strip()

                    # 流式模式已经打印过内容了，这里只打印耗时和日志
                    print(f"⏱️  耗时: {result['elapsed_time']:.2f}秒\n")

                    # 记录到日志
                    log_description(current_time, description, result['elapsed_time'], loop_count)
                else:
                    print(f"❌ 描述失败: {result.get('error', '未知错误')}\n")
            else:
                result = call_qwen_vision_single(screenshot_path)

                if result["success"]:
                    description = result["answer"].strip()
                    # 清理描述（移除可能的markdown标记）
                    if description.startswith("```"):
                        lines = description.split('\n')
                        if len(lines) > 1:
                            description = '\n'.join(lines[1:-1])
                        else:
                            description = description.replace('```', '').strip()

                    # 非流式模式需要打印描述
                    print(f"📝 {description}")
                    print(f"⏱️  耗时: {result['elapsed_time']:.2f}秒\n")

                    # 记录到日志
                    log_description(current_time, description, result['elapsed_time'], loop_count)
                else:
                    print(f"❌ 描述失败: {result.get('error', '未知错误')}\n")

            # 删除临时截图文件（如果不需要保存）
            if not SAVE_SCREENSHOTS:
                try:
                    os.remove(screenshot_path)
                except:
                    pass

            # 等待指定时间
            if loop_count < MAX_SCREENSHOT_LOOPS or MAX_SCREENSHOT_LOOPS == 0:
                time.sleep(SCREENSHOT_INTERVAL)

    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print(f"\n\n{'='*70}")
        print(f"🛑 用户手动停止")
        print(f"{'='*70}")
        print(f"📊 统计信息:")
        print(f"   • 运行次数: {loop_count}次")
        print(f"   • 总运行时间: {total_time:.2f}秒")
        if loop_count > 0:
            avg_time = total_time / loop_count
            print(f"   • 平均每次: {avg_time:.2f}秒")
        print(f"   • 日志文件: {LOG_FILE}")
        if SAVE_SCREENSHOTS:
            print(f"   • 截图文件夹: {screenshot_dir}")
        print(f"{'='*70}\n")
    finally:
        # 清理相机资源（如果使用了相机）
        if USE_CAMERA and USE_SOCKET_CAMERA and socket_camera is not None:
            # 使用Socket相机时，断开Socket连接
            try:
                socket_camera.disconnect()
                print(f"✅ Socket相机已断开连接")
            except Exception as e:
                print(f"⚠️  Socket相机断开时出错: {str(e)}")
        elif USE_CAMERA and not USE_SHARED_CAMERA and camera_instance is not None:
            # 只有使用独立相机实例时才需要清理
            try:
                camera_instance.stop_video_stream()
                camera_instance.disconnect()
                print(f"✅ 相机已断开连接")
            except Exception as e:
                print(f"⚠️  相机断开时出错: {str(e)}")
        elif USE_CAMERA and USE_SHARED_CAMERA:
            # 使用全局共享帧时，不清理相机资源
            print(f"ℹ️  使用全局共享帧，不清理相机资源")

    # 清理临时文件夹
    if not SAVE_SCREENSHOTS:
        try:
            os.rmdir(temp_dir)
        except:
            pass

def call_qwen_vision_streaming(image_path):
    """调用图像识别接口（流式输出，用于描述模式）"""
    # 根据配置选择API地址和密钥
    if USE_LOCAL_MODEL:
        url = LOCAL_API_URL
        api_key = LOCAL_API_KEY
    else:
        url = ALIYUN_API_URL
        api_key = ALIYUN_API_KEY

    headers = {
        "Content-Type": "application/json"
    }

    # 如果有API密钥则添加
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 编码图片
    base64_image = encode_image_to_base64(image_path)
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": QUESTION
                    }
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": True  # 启用流式输出
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=120, stream=True)
            end_time = time.time()
            elapsed_time = end_time - start_time

            response.raise_for_status()

            # 收集流式响应
            full_content = ""
            print("🤖 ", end="", flush=True)

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end="", flush=True)
                                    full_content += content
                        except json.JSONDecodeError:
                            continue

            print()  # 换行

            return {
                "success": True,
                "answer": full_content,
                "coordinates": None,
                "elapsed_time": elapsed_time
            }

        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "error": "连接错误",
                    "detail": str(e)
                }

        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return {
                    "success": False,
                    "error": f"HTTP错误 {response.status_code}",
                    "status_code": response.status_code
                }

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return {
                    "success": False,
                    "error": "请求超时"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def call_qwen_vision_single(image_path):
    """调用图像识别接口（支持本地和云端）"""
    # 根据配置选择API地址和密钥
    if USE_LOCAL_MODEL:
        url = LOCAL_API_URL
        api_key = LOCAL_API_KEY
    else:
        url = ALIYUN_API_URL
        api_key = ALIYUN_API_KEY
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 如果有API密钥则添加
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if image_path.startswith("http"):
        image_content = {
            "type": "image_url",
            "image_url": {"url": image_path}
        }
    else:
        base64_image = encode_image_to_base64(image_path)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": QUESTION
                    }
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]

                # 根据任务模式返回不同的结果
                if TASK_MODE == "coordinate":
                    coord_result = extract_and_convert_coordinates(answer)
                    return {
                        "success": True,
                        "answer": answer,
                        "coordinates": coord_result,
                        "elapsed_time": elapsed_time
                    }
                else:  # description mode
                    return {
                        "success": True,
                        "answer": answer,
                        "coordinates": None,
                        "elapsed_time": elapsed_time
                    }
            else:
                return {
                    "success": False,
                    "error": "接口返回异常",
                    "detail": result,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "error": "连接错误",
                    "detail": str(e)
                }
                
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP错误 {response.status_code}: {response.text[:200]}"
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return {
                    "success": False,
                    "error": "请求超时"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def process_folder(folder_path):
    """批量处理文件夹内所有JPG图片（并发处理）"""
    if not os.path.exists(folder_path):
        print(f"\n❌ 错误：文件夹不存在 - {folder_path}\n")
        return []
    
    # 创建输出文件夹
    output_dir = os.path.join(os.path.dirname(folder_path) if os.path.isfile(folder_path) else folder_path, OUTPUT_FOLDER)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出文件夹: {output_dir}\n")
    
    # 查找所有jpg文件并去重
    jpg_files_lower = glob.glob(os.path.join(folder_path, "*.jpg"))
    jpg_files_upper = glob.glob(os.path.join(folder_path, "*.JPG"))
    
    # 合并并去重（使用集合去除重复路径）
    jpg_files = list(set(jpg_files_lower + jpg_files_upper))
    
    # 按文件名排序，保证处理顺序一致
    jpg_files.sort()
    
    if not jpg_files:
        print(f"\n⚠️ 警告：文件夹中没有找到JPG图片 - {folder_path}\n")
        return []

    model_type = "本地模型" if USE_LOCAL_MODEL else "阿里云API"
    print(f"\n{'='*70}")
    print(f"📁 找到 {len(jpg_files)} 张JPG图片")
    print(f"🤖 模型类型: {model_type}")
    print(f"🔗 API地址: {LOCAL_API_URL if USE_LOCAL_MODEL else ALIYUN_API_URL}")
    print(f"⚡ 并发数: {MAX_WORKERS}")
    print(f"📂 输出目录: {output_dir}")
    print(f"{'='*70}\n")

    results = []
    total_start = time.time()
    completed = 0

    def process_with_progress(img_path):
        """处理单张图片并显示进度"""
        nonlocal completed
        filename = os.path.basename(img_path)
        result = call_qwen_vision_single(img_path)
        result["filename"] = filename
        result["filepath"] = img_path

        # 根据任务模式处理结果
        if result["success"]:
            if TASK_MODE == "coordinate" and result["coordinates"]:
                # 坐标识别模式：标记图片
                coords = result["coordinates"]
                # 生成输出文件名：原文件名_result.jpg
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"{name_without_ext}_result.jpg"
                output_path = os.path.join(output_dir, output_filename)

                # 标记矩形框
                marked = mark_rectangle_on_image(
                    img_path,
                    coords['point_1'],
                    coords['point_2'],
                    output_path
                )
                result["marked_image"] = output_path if marked else None
                result["marked_success"] = marked
            elif TASK_MODE == "description":
                # 描述模式：不需要标记图片
                result["marked_image"] = None
                result["marked_success"] = False

        completed += 1
        return result

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(process_with_progress, img_path): img_path 
                        for img_path in jpg_files}
        
        for future in as_completed(future_to_path):
            img_path = future_to_path[future]
            filename = os.path.basename(img_path)
            
            try:
                result = future.result()
                results.append(result)
                
                # 美观的输出格式
                status_icon = "✅" if result["success"] else "❌"
                print(f"{status_icon} [{completed}/{len(jpg_files)}] {filename}")
                
                if result["success"]:
                    if TASK_MODE == "coordinate":
                        coords = result["coordinates"]
                        if coords:
                            print(f"   📍 点1像素坐标: ({coords['point_1']['pixel_x']}, {coords['point_1']['pixel_y']})")
                            print(f"   📍 点2像素坐标: ({coords['point_2']['pixel_x']}, {coords['point_2']['pixel_y']})")
                            print(f"   ⏱️  耗时: {result['elapsed_time']:.2f}秒")

                            # 显示标记图片状态
                            if result.get("marked_success"):
                                marked_file = os.path.basename(result["marked_image"])
                                print(f"   🖼️  已标记: {marked_file}")
                            else:
                                print(f"   ⚠️  标记图片失败")
                        else:
                            print(f"   ⚠️  坐标提取失败")
                    else:  # description mode
                        print(f"   📝 AI描述内容:")
                        # 打印AI返回的内容（限制长度避免过长）
                        description = result.get("answer", "")
                        lines = description.split('\n')
                        for line in lines[:15]:  # 最多显示15行
                            print(f"      {line}")
                        if len(lines) > 15:
                            print(f"      ... (还有{len(lines)-15}行)")
                        print(f"   ⏱️  耗时: {result['elapsed_time']:.2f}秒")
                else:
                    error_detail = result.get('error', '未知错误')
                    if len(error_detail) > 100:
                        error_detail = error_detail[:100] + "..."
                    print(f"   💥 错误: {error_detail}")
                
                print()
                
            except Exception as e:
                print(f"❌ [{completed}/{len(jpg_files)}] {filename}")
                print(f"   💥 异常: {str(e)}\n")
                results.append({
                    "filename": filename,
                    "filepath": img_path,
                    "success": False,
                    "error": str(e)
                })

    total_elapsed = time.time() - total_start
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    marked_count = sum(1 for r in results if r.get('marked_success'))

    print(f"\n{'='*70}")
    print(f"🎉 批量处理完成！")
    print(f"{'='*70}")
    print(f"📊 统计信息:")
    print(f"   • 总计: {len(jpg_files)} 张")
    print(f"   • ✅ 成功: {success_count} 张")
    print(f"   • ❌ 失败: {fail_count} 张")
    print(f"   • 🖼️  已标记: {marked_count} 张")
    print(f"   • ⏱️  总耗时: {total_elapsed:.2f}秒")
    if success_count > 0:
        avg_time = total_elapsed / len(jpg_files)
        print(f"   • 📈 平均每张: {avg_time:.2f}秒")
    print(f"   • 📂 输出目录: {output_dir}")
    print(f"{'='*70}\n")

    return results

def call_qwen_vision():
    """调用图像识别接口（单张图片）"""
    # 根据配置选择API地址和密钥
    if USE_LOCAL_MODEL:
        url = LOCAL_API_URL
        api_key = LOCAL_API_KEY
    else:
        url = ALIYUN_API_URL
        api_key = ALIYUN_API_KEY
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if IMAGE_PATH.startswith("http"):
        image_content = {
            "type": "image_url",
            "image_url": {"url": IMAGE_PATH}
        }
    else:
        base64_image = encode_image_to_base64(IMAGE_PATH)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": QUESTION
                    }
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    model_type = "本地模型" if USE_LOCAL_MODEL else "阿里云API"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 正在尝试第 {attempt + 1} 次请求... ({model_type})")
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                print(f"\n{'='*70}")
                print(f"🤖 AI原始回答:")
                print(f"{'='*70}")
                print(answer)
                print(f"{'='*70}")
                print(f"⏱️  请求耗时: {elapsed_time:.2f}秒\n")

                if TASK_MODE == "coordinate":
                    coord_result = extract_and_convert_coordinates(answer)
                    if coord_result:
                        print(f"{'='*70}")
                        print(f"📍 坐标转换结果:")
                        print(f"{'='*70}")
                        print(f"   • 点1归一化坐标: ({coord_result['point_1']['norm_x']}, {coord_result['point_1']['norm_y']})")
                        print(f"   • 点1像素坐标: ({coord_result['point_1']['pixel_x']}, {coord_result['point_1']['pixel_y']})")
                        print(f"   • 点2归一化坐标: ({coord_result['point_2']['norm_x']}, {coord_result['point_2']['norm_y']})")
                        print(f"   • 点2像素坐标: ({coord_result['point_2']['pixel_x']}, {coord_result['point_2']['pixel_y']})")
                        print(f"   • 图像尺寸: 3072 x 2048")
                        print(f"{'='*70}\n")

                        # 标记单张图片
                        output_dir = os.path.join(os.path.dirname(IMAGE_PATH), OUTPUT_FOLDER)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        filename = os.path.basename(IMAGE_PATH)
                        name_without_ext = os.path.splitext(filename)[0]
                        output_filename = f"{name_without_ext}_result.jpg"
                        output_path = os.path.join(output_dir, output_filename)

                        print(f"🖼️  正在标记图片...")
                        marked = mark_rectangle_on_image(
                            IMAGE_PATH,
                            coord_result['point_1'],
                            coord_result['point_2'],
                            output_path
                        )

                        if marked:
                            print(f"✅ 标记图片已保存: {output_path}\n")
                        else:
                            print(f"❌ 标记图片失败\n")
                
                return answer
            else:
                print(f"\n❌ 接口返回异常：")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print(f"⏱️  请求耗时: {elapsed_time:.2f}秒\n")
                
        except requests.exceptions.ConnectionError as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n❌ 连接错误（尝试 {attempt + 1}/{max_retries}）：{str(e)}")
            print(f"⏱️  已耗时: {elapsed_time:.2f}秒")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ 等待 {wait_time} 秒后重试...\n")
                time.sleep(wait_time)
            else:
                print(f"\n💥 已达到最大重试次数，请求失败\n")
                
        except requests.exceptions.HTTPError as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            error_msg = f"HTTP错误 {response.status_code}"
            print(f"\n❌ {error_msg}（尝试 {attempt + 1}/{max_retries}）")
            print(f"响应内容: {response.text[:200]}")
            print(f"⏱️  已耗时: {elapsed_time:.2f}秒")
            if attempt < max_retries - 1:
                print(f"⏳ 等待 2 秒后重试...\n")
                time.sleep(2)
            else:
                print(f"\n💥 已达到最大重试次数，请求失败\n")
                
        except requests.exceptions.Timeout:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n⏰ 请求超时（尝试 {attempt + 1}/{max_retries}）")
            print(f"⏱️  已耗时: {elapsed_time:.2f}秒")
            if attempt < max_retries - 1:
                print(f"⏳ 等待 2 秒后重试...\n")
                time.sleep(2)
            else:
                print(f"\n💥 已达到最大重试次数，请求失败\n")
                
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n💥 请求失败：{str(e)}")
            print(f"⏱️  已耗时: {elapsed_time:.2f}秒")
            if 'response' in locals():
                print(f"📄 错误详情：{response.text[:200]}")
            break

if __name__ == "__main__":

    model_type = "本地模型" if USE_LOCAL_MODEL else "阿里云API"
    api_url = LOCAL_API_URL if USE_LOCAL_MODEL else ALIYUN_API_URL

    print(f"\n{'='*70}")
    print(f"🖼️ 图像识别工具 - {model_type}")
    print(f"🔗 API: {api_url}")
    print(f"{'='*70}\n")
    print(f"请选择任务模式：")
    print(f" 1️⃣ 坐标识别模式（方框标记）")
    print(f" 2️⃣ 描述模式（屏幕截图循环）\n")

    task_choice = input(f"请输入任务模式 (1/2，直接回车默认为配置文件中的设置): ").strip()
    if task_choice == "1":
        TASK_MODE = "coordinate"
        QUESTION = COORDINATE_QUESTION
        print(f"✅ 已切换到坐标识别模式\n")
    elif task_choice == "2":
        TASK_MODE = "description"
        QUESTION = DESCRIPTION_QUESTION
        print(f"✅ 已切换到描述模式（屏幕截图循环）\n")
    else:
        print(f"✅ 使用配置文件中的设置: {TASK_MODE}模式\n")

    # 如果是描述模式，直接启动截图循环
    if TASK_MODE == "description":
        # 询问是否需要管理日志
        if os.path.exists(LOG_FILE):
            print(f"📝 检测到日志文件已存在: {LOG_FILE}")
            log_choice = input(f"请选择操作 (v=查看日志, c=清空日志, Enter=继续): ").strip().lower()
            if log_choice == 'v':
                view_count = input(f"显示最近多少条记录? (默认20): ").strip()
                try:
                    count = int(view_count) if view_count else 20
                    view_logs(count)
                except:
                    view_logs(20)
                input("按回车继续...")
            elif log_choice == 'c':
                clear_logs()
                input("按回车继续...")

        capture_screen_description()
    else:
        # 坐标识别模式，继续原有流程
        print(f"请选择处理模式：")
        print(f" 1️⃣ 单张图片处理")
        print(f" 2️⃣ 批量处理文件夹（并发加速）\n")
        choice = input(f"请输入选项 (1/2，直接回车默认为1): ").strip()

        if choice == "2":
            folder = input(f"📁 请输入文件夹路径 (默认: {IMAGE_FOLDER}): ").strip()
            if not folder:
                folder = IMAGE_FOLDER

            results = process_folder(folder)

            if results:
                output_file = "batch_results.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"💾 结果已保存到: {output_file}\n")
        else:
            call_qwen_vision()