import socket
import threading
import json
import numpy as np
import time
from datetime import datetime
import cv2
import logging 
import sys
import os

# 关键：将realtime_test所在目录加入Python路径，确保导入的是同一个实例
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from realtime_test_origin import detector  # 导入全局检测器实例

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ],
)

def np_encoder(obj):
    """numpy类型序列化"""
    if isinstance(obj, np.ndarray):  
        return obj.tolist()
    elif isinstance(obj, np.generic):    
        return obj.item()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

class VisionAPISocketServer:
    """视觉API Socket服务端"""
    
    def __init__(self, host='0.0.0.0', port=65432):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(5)  # 增加监听队列大小

    def start(self):
        """启动服务端"""
        print(f'Server started on {self.host}:{self.port}')
        # 先启动检测器（后台线程运行，不阻塞Socket）
        threading.Thread(target=detector.run, daemon=True).start()
        time.sleep(2)  # 等待检测器初始化完成
        
        while True:
            conn, addr = self.sock.accept()
            print(f'Connected: {addr}')
            threading.Thread(target=self.handle_client, args=(conn,)).start()

    def handle_client(self, conn):
        """处理客户端请求"""
        with conn:
            while True:
                receive_time = time.time()
                receive_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                data = conn.recv(1024)
                if not data:
                    break

                req = data.decode('utf-8').strip()
                print(f'[{receive_datetime}] Received: {req}')
                
                process_start_time = time.time()
                resp = self.process_request(req)
                process_end_time = time.time()

                if resp is not None:
                    self.send_response(conn, resp, receive_time, 
                                      send_time=time.time(), 
                                      process_time=process_end_time - process_start_time)

    def process_request(self, req):
        """处理请求"""
        try:
            if req == 'cross_points_without_granding':
                detect_result = detector.get_result()
                position_pixels = detect_result['cross']
                if len(position_pixels) > 0:
                    print('cross_points successed:', position_pixels)
                    response = {'cross_points_pixels': position_pixels}
                else:
                    print('cross_points failed: 未检测到十字交叉点')
                    response = {'cross_points_pixels': []}
                return self.clean_for_utf8(response)
                    
            elif req == 'screw_points':
                detect_result = detector.get_result()
                position_pixels = detect_result['stud']
                response = {'screw_points_pixels': position_pixels if len(position_pixels) > 0 else []}
                return self.clean_for_utf8(response)
            
            elif req == 'board_angle':
                detect_result = detector.get_result()
                angle = detect_result['board_angle']
                response = {'board_angle': angle if angle is not None else None}
                return self.clean_for_utf8(response)
            elif req == 'board_position':
                detect_result = detector.get_result()
                position = detect_result['board_position']
                response = {'board_position': position if position is not None else None}
                return self.clean_for_utf8(response)
            elif req == 'u_points':
                detect_result = detector.get_result()
                u_points = detect_result['u']
                response = {'u_points': u_points if len(u_points) > 0 else []}
                return self.clean_for_utf8(response)
                
            else:
                return self.clean_for_utf8({"status": "connected", "message": "Server is ready"})
                
        except Exception as e:
            logging.error(f"处理请求时出错: {e}", exc_info=True)
            return self.clean_for_utf8({'error': str(e)})
        
    def clean_for_utf8(self, obj):
        """清理非UTF-8字符"""
        if isinstance(obj, dict):
            return {self.clean_for_utf8(k): self.clean_for_utf8(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_for_utf8(item) for item in obj]
        elif isinstance(obj, str):
            return obj.encode('utf-8', errors='ignore').decode('utf-8')
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj).encode('utf-8', errors='ignore').decode('utf-8')

    def send_response(self, conn, resp, receive_time, send_time, process_time):
        """发送响应"""
        try:
            send_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if isinstance(resp, dict):
                resp['timestamp'] = {
                    'received': send_datetime,
                    'sent': send_datetime,
                    'time_diff_ms': round((send_time - receive_time) * 1000, 3),
                    'processing_time_ms': round(process_time * 1000, 3)
                }
            
            json_str = json.dumps(resp, default=np_encoder, ensure_ascii=False)
            conn.sendall(json_str.encode('utf-8'))
            print(f'[{send_datetime}] Sent: {json_str} (耗时: {process_time*1000:.3f}ms)')
            
        except Exception as e:
            logging.error(f"发送响应时出错: {e}", exc_info=True)
            error_msg = json.dumps({'error': 'Server error', 'message': str(e)}, ensure_ascii=False)
            conn.sendall(error_msg.encode('utf-8'))

if __name__ == '__main__':
    # 启动Socket服务端（会自动启动检测器）
    server = VisionAPISocketServer()
    server.start()