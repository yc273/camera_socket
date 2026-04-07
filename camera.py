import cv2
import numpy as np
from ctypes import *
from threading import Thread, Event
from MvCameraControl_class import *

class Camera:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cam = None
        self.running = Event()  # 控制视频流线程
        self.frame = None       # 存储最新帧

    def connect(self):
        """连接相机"""
        device_list = self.enum_devices()
        if not device_list:
            return False
        self.cam = self.create_camera_handle(device_list, self.device_index)
        return self.cam is not None

    def enum_devices(self):
        """枚举设备"""
        devicelist = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, devicelist)
        if ret != 0 or devicelist.nDeviceNum == 0:
            print("未发现设备！")
            return None
        return devicelist

    def create_camera_handle(self, device_list, index=0):
        """创建相机句柄（新增自动曝光设置）"""
        if index >= device_list.nDeviceNum:
            print("选择的设备序号超出范围！")
            return None
        cam = MvCamera()
        dev_info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            print("创建相机句柄失败！ret[0x%x]" % ret)
            return None
        
        # 1. 先打开设备（必须先打开才能设置参数）
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("打开设备失败！ret[0x%x]" % ret)
            cam.MV_CC_DestroyHandle()
            return None
        
        # 2. 新增：设置自动曝光（核心修改）
        # 先关闭触发模式（避免影响自动曝光）
        ret_trigger = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret_trigger != 0:
            print(f"关闭触发模式失败！ret[0x%x]" % ret_trigger)
        
        # 设置自动曝光模式：2=自动，1=手动，0=关闭
        ret_expo = cam.MV_CC_SetEnumValue("ExposureAuto", 2)
        if ret_expo == 0:
            print("自动曝光已成功开启！")
        else:
            print(f"开启自动曝光失败！ret[0x%x]" % ret_expo)
        ret_gain = cam.MV_CC_SetEnumValue("GainAuto", 2)
        if ret_gain == 0:
            print("自动增益已成功开启！")
        else:
            print(f"开启自动增益失败！ret[0x%x]" % ret_gain)
        # 可选：设置自动曝光范围（根据需要调整）
        # # 最小曝光时间（单位：μs）
        # ret_min = cam.MV_CC_SetIntValue("ExposureTimeMin", 1000)
        # # 最大曝光时间（单位：μs）
        # ret_max = cam.MV_CC_SetIntValue("ExposureTimeMax", 200000)
        # if ret_min == 0 and ret_max == 0:
        #     print("自动曝光范围已设置！")
        
        return cam
    def start_video_stream(self, callback=None):
        """启动视频流线程"""
        self.running.set()
        thread = Thread(target=self._video_loop, args=(callback,))
        thread.daemon = True
        thread.start()

    def stop_video_stream(self):
        """停止视频流"""
        self.running.clear()

    def _video_loop(self, callback):
        """视频流循环"""
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("获取 PayloadSize 失败！ret[0x%x]" % ret)
            return
        nPayloadSize = stParam.nCurValue
        data_buf = (c_ubyte * nPayloadSize)()
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("启动抓取失败！ret[0x%x]" % ret)
            return
        stDeviceList = MV_FRAME_OUT_INFO_EX()
        memset(byref(stDeviceList), 0, sizeof(stDeviceList))
        try:
            while self.running.is_set():
                ret = self.cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stDeviceList, 1000)
                if ret == 0:
                    width = stDeviceList.nWidth
                    height = stDeviceList.nHeight
                    pixel_type = stDeviceList.enPixelType
                    data_array = np.ctypeslib.as_array(data_buf)
                    
                    # 与main.py保持完全一致的处理方式
                    if pixel_type == PixelType_Gvsp_Mono8:
                        self.frame = data_array.reshape(height, width)
                    elif pixel_type == PixelType_Gvsp_BayerRG8:
                        self.frame = data_array.reshape(height, width)
                        # 使用与main.py相同的颜色转换
                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BAYER_RG2RGB)
                    else:
                        print("Unsupported pixel format: 0x%x" % pixel_type)
                        continue
                    
                    # 添加调试信息，便于排查问题
                    # print(f"Width: {width}, Height: {height}, Pixel Type: 0x{pixel_type:x}")
                    
                    if callback:
                        callback(self.frame)
        finally:
            self.cam.MV_CC_StopGrabbing()

    def save_frame(self, save_path="data/frame.jpg", quality=95):
        """保存当前帧
        
        Args:
            save_path (str): 保存路径
            quality (int): JPEG质量，范围0-100，默认95
                          40表示40%质量压缩
        """
        if self.frame is not None:
            # 设置JPEG压缩参数
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            cv2.imwrite(save_path, self.frame, encode_param)
            # print(f"图像已保存至: {save_path} (质量: {quality}%)")
            return True
        print("无可用帧！")
        return False

    def disconnect(self):
        """断开连接"""
        if self.cam:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None