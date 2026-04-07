import torch
# 核心验证：返回True即说明GPU加速可用（兼容成功）
print(torch.cuda.is_available())
# 查看PyTorch编译时使用的CUDA版本（会显示cu121或cu124，正常）
print(torch.version.cuda)
# 查看显卡信息，确认是否正确识别
print(torch.cuda.get_device_name(0))