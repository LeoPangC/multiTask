import torch
import numpy as np
from MultiTaskUnet import Generator
from draw import draw_pics


device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
# 模型加载
model = Generator()
model.load_state_dict(torch.load('../save/unet/unet_199.pth'))
model.to(device)
model.eval()
model.cuda()

# 数据读取
left = 9700
right = 9900
gfs_u10_input = np.load('../Dataset/u10_gfs.npy')[left:right, :96, :96]
gfs_v10_input = np.load('../Dataset/v10_gfs.npy')[left:right, :96, :96]
era5_u10_label = np.load('../Dataset/u10_era5.npy')[left:right, :96, :96]
era5_v10_label = np.load('../Dataset/u10_era5.npy')[left:right, :96, :96]
gfs_u10_input = np.expand_dims(gfs_u10_input, axis=1)
gfs_v10_input = np.expand_dims(gfs_v10_input, axis=1)
era5_u10_label = np.expand_dims(era5_u10_label, axis=1)
era5_v10_label = np.expand_dims(era5_v10_label, axis=1)

# 转换为速度角度
deg = 180.0 / np.pi
era5_speed = np.sqrt(era5_u10_label ** 2 + era5_v10_label ** 2)
gfs_speed = np.sqrt(gfs_u10_input ** 2 + gfs_v10_input ** 2)
era5_dir = 180.0 + np.arctan2(era5_u10_label, era5_v10_label) * deg
gfs_dir = 180.0 + np.arctan2(gfs_u10_input, gfs_v10_input) * deg
# 归一化
s_max = 64.817894
era5_speed = era5_speed / s_max
gfs_speed = gfs_speed / s_max
era5_dir = era5_dir / 360.0
gfs_dir = gfs_dir / 360.0

# 数据拼接
# gfs_input = np.concatenate((gfs_u10_input, gfs_v10_input), axis=1)
# era5_label = np.concatenate((era5_u10_label, era5_v10_label), axis=1)
gfs_input = np.concatenate((gfs_speed, gfs_dir), axis=1)
era5_label = np.concatenate((era5_speed, era5_dir), axis=1)

# 计算初始RMSE
b, _, h, w = gfs_u10_input.shape
u10_rmse_origin = np.sqrt(np.sum((era5_u10_label-gfs_u10_input)**2)/(b*h*w))
print('u10的初始RMSE是：', u10_rmse_origin)
v10_rmse_origin = np.sqrt(np.sum((era5_v10_label-gfs_v10_input)**2)/(b*h*w))
print('v10的初始RMSE是：', v10_rmse_origin)
# 生成图片便于观察
# output = np.concatenate((gfs_v10_input, era5_v10_label), axis=1)
# for i in range(10):
#     draw_pics(torch.tensor(output[i*6]), i)
# 生成偏差订正数据
with torch.no_grad():
    gfs_input = torch.tensor(gfs_input)
    gfs_input = gfs_input.to(device)
    predict_result = model(gfs_input)
    predict_result = predict_result.cpu().numpy()

# u10 = predict_result[:, :1]
# v10 = predict_result[:, 1:]
# 返归一化
predict_speed = predict_result[:, :1] * s_max
predict_dir = predict_result[:, 1:] * 360.0
rad = np.pi/180.0
u10 = -predict_speed * np.sin(predict_dir * rad)
v10 = -predict_speed * np.cos(predict_dir * rad)

u10_rmse_predict = np.sqrt(np.sum((u10-era5_u10_label)**2)/(b*h*w))
print('u10的订正RMSE是：', u10_rmse_predict)
v10_rmse_predict = np.sqrt(np.sum((v10-era5_v10_label)**2)/(b*h*w))
print('v10的订正RMSE是：', v10_rmse_predict)
print('u10提升度:', (u10_rmse_origin-u10_rmse_predict)/u10_rmse_origin)
print('v10提升度:', (v10_rmse_origin-v10_rmse_predict)/v10_rmse_origin)

# 生成订正后的图
# output = np.concatenate((predict_result[:, 0:1], era5_v10_label), axis=1)
# for i in range(10):
#     draw_pics(torch.tensor(output[i*6]), i)
