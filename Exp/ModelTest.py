import torch
import numpy as np
from MultiTaskGan import Generator
from draw import draw_pics


# 模型加载
model = Generator()
# model.load_state_dict(torch.load('../save/unet/'))
model.eval()

# 数据读取
gfs_u10_input = np.load('../Dataset/u10_gfs.npy')[9500:, :96, :96]
gfs_v10_input = np.load('../Dataset/v10_gfs.npy')[9500:, :96, :96]
era5_u10_label = np.load('../Dataset/u10_era5.npy')[9500:, :96, :96]
era5_v10_label = np.load('../Dataset/u10_era5.npy')[9500:, :96, :96]
gfs_u10_input = np.expand_dims(gfs_u10_input, axis=1)
gfs_v10_input = np.expand_dims(gfs_v10_input, axis=1)
era5_u10_label = np.expand_dims(era5_u10_label, axis=1)
era5_v10_label = np.expand_dims(era5_v10_label, axis=1)

gfs_input = np.concatenate((gfs_u10_input, gfs_v10_input), axis=1)
era5_label = np.concatenate((era5_u10_label, era5_v10_label), axis=1)

# 计算初试RMSE
u10_rmse_origin = np.sqrt(np.sum((era5_u10_label-gfs_u10_input)**2)/np.sum(gfs_u10_input.shape))
print('u10的初始RMSE是：', u10_rmse_origin)
v10_rmse_origin = np.sqrt(np.sum((era5_v10_label-gfs_v10_input)**2)/np.sum(gfs_v10_input.shape))
print('v10的初始RMSE是：', u10_rmse_origin)
# 生成图片便于观察
# output = np.concatenate((gfs_v10_input, era5_v10_label), axis=1)
# for i in range(10):
#     draw_pics(torch.tensor(output[i*6]), i)
# 生成偏差订正数据
predict_result = model(gfs_input)
u10_rmse_predict = np.sqrt(np.sum((predict_result[:, 0]-gfs_u10_input)**2)/np.sum(gfs_u10_input.shape))
print('u10的订正RMSE是：', u10_rmse_origin)
v10_rmse_predict = np.sqrt(np.sum((predict_result[:, 1]-gfs_v10_input)**2)/np.sum(gfs_v10_input.shape))
print('v10的订正RMSE是：', u10_rmse_origin)
# 生成订正后的图，
output = np.concatenate((predict_result[:, 1], era5_v10_label), axis=1)
for i in range(10):
    draw_pics(torch.tensor(output[i*6]), i)
