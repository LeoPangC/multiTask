import torch
import numpy as np
from MultiTaskGan import Generator
from draw import draw_pics


device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
# 模型加载
model = Generator().to(device)
model.load_state_dict(torch.load('../save/unet/unet_199.pth'))
model.eval()
model.cuda()

# 数据读取
left = 2000
right = 2200
gfs_u10_input = np.load('../Dataset/u10_gfs.npy')[left:right, :96, :96]
gfs_v10_input = np.load('../Dataset/v10_gfs.npy')[left:right, :96, :96]
era5_u10_label = np.load('../Dataset/u10_era5.npy')[left:right, :96, :96]
era5_v10_label = np.load('../Dataset/u10_era5.npy')[left:right, :96, :96]
gfs_u10_input = np.expand_dims(gfs_u10_input, axis=1)
gfs_v10_input = np.expand_dims(gfs_v10_input, axis=1)
era5_u10_label = np.expand_dims(era5_u10_label, axis=1)
era5_v10_label = np.expand_dims(era5_v10_label, axis=1)

gfs_input = np.concatenate((gfs_u10_input, gfs_v10_input), axis=1)
era5_label = np.concatenate((era5_u10_label, era5_v10_label), axis=1)

# 计算初试RMSE
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
u10_rmse_predict = np.sqrt(np.sum((predict_result[:, :1]-era5_u10_label)**2)/(b*h*w))
print('u10的订正RMSE是：', u10_rmse_predict)
v10_rmse_predict = np.sqrt(np.sum((predict_result[:, 1:]-era5_v10_label)**2)/(b*h*w))
print('v10的订正RMSE是：', v10_rmse_predict)
print('u10:', (u10_rmse_origin-u10_rmse_predict)/u10_rmse_origin)
print('v10:', (v10_rmse_origin-v10_rmse_predict)/v10_rmse_origin)

# 生成订正后的图
# output = np.concatenate((predict_result[:, 0:1], era5_v10_label), axis=1)
# for i in range(10):
#     draw_pics(torch.tensor(output[i*6]), i)
