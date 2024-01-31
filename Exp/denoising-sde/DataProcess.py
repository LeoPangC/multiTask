import torch
import os
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# from cartopy import crs as ccrs
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt


class WindSet(Dataset):
    def __init__(self, gfs_path, era5_path):
        data = []
        gfs_data = np.load(gfs_path)
        era5_data = np.load(era5_path)
        assert gfs_data.shape == era5_data.shape, '数据数量不匹配'
        gfs_data = np.expand_dims(gfs_data, axis=1)
        era5_data = np.expand_dims(era5_data, axis=1)
        for i in range(gfs_data.shape[0]):
            data.append((gfs_data[i], era5_data[i]))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pic, label = self.data[index]
        # pic = transforms.ToTensor()(pic)
        # label = transforms.ToTensor()(label)
        return pic, label


class GetWindSet(Dataset):
    def __init__(self, gfs_data_1, era5_data_1, gfs_data_2, era5_data_2):
        data = []
        assert gfs_data_1.shape == era5_data_2.shape, '数据数量不匹配'
        gfs_data_1 = np.expand_dims(gfs_data_1, axis=1)
        era5_data_1 = np.expand_dims(era5_data_1, axis=1)
        gfs_data_2 = np.expand_dims(gfs_data_2, axis=1)
        era5_data_2 = np.expand_dims(era5_data_2, axis=1)
        gfs_data = np.concatenate((gfs_data_1, gfs_data_2), axis=1)
        era5_data = np.concatenate((era5_data_1, era5_data_2), axis=1)

        for i in range(gfs_data.shape[0]):
            data.append((gfs_data[i], era5_data[i]))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pic, label = self.data[index]
        # pic = transforms.ToTensor()(pic)
        # label = transforms.ToTensor()(label)
        return pic, label


# if __name__ == '__main__':
#     era5u_path = 'era5u_test.npy'
#     gfs_path = 'gfsu_test.npy'
#     windU = WindSet(gfs_path=gfs_path, era5_path=era5u_path)
#     dataloader = DataLoader(windU, batch_size=64, shuffle=False, drop_last=False)
#     for i, data in enumerate(dataloader):
#         gfs = data[0][0, 0].numpy()
#         era5 = data[1][0, 0].numpy()
#         lon = np.arange(100, 145, 0.25)
#         lat = np.arange(0, 45, 0.25)
#
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
#         gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5, color='k', alpha=0.5, linestyle='--')
#         gl.xlabels_top = False
#         gl.ylabels_right = False
#         gl.xformatter = LONGITUDE_FORMATTER
#         gl.yformatter = LATITUDE_FORMATTER
#         gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5, color='k', alpha=0.5, linestyle='--')
#         gl.xlabels_top = False
#         gl.ylabels_right = False
#         gl.xformatter = LONGITUDE_FORMATTER
#         gl.yformatter = LATITUDE_FORMATTER
#         color_max = max(int(gfs.max()), int(era5.max())) + 1
#         color_min = min(int(gfs.min()), int(era5.min()))
#         n_gap = (color_max - color_min)/20
#         cbar_kwargs = {'ticks': np.arange(color_min, color_max+n_gap, n_gap*2)}
#         levels = np.arange(color_min, color_max+n_gap, n_gap)
#
#         filled_a = ax1.contourf(lon, lat, gfs, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
#         plt.contourf(lon, lat, gfs, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
#         fig.colorbar(filled_a, ax=ax1, fraction=0.045)
#
#         filled_b = ax2.contourf(lon, lat, era5, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
#         plt.contourf(lon, lat, era5, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
#         fig.colorbar(filled_b, ax=ax2, fraction=0.045)
#
#         plt.savefig('images/{}.png'.format(i))