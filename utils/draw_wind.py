from cartopy import crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import cmaps
import os

def wind_quiver(gfs_data, bc_data, era5_data, save_path):
    gfs_u10 = gfs_data[:, 0]
    gfs_v10 = gfs_data[:, 1]
    bc_u10 = bc_data[:, 0]
    bc_v10 = bc_data[:, 1]
    era5_u10 = era5_data[:, 0]
    era5_v10 = era5_data[:, 1]
    lon = np.arange(100, 124, 0.25)
    lat = np.arange(0, 24, 0.25)

    for k in range(bc_data.shape[0]):
        # fig, ax = plt.subplots()
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True, subplot_kw={'projection': ccrs.PlateCarree()})
        # fig.set_size_inches(20, 10)
        ax0.plot([1, 2, 3], [1, 2, 3])
        ax1.plot([1, 2, 3], [3, 2, 1])
        ax2.plot([1, 2, 3], [1, 1, 1])
        gl = ax0.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator([100, 110, 120, 130])
        gl.ylocator = mticker.FixedLocator([0, 10, 20, 30])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator([100, 110, 120, 124])
        gl.ylocator = mticker.FixedLocator([0, 10, 20, 24])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
        gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator([100, 110, 120, 124])
        gl.ylocator = mticker.FixedLocator([0, 10, 20, 24])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

        gfs_wind_speed = np.sqrt(gfs_u10 ** 2 + gfs_v10 ** 2)
        bc_wind_speed = np.sqrt(bc_u10 ** 2 + bc_v10 ** 2)
        era5_wind_speed = np.sqrt(era5_u10 ** 2 + era5_v10 ** 2)
        color_max = max(int(gfs_wind_speed.max()), int(bc_wind_speed.max()), int(era5_wind_speed.max())) + 1
        color_min = min(int(gfs_wind_speed.min()), int(bc_wind_speed.min()), int(era5_wind_speed.min())) - 2
        n_gap = (color_max - color_min) / 20
        levels = np.arange(color_min, color_max, n_gap)

        # Wind Speed
        # clevs = np.arange(0, 14.5, 0.4)
        # filled_a = ax0.contourf(lon, lat, gfs_wind_speed[k], levels, transform=ccrs.PlateCarree(), cmap=cmaps.GMT_polar)
        filled_a = ax0.contourf(lon, lat, gfs_wind_speed[k], levels, transform=ccrs.PlateCarree(), cmap='Spectral')
        filled_b = ax1.contourf(lon, lat, bc_wind_speed[k], levels, transform=ccrs.PlateCarree(), cmap='Spectral')
        filled_c = ax2.contourf(lon, lat, era5_wind_speed[k], levels, transform=ccrs.PlateCarree(), cmap='Spectral')
        ax0.set_title('GFS', size=16)
        ax1.set_title('Diffusion Result', size=16)
        ax2.set_title('ERA5', size=16)

        cb = fig.colorbar(filled_a, ax=ax0, orientation='horizontal', pad=0.05, aspect=15, drawedges=True, shrink=0.5)
        cb.set_label('m/s', size=10, rotation=0, labelpad=1.2, loc='center')
        cb.ax.tick_params(labelsize=8)
        cb = fig.colorbar(filled_b, ax=ax1, orientation='horizontal', pad=0.05, aspect=15, drawedges=True, shrink=0.5)
        cb.set_label('m/s', size=10, rotation=0, labelpad=1.2, loc='center')
        cb.ax.tick_params(labelsize=8)
        cb = fig.colorbar(filled_c, ax=ax2, orientation='horizontal', pad=0.05, aspect=15, drawedges=True, shrink=0.5)
        cb.set_label('m/s', size=10, rotation=0, labelpad=1.2, loc='center')
        cb.ax.tick_params(labelsize=8)

        # Wind vectors
        # ax0.quiver(lon, lat, gfs_u10[k], gfs_v10[k], color='k', linewidth=0.5)
        # ax1.quiver(lon, lat, bc_u10[k], bc_v10[k], color='k', linewidth=0.5)
        # ax2.quiver(lon, lat, era5_u10[k], era5_v10[k], color='k', linewidth=0.5)
        ax0.streamplot(lon, lat, gfs_u10[k], gfs_v10[k], color='k', linewidth=0.5)
        ax1.streamplot(lon, lat, bc_u10[k], bc_v10[k], color='k', linewidth=0.5)
        ax2.streamplot(lon, lat, era5_u10[k], era5_v10[k], color='k', linewidth=0.5)
        plt.savefig(os.path.join(save_path, 'pic_{}.png'.format(k)))


if __name__ == '__main__':
    bc = np.load(r'/home/hy4080/PycharmProjects/multiTask/Dataset/test/bc_32.npy')
    origin = np.load(r'/home/hy4080/PycharmProjects/multiTask/Dataset/test/origin_32.npy')
    label = np.load(r'/home/hy4080/PycharmProjects/multiTask/Dataset/test/label_32.npy')
    wind_quiver(origin, bc, label, save_path=r'/home/hy4080/PycharmProjects/multiTask/images/test')