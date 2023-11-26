from cartopy import crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np

def draw_pics(data, k):
    path = '../images/unet/'

    u10 = data[0].cpu().numpy()
    v10 = data[1].cpu().numpy()
    lon = np.arange(100, 124, 0.25)
    lat = np.arange(0, 24, 0.25)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5, color='k', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5, color='k', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    color_max = max(int(u10.max()), int(v10.max())) + 1
    color_min = min(int(u10.min()), int(v10.min()))
    n_gap = (color_max - color_min)/20
    cbar_kwargs = {'ticks': np.arange(color_min, color_max+n_gap, n_gap*2)}
    levels = np.arange(color_min, color_max+n_gap, n_gap)

    filled_a = ax1.contourf(lon, lat, u10, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
    plt.contourf(lon, lat, u10, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
    fig.colorbar(filled_a, ax=ax1, fraction=0.045)

    filled_b = ax2.contourf(lon, lat, v10, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
    plt.contourf(lon, lat, v10, levels=levels, cmap='Blues', cbar_kwargs=cbar_kwargs)
    fig.colorbar(filled_b, ax=ax2, fraction=0.045)

    plt.savefig(path + 'pic_{}.png'.format(k))