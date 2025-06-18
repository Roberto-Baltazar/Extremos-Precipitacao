import numpy as np
import matplotlib.pyplot as plt
import shapefile
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import pandas as pd
import os

#%%

# Este script utiliza valores de precipitação do ERA5 para gerar mapas de precipitação acumulada anual na cidade do Rio de Janeiro. 

#%%


nc_path = 'C:/caminho do seu arquivo nc/PREC_ERA5.nc'  
shapefile_path = 'C:/caminho do seu shape da cidade do RJ/shape_rj.shp'
output = 'C:/caminho de onde voce quer salvar as figuras/figuras_nc/anuais'
os.makedirs(output, exist_ok=True)

# shape
def read_shapefile(shapefile_path):
    sf = shapefile.Reader(shapefile_path)
    shapes = sf.shapes()
    patches = []
    for shape in shapes:
        points = shape.points
        parts = list(shape.parts) + [len(points)]
        for i in range(len(parts) - 1):
            polygon = Polygon(points[parts[i]:parts[i+1]], closed=True)
            patches.append(polygon)
    return patches, sf.bbox


patches, bbox = read_shapefile(shapefile_path)
xmin, ymin, xmax, ymax = bbox
aspect_ratio = (ymax - ymin) / (xmax - xmin)
fig_width = 8
fig_height = fig_width * aspect_ratio

# dataset
ds = xr.open_dataset(nc_path)
ds = ds.rename({'valid_time': 'time'}) if 'valid_time' in ds else ds
precip = ds['tp'] * 30000
lat = ds['latitude']
lon = ds['longitude']
time = ds['time']


if not np.issubdtype(time.dtype, np.datetime64):
    time = pd.to_datetime(time.values)
    precip.coords['time'] = time

precip.coords['year'] = ('time', time.dt.year.data)

# paleta e escala
vmin = 0
vmax = 2000
levels = np.arange(vmin, vmax + 100, 100)
cmap = plt.get_cmap('jet_r', len(levels) - 1)
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)


anos = np.unique(time.dt.year)

#%% acumulados anuais

for ano in anos:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f'Precipitação Acumulada Anual - {ano}', fontsize=16)

    dados = precip.where(precip.year == ano, drop=True)

    if dados.time.size == 0:
        ax.set_title(f"{ano} - Sem dados")
        ax.axis('off')
        plt.close()
        continue

    dados_anual = dados.sum(dim='time')

    mesh = ax.pcolormesh(lon, lat, dados_anual, cmap=cmap, norm=norm, shading='nearest', transform=ccrs.PlateCarree())

    for patch in patches:
        ax.add_patch(Polygon(patch.get_xy(), facecolor='none', edgecolor='black', linewidth=0.8, transform=ccrs.PlateCarree()))

    ax.set_extent([xmin, xmax, ymin, ymax])
    ax.set_aspect('auto')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label='Precipitação (mm)', extend='max', ticks=levels)

    fig.subplots_adjust(left=0.05, right=0.9, top=0.88, bottom=0.1)

    filename = os.path.join(output, f"precip_rj_anual_{ano}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
