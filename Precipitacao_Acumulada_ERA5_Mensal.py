import numpy as np
import matplotlib.pyplot as plt
import shapefile  # pyshp
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import pandas as pd
import os

#%%

# Este script utiliza valores de precipitação do ERA5 para gerar mapas de precipitação acumulada mensal na cidade do Rio de Janeiro. 

#%%

nc_path = 'C:/caminho do seu arquivo nc/PREC_ERA5.nc'  
shapefile_path = 'C:/caminho do seu shape da cidade do RJ/shape_rj.shp'
output = 'C:/caminho de onde voce quer salvar as figuras/figuras_nc/mensais'

os.makedirs(output_dir, exist_ok=True)

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
fig_width = 22
fig_height = fig_width * aspect_ratio

# dataset
ds = xr.open_dataset(nc_path)
print(ds)


ds = ds.rename({'valid_time': 'time'})
precip = ds['tp'] * 30000 # passando para a escala acumulado mensal
lat = ds['latitude']
lon = ds['longitude']
time = ds['time']

if not np.issubdtype(time.dtype, np.datetime64):
    time = pd.to_datetime(time.values)
    precip.coords['time'] = time


precip.coords['year'] = ('time', time.dt.year.values)
precip.coords['month'] = ('time', time.dt.month.values)

# paleta e escala
vmin = 0
vmax = 400
levels = np.arange(vmin, vmax + 20, 20)
cmap = plt.get_cmap('jet_r', len(levels) - 1)
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

nomes_meses = [
    "Janeiro", "Fevereiro", "Março", "Abril",
    "Maio", "Junho", "Julho", "Agosto",
    "Setembro", "Outubro", "Novembro", "Dezembro"
]

#%% acumulados mensais

anos = np.unique(time.dt.year)

for ano in anos:
    fig, axes = plt.subplots(3, 4, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})
    #fig.suptitle(f'Precipitação Acumulada Mensal - {ano}', fontsize=20)
    mesh = None

    for mes in range(1, 13):
        ax = axes[(mes - 1) // 4][(mes - 1) % 4]

        dados = precip.where((precip.year == ano) & (precip.month == mes), drop=True)

        if dados.time.size == 0:
            ax.set_title(f"{nomes_meses[mes - 1]} - Sem dados")
            ax.axis('off')
            continue

        dados_mes = dados.mean(dim='time')

        mesh = ax.pcolormesh(lon, lat, dados_mes, cmap=cmap, norm=norm, shading='nearest', transform=ccrs.PlateCarree())

        for patch in patches:
            ax.add_patch(Polygon(patch.get_xy(), facecolor='none', edgecolor='black', linewidth=0.8, transform=ccrs.PlateCarree()))

        ax.set_title(nomes_meses[mes - 1], fontsize=10)
        ax.set_extent([xmin, xmax, ymin, ymax])
        ax.set_aspect('auto')

        # grade
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = plt.FixedLocator(np.round(np.linspace(xmin, xmax, 5), 2))
        gl.ylocator = plt.FixedLocator(np.round(np.linspace(ymin, ymax, 5), 2))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    # barra de cores
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label='Precipitação (mm)', extend='max', ticks=levels)

    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, hspace=0.3, wspace=0.2)
    
    filename = os.path.join(output, f"precip_rj_{str(ano)}.png")
    plt.savefig(filename, dpi=300)

    plt.close()
