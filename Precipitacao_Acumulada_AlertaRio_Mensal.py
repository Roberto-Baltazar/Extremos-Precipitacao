import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapefile  # pyshp
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#%%
# Este script utiliza valores de acumulados mensais do Alerta Rio e os interpola espacialmente
# para o shape da cidade do Rio de Janeiro, utilizando a interpolação Inverse Distance Weighting (IDW).
# Os dados das estações precisam estar em um arquivo .csv organizado da seguinte forma:
# EXEMPLO:
#    | latitude | longitude | ano | mes | precipitacao |
#    | -22.9658 | -43.2783  | 2011|  1  |    132.2     |
#    | -22.9658 | -43.2783  | 2012|  1  |    197.2     |
# etc...
# Todos os dados de todos os meses, anos e estações precisam estar no mesmo arquivo csv.
# Certifique que o csv está usando ';' como separador. Se estiver usando ',' ou '/', corrija na parte "# ler csv".
# O output é uma única figura para cada ano contendo os valores acumulados mensais de cada mês (jan-dez).

#%%

csv_path = 'C:/caminho do seu arquivo csv/estacoes_precipitacao_csv.csv'  
shapefile_path = 'C:/caminho do seu shape da cidade do RJ/shape_rj.shp'
output1 = 'C:/caminho de onde voce quer salvar as figuras/figuras/mensais/'

resolucao = 0.01  # resolução do grid que deseja

# ler csv
df = pd.read_csv(csv_path, sep=';')

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

# criando o grid
lat_vals = np.round(np.arange(ymin, ymax + resolucao, resolucao), 2)
lon_vals = np.round(np.arange(xmin, xmax + resolucao, resolucao), 2)
LAT, LON = lat_vals, lon_vals

def Euclidean(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# interpolação IDW (Inverse Distance Weighting)
# https://towardsdatascience.com/3-best-methods-for-spatial-interpolation-912cab7aee47/

def IDW(data, LAT, LON, betta=2):
    array = np.empty((LAT.shape[0], LON.shape[0]))
    for i, lat in enumerate(LAT):
        for j, lon in enumerate(LON):
            weights = data.apply(lambda row: Euclidean(row.longitude, lon, row.latitude, lat)**(-betta), axis=1)
            z = sum(weights * data.precipitacao) / weights.sum()
            array[i, j] = z
    return array

anos = sorted(df['ano'].unique())

nomes_meses = [
    "Janeiro", "Fevereiro", "Março", "Abril",
    "Maio", "Junho", "Julho", "Agosto",
    "Setembro", "Outubro", "Novembro", "Dezembro"
]

#%% acumulados mensais

vmin = 0
vmax = 400
levels = np.arange(vmin, vmax + 20, 20)  # passos de mm
cmap = plt.get_cmap('jet_r', len(levels) - 1)
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

for ano in anos:
    
    fig, axes = plt.subplots(3, 4, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f'Precipitação Acumulada Mensal - {ano}', fontsize=20)

    mesh = None  
    
    for mes in range(1, 13):
        ax = axes[(mes - 1) // 4][(mes - 1) % 4]

        dados_mes = df[(df['ano'] == ano) & (df['mes'] == mes)]

        if dados_mes.empty:
            ax.set_title(f"{nomes_meses[mes - 1]} - Sem dados")
            ax.axis('off')
            continue

        x = dados_mes['longitude'].values
        y = dados_mes['latitude'].values
        z = dados_mes['precipitacao'].values

        dados = pd.DataFrame({
            'longitude': x,
            'latitude': y,
            'precipitacao': z
        })

        idw_result = IDW(dados, LAT, LON)

        lon_grid, lat_grid = np.meshgrid(LON, LAT)
        
        mesh = ax.pcolormesh(lon_grid, lat_grid, idw_result, cmap=cmap, norm=norm, shading='nearest', transform=ccrs.PlateCarree())

        for patch in patches:
            ax.add_patch(Polygon(patch.get_xy(), facecolor='none', edgecolor='black', linewidth=0.8, transform=ccrs.PlateCarree()))

        ax.scatter(x, y, c='red', edgecolor='black', s=25, transform=ccrs.PlateCarree())
        ax.set_title(nomes_meses[mes - 1], fontsize=10)
        ax.set_extent([xmin, xmax, ymin, ymax])
        ax.set_aspect('auto')

        # grade cartografica
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.round(np.linspace(xmin, xmax, 5), 2))
        gl.ylocator = mticker.FixedLocator(np.round(np.linspace(ymin, ymax, 5), 2))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    # barra de cores
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label='Precipitação (mm)', extend='max', ticks=levels)

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(f"{output1}/precip_rj_{ano}.png", dpi=300)
    plt.close()