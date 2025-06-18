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
# Este script utiliza valores de acumulados mensais do Alerta Rio e os interpola espacialmente para gerar acumulados sazonais
# para o shape da cidade do Rio de Janeiro, utilizando a interpolação Inverse Distance Weighting (IDW).
# Os dados das estações precisam estar em um arquivo .csv organizado da seguinte forma:
# EXEMPLO:
#    | latitude | longitude | ano | mes | precipitacao |
#    | -22.9658 | -43.2783  | 2011|  1  |    132.2     |
#    | -22.9658 | -43.2783  | 2012|  1  |    197.2     |
# etc...
# Todos os dados de todos os meses, anos e estações precisam estar no mesmo arquivo csv.
# Certifique que o csv está usando ';' como separador. Se estiver usando ',' ou '/', corrija na parte "# ler csv".
# O output é uma única figura para cada ano contendo os valores acumulados sazonais de cada estação do ano (DJF, MAM, JJA, SON).
# O cálculo dos acumulados sazonais do Verão (DJF) utilizam o acumulado do dezembro do ano anterior no cálculo.
# Ex: Verão de 2011 = dezembro 2010 + janeiro 2011 + fevereiro 2011

#%%

csv_path = 'C:/caminho do seu arquivo csv/estacoes_precipitacao_csv.csv'  
shapefile_path = 'C:/caminho do seu shape da cidade do RJ/shape_rj.shp'
output_dir2 = 'C:/caminho de onde voce quer salvar as figuras/figuras/sazonais/'

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


#%% acumulados sazonais (DJF, MAM, JJA, SON)


patches, bbox = read_shapefile(shapefile_path)
xmin, ymin, xmax, ymax = bbox
aspect_ratio = (ymax - ymin) / (xmax - xmin)
fig_width = 16
fig_height = fig_width * aspect_ratio

# dicionário de estações com seus respectivos meses
sazonais = {
    'Verão': [12, 1, 2],
    'Outono': [3, 4, 5],
    'Inverno': [6, 7, 8],
    'Primavera': [9, 10, 11]
}

for ano in anos:
    
    vmin = 0
    vmax = 800
    levels = np.arange(vmin, vmax + 50, 50)  # passos de mm
    cmap = plt.get_cmap('jet_r', len(levels) - 1)
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f'Precipitação Acumulada Sazonal - {ano}', fontsize=20)

    for idx, (estacao, meses) in enumerate(sazonais.items()):
        ax = axes[idx // 2][idx % 2]

        if estacao == 'DJF':
            dados = df[((df['ano'] == ano) & (df['mes'].isin([1, 2]))) |
                       ((df['ano'] == ano - 1) & (df['mes'] == 12))].copy()
        else:
            dados = df[(df['ano'] == ano) & (df['mes'].isin(meses))].copy()

        if dados.empty:
            ax.set_title(f"{estacao} - Sem dados")
            ax.axis('off')
            continue

        # agrupa por estação
        dados_agg = dados.groupby(['latitude', 'longitude'])['precipitacao'].sum().reset_index()

        idw_result = IDW(dados_agg, LAT, LON)
        lon_grid, lat_grid = np.meshgrid(LON, LAT)

        mesh = ax.pcolormesh(lon_grid, lat_grid, idw_result, cmap=cmap, norm=norm, shading='nearest', transform=ccrs.PlateCarree())

        for patch in patches:
            ax.add_patch(Polygon(patch.get_xy(), facecolor='none', edgecolor='black', linewidth=0.8, transform=ccrs.PlateCarree()))

        ax.scatter(dados_agg['longitude'], dados_agg['latitude'], c='red', edgecolor='black', s=25, transform=ccrs.PlateCarree())
        ax.set_title(estacao, fontsize=12)
        ax.set_extent([xmin, xmax, ymin, ymax])
        ax.set_aspect('auto')

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        #gl.xlocator = mticker.FixedLocator(np.round(np.linspace(xmin, xmax, 5), 2))
        #gl.ylocator = mticker.FixedLocator(np.round(np.linspace(ymin, ymax, 5), 2))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label='Precipitação (mm)', extend='max', ticks=levels)
    
    fig.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05, hspace=0.3, wspace=0.2)

    #plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(f"{output2}/precip_sazonal_rj_{ano}.png", dpi=300)
    plt.close()