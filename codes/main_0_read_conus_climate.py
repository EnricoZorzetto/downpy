# load and plot Koppen Geiger climate data
import os
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo.gdalconst import *
import conuscodes as cc
import h5py

# Office_PC = False
# Cluster = True


plot_true = True


# if Office_PC:
# datadir = os.path.join('C:\\', 'Users', 'ez23', 'Desktop', 'data_world_climate')
listdir = os.listdir(cc.climate_data)
imgfile = os.path.join(cc.climate_data, 'USGS_NED_1_n35w119_IMG', 'USGS_NED_1_n35w119_IMG.img')
# outdir = os.path.join('..', 'output', 'hpd_correlation')
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap, cm, maskoceans

# elif Cluster:
#     datadir = os.path.join('..', 'data', 'data_world_climate')
#     outdir = os.path.join('..', 'output', 'pixel_stats')

raster_file = os.path.join(cc.climate_data, 'Peel_2007', 'Raster_files', 'world_koppen', 'w001001.adf')
raster_file2 = os.path.join(cc.climate_data, 'Peel_2007', 'Raster_files', 'world_koppen', 'w001001x.adf')
raster_file3 = os.path.join(cc.climate_data, 'Peel_2007', 'Raster_files', 'world_koppen', 'vat.adf')
#
# box_name = 'Conus'
# solat = 22    # south bound
# nolat = 50    # north
# welon = -130  # west
# ealon = - 60   # east
#

#Opening the raster file
dataset = gdal.Open(raster_file, GA_ReadOnly )
dataset2 = gdal.Open(raster_file2, GA_ReadOnly )
dataset3 = gdal.Open(raster_file3, GA_ReadOnly )

rastermat2 = dataset2.ReadAsArray()
rastermat3 = dataset3.ReadAsArray()


# plt.imshow(rastermat2)
print(dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize)
prj=dataset.GetProjection()
rastermat = dataset.ReadAsArray()
rlat = np.flipud(np.arange(-90, 90.001, 0.1))
rlon = np.arange(-180, 180.001, 0.1)
transform = dataset.GetGeoTransform()

xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]

# data = ds.ReadAsArray()
# gt = ds.GetGeoTransform()
# proj = ds.GetProjection()



# get the edge coordinates and add half the resolution
# to go to center coordinates
xmin = xOrigin + pixelWidth * 0.1
xmax = xOrigin + (pixelWidth * dataset.RasterXSize) - pixelWidth * 0.1
ymin = yOrigin + (pixelHeight * dataset.RasterYSize) + pixelHeight * 0.1
ymax = yOrigin - pixelHeight * 0.1


# create a grid of xy coordinates in the original projection
xy_source = np.mgrid[xmin:xmax+pixelWidth:pixelWidth, ymax+pixelHeight:ymin:pixelHeight]

# get only north america
rblat = np.logical_and(rlat >= cc.solat, rlat <= cc.nolat)
rblon = np.logical_and(rlon >= cc.welon, rlon <= cc.ealon)

roxlat = rlat[rblat]
roxlon = rlon[rblon]

rx = np.arange(np.size(rlon))[rblon]
ry = np.arange(np.size(rlat))[rblat]

nrlon = np.size(roxlon)
nrlat = np.size(roxlat)

conusr = rastermat[ry[0]:ry[-1]+1, rx[0]:rx[-1]+1]

# plt.imshow(conusr)
rdict = {'Sea': 255, 'Af':1, 'Am':2, 'Aw':3, 'BWh':4, 'BWk':5, 'BSh':6, 'BSk':7, 'Csa':8, 'Csb':9,
         'Csc':10, 'Cwa':11, 'Cwb':12, 'Cwc':13, 'Cfa':14, 'Cfb':15, 'Cfc':16, 'Dsa':17, 'Dsb':18,
         'Dsc':19, 'Dsd':20, 'Dwa':21, 'Dwb':22, 'Dwc':23, 'Dwd':24, 'Dfa':25, 'Dfb':26, 'Dfc':27,
         'Dfd':28, 'ET':29, 'EF':30}

# simplify raster
new_dict = {'BW':[4, 5],
         'BS':[6, 7],
         'Cs':[8,9,10],
         'Cf':[14, 15, 16],
         'Dfa':25,
         'Dfb':26,
         'Sea':-1,
         'Other':[1,2,3,11,12,13, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30 ]}

# simplify raster
sdict = {'BW':0,
         'BS':1,
         'Cs':2,
         'Cf':3,
         'Dfa':4,
         'Dfb':5,
         'Sea':-1,
         'Other':6}

def simplify_types(conusr):
    coarser = np.zeros((np.shape(conusr)[0], np.shape(conusr)[1]))
    for ii in range(np.shape(conusr)[1]): # lon
        for jj in range(np.shape(conusr)[0]): # lat
            if conusr[jj,ii] in [4, 5]:
                coarser[jj,ii] = 0
            elif conusr[jj, ii] in [6, 7]:
                coarser[jj,ii]=1
            elif conusr[jj,ii] in [8,9,10]:
                coarser[jj,ii] =2
            elif conusr[jj,ii] in [14, 15, 16]:
                coarser[jj,ii] = 3
            elif conusr[jj,ii] == 25:
                coarser[jj,ii] = 4
            elif conusr[jj,ii] == 26:
                coarser[jj,ii] = 5
            elif conusr[jj,ii] == 255:
                coarser[jj,ii] =-1
            else:
                coarser[jj,ii] = 6 # other
    return coarser

coarser = simplify_types(conusr)
plt.imshow(coarser)



statdir = os.path.join('C:\\', 'Users', 'ez23','Desktop','NOAA_Hourly_prcp_conus', 'pub', 'data', 'hpd', 'auto', 'v1', 'beta')
dfs = pd.read_csv( os.path.join(cc.list_gauges_dir, 'HOURLY_LOC_NYEARS.csv'), index_col = 0)

# lats = np.arange(-49.875, 49.876, 0.25) # south to North
# lons = np.arange(-179.875, 179.876, 0.25) # West to East
# lonmat, latmat = np.meshgrid(lons, lats)
# dx = 0.25
# npixels = 1
# nlats = np.size(lats)
# nlons = np.size(lons)

npixels = 1

# kg_us = kg[ (kg['Lat'] <nolat )& (kg['Lat'] > solat) & (kg['Lon'] < ealon) & (kg['Lon'] > welon)].copy()
# Cls = np.unique(kg_us['Cls'].values)
# kg_lats = np.unique(kg_us['Lat'].values)
# ncls = np.size(Cls)
# mask arrays for selected  bounding box
bblat = np.logical_and(cc.lats >= cc.solat, cc.lats <= cc.nolat)
bblon = np.logical_and(cc.lons >= cc.welon, cc.lons <= cc.ealon)


boxlat = cc.lats[bblat]
boxlon = cc.lons[bblon]

boxx = np.arange(cc.nlons)[bblon]
boxy = np.arange(cc.nlats)[bblat]

nblon = np.size(boxlon)
nblat = np.size(boxlat)

# def simple_climate(code):
#     if code in ['Dfa', 'Dfb']:
#         return code
#     elif code[:2] in ['BW', 'BS', 'Cs', 'Cf']:
#         return code[:2]
#     else:
#         return 'other'
#
# kg_us['Cl2']= kg_us['Cls'].apply(simple_climate)
# Cls2 = np.unique(kg_us['Cl2'])
# ncl2 = np.size(Cls2)
# values = Cls2
# numvalues = np.arange(np.size(values))
# dictvalues = dict(zip(values, numvalues))

# conus_kg = pd.DataFrame(index = boxlat, columns = boxlon)
conus_vals = np.zeros((nblon, nblat), dtype = int)
for ii in range( nblon):
    print('lon_ii', ii)
    center_lon = boxlon[ii]
    for jj in range(nblat):
        center_lat = boxlat[jj]
        # get minimum distance
        distx = (center_lon - roxlon)**2
        disty = (center_lat - roxlat)**2
        myindx = np.argmin(distx)
        myindy = np.argmin(disty)
        conus_vals[ii, jj] = coarser[myindy, myindx]
        # conus_kg.iloc[jj, ii] = kg_us['Cl2'].iloc[myind]
        # conus_vals[jj, ii] = dictvalues[kg_us['Cl2'].iloc[myind]]


dictvalues = sdict
numvalues =[0,1,2,3,4,5,6,-1]



with h5py.File(os.path.join(cc.outdir_data, "KoppGeiger.hdf5"), "w") as f:
    f.create_dataset("KoppGeiger", data = conus_vals, dtype='f')
    f.create_dataset("lat", data = boxlat, dtype='f')
    f.create_dataset("lon", data = boxlon, dtype='f')



if plot_true:


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='merc',
                resolution='l',
                llcrnrlon = -130,   # lower right cornet Lat
                llcrnrlat = 22,
                urcrnrlon = -60,  # upper right corner
                urcrnrlat = 52,
                lat_ts = 35)  # latitude of the true scale
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    parallels = np.arange(0.,90,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(180.,360.,10.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    ny = conus_vals.shape[0]; nx = conus_vals.shape[1]
     #lonsp, latsp = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    # maplon, maplat= np.meshgrid(boxlon, boxlat)
    maplat, maplon= np.meshgrid(boxlat, boxlon)
    x, y = m(maplon, maplat) # compute map proj coordinates.
    # raplon, raplat= np.meshgrid(roxlon, roxlat)
    # xrr, yrr = m(raplon, raplat) # compute map proj coordinates.
    # x, y = m(lonsp, latsp) # compute map proj coordinates.
    # draw filled contours.
    # clevs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # clevs = [0,  1]
    # mask_ocean =False
    # if mask_ocean == True: # do not plot the contour over water
    #     # newdata = maskoceans(lonsp, latsp, data)
    #     newdata = maskoceans(maplon, maplat, data)
    #     cs = m.contourf(x, y, newdata,cmap=cm.s3pcpn)
    # else: # plot everywhere
    # cs = m.contourf(x, y, data,cmap=cm.s3pcpn)
    # cs = m.contourf(x, y, conus_vals)
    cs = m.pcolormesh(x, y, conus_vals)
    # extent = [x[0], y[0], x[-1], y[-1]] # [left, right, bottom, top]
    # cs = m.imshow(conus_vals, extent =extent,  interpolation = None)
    # cs = m.imshow(conus_vals, interpolation = None)
    # colors = cmap('hsv', 8)
    colors = [ cs.cmap(cs.norm(dictvalues[key])) for key in dictvalues.keys()]
    patches = [ mpatches.Patch(color=colors[i], label=" {}".format(list(dictvalues.keys())[i]) )
                for i in range(len(numvalues)) ]

    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(0.80, 0.5), loc=2, borderaxespad=0. )
    # plt.grid(True)
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='w', lakes=True)

    sc = m.scatter(dfs['LON'].values,dfs['LAT'].values, marker='o', c= 'red', s=1,  zorder=10 , latlon=True)
    # add colorbar.
    # cbar = m.colorbar(cs, cmap = colors, location='bottom',pad="8%")
    # cbar.set_label('Gauge Coverage')
    plt.title('Koppen Geiger classification',fontsize=12)
    # plt.tight_layout()
    plt.savefig( os.path.join( cc.outdir_fig, 'Koppen_Geiger.png'), dpi = 100)
    plt.show()




# plt.plot()
# plt.imshow(conus_vals)
# plt.show()