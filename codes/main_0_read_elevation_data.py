import os
import numpy as np
import h5py
from scipy.io import netcdf
import pandas as pd
import conusfun as cfun
from mpl_toolkits.basemap import Basemap, cm, maskoceans
import matplotlib.pyplot as plt


# Process gridded topographic dataset obtained at:
# https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/
# (using version with top of sea ice and grid-centered), and unzipped
# Cluster = True
# Office_PC = False
#


dfs = pd.read_csv( os.path.join(cfun.list_gauges_dir, 'HOURLY_LOC_NYEARS.csv'),
                   index_col = 0)
plot_true = True


myfile = os.path.join(cfun.elev_dir, 'etopo1.nc')
f = netcdf.netcdf_file(myfile, 'r')


print(f.dimensions.keys())
print(f.variables.keys())


rlat = f.variables['lat'][:]
rlon = f.variables['lon'][:]
nrlon = np.size(rlon)
nrlat = np.size(rlat)
crs = f.variables['crs']
Band1 = f.variables['Band1'][:]
print(np.shape(Band1))



npixels = 5 # box size to compute stdv

bblat = np.logical_and(cfun.lats >= cfun.solat, cfun.lats <= cfun.nolat)
bblon = np.logical_and(cfun.lons >= cfun.welon, cfun.lons <= cfun.ealon)

boxlat = cfun.lats[bblat]
boxlon = cfun.lons[bblon]

boxx = np.arange(cfun.nlons)[bblon]
boxy = np.arange(cfun.nlats)[bblat]

nblon = np.size(boxlon)
nblat = np.size(boxlat)

mean_el = np.zeros((nblon, nblat))
for ii in range( nblon):
    print('lon_ii', ii)
    center_lon = boxlon[ii]
    webound = center_lon - 1/2*cfun.dx
    eabound = center_lon + 1/2*cfun.dx
    for jj in range(nblat):
        center_lat = boxlat[jj]
        nobound = center_lat + 1/2*cfun.dx
        sobound = center_lat - 1/2*cfun.dx
        indlat = np.logical_and(rlat > sobound, rlat < nobound)
        indlon = np.logical_and(rlon > webound, rlon <eabound)
        mybox = Band1[ np.ix_(indlat, indlon)]
        mean_el[ii, jj] = np.mean(mybox)


stdv_el = np.zeros((nblon, nblat))
for ii in range( nblon):
    print('lon_ii', ii)
    center_lon = boxlon[ii]
    webound = center_lon - npixels/2*cfun.dx
    eabound = center_lon + npixels/2*cfun.dx
    for jj in range(nblat):
        center_lat = boxlat[jj]
        nobound = center_lat + npixels/2*cfun.dx
        sobound = center_lat - npixels/2*cfun.dx

        indlat = np.logical_and(rlat > sobound, rlat < nobound)
        indlon = np.logical_and(rlon > webound, rlon <eabound)
        mybox = Band1[ np.ix_(indlat, indlon)]
        stdv_el[ii, jj] = np.std(mybox)


with h5py.File(os.path.join(cfun.tmpa_dir, "elev.hdf5"), "w") as fo:
    fo.create_dataset("mean_el", data = mean_el, dtype='f')
    fo.create_dataset("stdv_el", data = stdv_el, dtype='f')
    fo.create_dataset("lat", data = boxlat, dtype='f')
    fo.create_dataset("lon", data = boxlon, dtype='f')
    fo.create_dataset("npix_stdv", data = npixels, dtype='int32')



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
    maprlon, maprlat= np.meshgrid(rlon, rlat)
    xr, yr = m(maprlon, maprlat) # compute map proj coordinates.
    cs = m.contourf(xr, yr, Band1)
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='w', lakes=True)
    sc = m.scatter(dfs['LON'].values,dfs['LAT'].values, marker='o',
                   c='red', s=1,  zorder=10 , latlon=True)
    cbar = m.colorbar(cs, location='bottom',pad="8%")
    cbar.set_label('Elevation [m MSL]')
    plt.title('Elevation',fontsize=12)
    plt.savefig(os.path.join(cfun.outplot, 'Original_elevation.png'), dpi=100)
    plt.show()



    data = mean_el

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
    maplat, maplon = np.meshgrid(boxlat, boxlon)
    x, y = m(maplon, maplat) # compute map proj coordinates.
    clevs = np.linspace(0, 4000, 50)
    mask_ocean = True
    if mask_ocean: # do not plot the contour over water
        newdata = maskoceans(maplon, maplat, data)
        cs = m.contourf(x, y, newdata,clevs, cmap= 'BrBG')
    else:
        cs = m.contourf(x, y, data, clevs, cmap = 'BrBG')
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='w', lakes=True)
    sc = m.scatter(dfs['LON'].values,dfs['LAT'].values, marker='o',
                   c='red', s=1,  zorder=10 , latlon=True)
    cbar = m.colorbar(cs, location='bottom',pad="8%")
    cbar.set_label('Elevation [m MSL]')
    plt.title('Mean elevation',fontsize=12)
    # fig.tight_layout()
    plt.savefig( os.path.join( cfun.outplot,
                               'Coarse_Elevation_mean.png'), dpi=100)
    plt.show()

    data = stdv_el

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
    # maplon, maplat= np.meshgrid(boxlon, boxlat) # if array nlat*nlon
    maplat, maplon = np.meshgrid(boxlat, boxlon)
    x, y = m(maplon, maplat) # compute map proj coordinates.
    clevs = np.linspace(0, 1200, 50)
    mask_ocean = True
    if mask_ocean: # do not plot the contour over water
        newdata = maskoceans(maplon, maplat, data)
        cs = m.contourf(x, y, newdata,clevs, cmap= 'BrBG')
    else:
        cs = m.contourf(x, y, data, clevs, cmap = 'BrBG')
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='w', lakes=True)
    sc = m.scatter(dfs['LON'].values,dfs['LAT'].values, marker='o',
                   c='red', s=1,  zorder=10 , latlon=True)
    cbar = m.colorbar(cs, location='bottom',pad="8%")
    cbar.set_label(r'$\sigma$ elevation [m MSL]')
    plt.title(r'$\sigma$ elevation',fontsize=12)
    # fig.tight_layout()
    plt.savefig( os.path.join( cfun.outplot,
           'Coarse_Elevation_stdv_{}_pixels.png'.format(npixels)), dpi=100)
    plt.show()
