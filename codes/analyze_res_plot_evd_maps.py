# plotting maps of extreme value estimates for a given quantile

# read hdf5 file with results of extreme value analysis
# and load TRMM land-sea mask
# and plot results over the desired domain


import os
import numpy as np
import h5py
# import pandas as pd
import conusfun as cfun
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# from datetime import datetime
import xarray as xr
# import dask.array as da
import matplotlib as mpl



import downscale as down
down.matplotlib_update_settings()

# note: not for codes that go in the cluster
from mpl_toolkits.basemap import Basemap, maskoceans


# relevant quantities:
# thresh = 1
# maxmiss = 36
# Tr = 50
# domain = 'conus' # conus or world
domain = 'world' # conus or world
# outname = "evd_conus_map_{}.hdf5".format(domain)
outname = 'tmpa_mev_global_quants.hdf5'
land_sea_mask = os.path.join(cfun.tmpa_dir, 'TRMM_TMPA_LandSeaMask.2.nc4')
mytr = 50 # return time for plotting

# plotting domain - used only if smaller than datset domain::
# plot_nb = 50.0
# plot_sb = -50.0
# plot_wb = -180.0
# plot_eb = 180.0

# if domain == 'conus':
#     nb = 50.0
#     sb = 22.0
#     eb = -60.0
#     wb = -130.0
#     vmin = 0
#     vmax = 1000
#     tmpa_hdf_file = os.path.join(cfun.tmpa_dir, 'data_tmpa_daily.hdf5')
# elif domain == 'world':
nb = 50.0
sb = -50.0
eb = 180.0
wb = -180.0
vmin = 1
vmax = 1200
    # tmpa_hdf_file = os.path.join(cfun.tmpa_dir, 'data_tmpa_world_daily.hdf5')
# else:
#     print('main_evd_maps ERROR:: must specify a valid domain!')

# get the smallest domain::
# nb = min(nb, plot_nb)
# sb = max(sb, plot_sb)
# wb = min(wb, plot_wb)
# eb = max(eb, plot_eb)

### reading land sea mask: ###
fmask = Dataset(land_sea_mask, 'r')
print(fmask.dimensions.keys())
print(fmask.variables.keys())
smlat = fmask.variables['lat'][:]
smlon = fmask.variables['lon'][:]
nmlon = np.size(smlon)
nmlat = np.size(smlat)
lsmask = fmask.variables['landseamask'][:]
print(np.shape(lsmask))
fmask.close()


plt.figure()
plt.imshow(lsmask)
plt.colorbar()
plt.show()

# restrict mask to trmm domain, and create land boolen values:
# Define land for pixel where at least 50% of the surface is land
land = lsmask < 50
xmask = xr.DataArray(land, coords=[smlon, smlat], dims=['lon', 'lat'])
trmm_cond = np.logical_and(
    np.logical_and(xmask.lon > wb, xmask.lon < eb),
    np.logical_and(xmask.lat > sb, xmask.lat < nb))
trmm_mask = xmask.where(trmm_cond, drop=True)
trmm_bool = trmm_mask.values.astype(bool)


# reading data structure with extreme value quantiles
with h5py.File(os.path.join(cfun.outdir_data, outname), "r") as fr:
    print(list(fr.keys()))
    qmev = fr['qmev'][:]
    qgev = fr['qgev'][:]
    Tr = fr['Tr'][()]
    thresh = fr['thresh'][()]
    tmpalat = fr['lat'][:]
    tmpalon = fr['lon'][:]

# to apply the mask directly on the dataset - I do in matplotlib instead.
# qmev[np.logical_not(trmm_bool)] = np.nan
# qgev[np.logical_not(trmm_bool)] = np.nan

xmev0 = xr.DataArray(qmev,
         coords=[tmpalon, tmpalat, Tr], dims=['lon', 'lat', 'Tr'])
xgev0 = xr.DataArray(qgev,
         coords=[tmpalon, tmpalat, Tr], dims=['lon', 'lat', 'Tr'])

xdiff0 = xmev0 - xgev0

# select return time of interest:
xmev = xmev0.sel(Tr=mytr)
xgev = xgev0.sel(Tr=mytr)
xdiff = xdiff0.sel(Tr=mytr)


dom_xmev = xmev
dom_xgev = xgev



print('mev max val = {}'.format(np.nanmax(xmev.values)))
print('mev min val = {}'.format(np.nanmin(xmev.values)))
print('gev max val = {}'.format(np.nanmax(xgev.values)))
print('gev min val = {}'.format(np.nanmin(xgev.values)))


def worldmap(grid_data=None, boxlat=None, boxlon=None, label=None,
             scatter=False, ax=None, df=None, dfkey=None,
             vmin=0, vmax=1000, nb=50.0, sb=22.0,
             wb=-130.0, eb=-60.0, maskocean = True, colorbar=True,
             return_cs = False):
    m = Basemap(projection='merc',
                resolution='l',
                # get these from conusdata
                llcrnrlon = wb,   # lower right cornet lat
                llcrnrlat = sb,
                urcrnrlon = eb,  # upper right corner
                urcrnrlat = nb,
                ax=ax)  # latitude of the true scale
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    parallels = np.arange(-50.,50,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    meridians = np.arange(-180.,180.,20.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    maplat, maplon = np.meshgrid(boxlat, boxlon)
    x, y = m(maplon, maplat) # compute map proj coordinates.
    np.shape(y)
    if maskocean:
        newdata = maskoceans(maplon, maplat, grid_data)
    else:
        newdata = grid_data
    cs = m.pcolormesh(x, y, newdata, cmap = 'jet', vmin=vmin, vmax=vmax,
                      norm=mpl.colors.LogNorm() )
    if not scatter:
        if colorbar:
            cbar = m.colorbar(cs,location='bottom',pad="8%",
                          norm=mpl.colors.LogNorm() )
            cbar.set_label(label)
    if scatter:
        sc = m.scatter(df['LON'].values, df['LAT'].values,
                               marker='o', c= df[dfkey].values, s=5,
                       zorder=10 , latlon=True, cmap='jet',
                       vmin=vmin, vmax=vmax)
        if colorbar:
            cbar2 = m.colorbar(sc,location='bottom',pad="8%")
            cbar2.set_label(label)
    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='w', lakes=True)
    if return_cs:
        return m, cs
    else:
        return m


fig, axes = plt.subplots(2, 1, figsize=(12,10))
maskocean = False
axes[0].text(0.05, 0.05, '(a) GEV Tr = {}'.format(mytr),
             transform=axes[0].transAxes, size=20)
m1, cs = worldmap(grid_data=xgev.values,
              boxlat=xgev.coords['lat'],
              ax=axes[0],
              boxlon=xgev.coords['lon'],
              # label='Daily rainfall [mm]',
              nb=nb, sb=sb, wb=wb, eb=eb, vmin=vmin, vmax = vmax,
              maskocean = maskocean, colorbar=False, return_cs=True)
axes[1].text(0.05, 0.05, '(b) MEV Tr = {}'.format(mytr),
             transform=axes[1].transAxes, size=20)
m2 = worldmap(grid_data=xmev.values,
              boxlat=xmev.coords['lat'],
              ax=axes[1],
              boxlon=xmev.coords['lon'],
              # label='Daily rainfall [mm]',
              nb = nb, sb = sb, wb = wb, eb = eb, vmin=vmin, vmax = vmax,
              maskocean = maskocean, colorbar=False)
# plot common colorbar::
cbar = m2.colorbar(cs, location='bottom', pad="8%",
                   norm=mpl.colors.LogNorm())
cbar.set_label('Annual maximum daily rainfall [mm]')
plt.tight_layout()
plt.savefig(os.path.join(cfun.outplot,
         '{}_domain_maps_evd.png'.format(domain)), dpi=600)
plt.show()

# fig, axes = plt.subplots(3, 1, figsize=(18,10))
# maskocean = False
# axes[0].text(0.05, 0.05, '(a) GEV Tr = {}'.format(mytr),
#              transform=axes[0].transAxes, size=20)
# m1, cs = worldmap(grid_data=xgev.values,
#               boxlat=xgev.coords['lat'],
#               ax=axes[0],
#               boxlon=xgev.coords['lon'],
#               # label='Daily rainfall [mm]',
#               nb=nb, sb=sb, wb=wb, eb=eb, vmin=vmin, vmax = vmax,
#               maskocean = maskocean, colorbar=False, return_cs=True)
# axes[1].text(0.05, 0.05, '(b) MEV Tr = {}'.format(mytr),
#              transform=axes[1].transAxes, size=20)
# m2 = worldmap(grid_data=xmev.values,
#               boxlat=xmev.coords['lat'],
#               ax=axes[1],
#               boxlon=xmev.coords['lon'],
#               # label='Daily rainfall [mm]',
#               nb = nb, sb = sb, wb = wb, eb = eb, vmin=vmin, vmax = vmax,
#               maskocean = maskocean, colorbar=False)
# vmind = -200
# vmaxd = +200
# axes[2].text(0.05, 0.05, '(b) MEV - GEV Tr = {}'.format(mytr),
#              transform=axes[2].transAxes, size=20)
# m3 = worldmap(grid_data=xdiff.values,
#               boxlat=xdiff.coords['lat'],
#               ax=axes[2],
#               boxlon=xdiff.coords['lon'],
#               # label='Daily rainfall [mm]',
#               nb = nb, sb = sb, wb = wb, eb = eb, vmin=vmind, vmax = vmaxd,
#               maskocean = maskocean, colorbar=False)
# # plot common colorbar::
# cbar = m2.colorbar(cs, location='bottom', pad="8%",
#                    norm=mpl.colors.LogNorm())
# cbar.set_label('Annual maximum daily rainfall [mm]')
# plt.tight_layout()
# plt.savefig(os.path.join(cfun.outplot,
#          '{}_domain_maps_diff.png'.format(domain)), dpi=600)
# plt.show()


fig, axes = plt.subplots(2, 1, figsize=(12,10))
maskocean = True
axes[0].text(0.05, 0.05, '(a) GEV Tr = {}'.format(mytr),
             transform=axes[0].transAxes, size=20)
m1 = worldmap(grid_data=xgev.values,
              boxlat=xgev.coords['lat'],
              ax=axes[0],
              boxlon=xgev.coords['lon'],
              label='Daily rainfall [mm]',
              nb=nb, sb=sb, wb=wb, eb=eb, vmin=vmin, vmax = vmax,
              maskocean = maskocean, colorbar=True)
axes[1].text(0.05, 0.05, '(b) MEV Tr = {}'.format(mytr),
             transform=axes[1].transAxes, size=20)
m2 = worldmap(grid_data=xmev.values,
              boxlat=xmev.coords['lat'],
              ax=axes[1],
              boxlon=xmev.coords['lon'],
              label='Daily rainfall [mm]',
              nb = nb, sb = sb, wb = wb, eb = eb, vmin=vmin, vmax = vmax,
              maskocean = maskocean, colorbar=True)
plt.tight_layout()
plt.savefig(os.path.join(cfun.outplot,
                         '{}_domain_maps_evd_withoceans.png'.format(
                             domain)), dpi=600)
plt.show()

# zoom in in southeast asia:
if domain == 'world':
    seasnb = 35.0
    seassb = -10.0
    seaswb = 70.0
    seaseb = 130.0

    # seas_cond = np.logical_and(
    #     np.logical_and(xmask.lat > seassb, xmask.lat < seasnb),
    #     np.logical_and(xmask.lon > seaswb, xmask.lon < seaseb))
    #


    seas_cond = np.logical_and(
        np.logical_and(trmm_mask.lon > seaswb, trmm_mask.lon < seaseb),
        np.logical_and(trmm_mask.lat > seassb, trmm_mask.lat < seasnb))

    seas_xmev = xmev.where(seas_cond, drop=True)
    seas_xgev = xgev.where(seas_cond, drop=True)

    land_cond = np.logical_and(seas_cond, trmm_bool)
    # land_cond = np.logical_and(seas_cond, trmm_bool)

    seas_land_xmev = xmev.where(land_cond, drop=True)
    seas_land_xgev = xgev.where(land_cond, drop=True)

    np.nanmax(seas_xmev.values)
    np.nanmin(seas_xmev.values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    maskocean = False
    # axes[0].set_title('GEV Quantiles Tr = {}'.format(Tr))
    axes[0].text(0.05, 0.05, '(a) GEV Tr = {}'.format(mytr),
                 transform=axes[0].transAxes, size=22)
    m1 = worldmap(grid_data=seas_xgev.values,
                  boxlat=seas_xgev.coords['lat'],
                  ax=axes[0],
                  boxlon=seas_xgev.coords['lon'],
                  label='Daily rainfall [mm]',
                  nb=seasnb, sb=seassb, wb=seaswb,
                  eb=seaseb, vmin=vmin, vmax=vmax,
                  maskocean = maskocean,
                  colorbar=False,
                  return_cs=False)

    # axes[1].set_title('MEV Quantiles Tr = {}'.format(Tr))
    axes[1].text(0.05, 0.05, '(b) MEV Tr = {}'.format(mytr),
                 transform=axes[1].transAxes, size=22)
    m2, cs2 = worldmap(grid_data=seas_xmev.values,
                  boxlat=seas_xmev.coords['lat'],
                  ax=axes[1],
                  boxlon=seas_xmev.coords['lon'],
                  label='Daily rainfall [mm]',
                  nb=seasnb, sb=seassb, wb=seaswb,
                  eb=seaseb, vmin=vmin, vmax=vmax,
                  maskocean=maskocean,
                  colorbar=False,
                  return_cs=True)

    fig.subplots_adjust(bottom=0.15)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.03])
    fig.colorbar(cs, cax=cbar_ax, orientation='horizontal',
                 label = 'Annual maximum daily rainfall [mm]')

    # cbar = m2.colorbar(cs2, norm=mpl.colors.LogNorm(), cax=cbar_ax)
    # cbar.set_label('Annual maximum daily rainfall [mm]')

    plt.tight_layout()
    plt.savefig(os.path.join(cfun.outplot,
                             'seasia_zoomed_maps_evd.png'), dpi=300)
    plt.show()





