
import downscale as down
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import mevpy as mev
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import gaussian_kde
import conusfun as cfun
from sklearn.model_selection import KFold
from skgarden import RandomForestQuantileRegressor
down.matplotlib_update_settings()
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib as mpl
import xarray as xr



#########################    MAIN    ###########################

# load data (eps/alp < 25 are removed here):
df = cfun.load_results_df(csvname='dfres_cluster_30580.csv')



# READ DATA OVER ENTIRE CONUS::
ncres = cfun.load_results_netcdf(ncname='ncres_cluster_30580.nc',
                                 elevname='elev.hdf5')

features = ['Cs', 'Ws', 'Ns', 'gam_s', 'melev', 'selev']
featurenames = [r'$C_L$', r'$W_L$', r'$N_L$',
                r'$\gamma_0$', r'$\mu_{e}$', r'$\sigma_{e}$']



normalize_features = False

# normalize features:
if normalize_features:
    for fi in features:
        df[fi] = (df[fi] - np.mean(df[fi])) / np.std(df[fi])

# tunable parameters:
Tr = 50 # must be one of the Tr for which I saved the gauges, for EV analysis
Fi = 1 - 1/Tr
min_samples_split = 10
n_estimators = 2000
max_features = None # default

##############################################################
##############################################################

####### TRAIN MODELS  ######

X = df[features]
X = np.array(X)

# def train_qrf():

# label = 'C' # quantity to predict


etaC_model = RandomForestQuantileRegressor(
         min_samples_split=min_samples_split,
         n_estimators=n_estimators, max_features=max_features)
etaW_model = RandomForestQuantileRegressor(
    min_samples_split=min_samples_split,
    n_estimators=n_estimators, max_features=max_features)
etaN_model = RandomForestQuantileRegressor(
    min_samples_split=min_samples_split,
    n_estimators=n_estimators, max_features=max_features)

etaC_model.fit(X, df['etaC'].values)
etaW_model.fit(X, df['etaW'].values)
etaN_model.fit(X, df['etaN'].values)





lon = ncres['lon'][:]
lat = ncres['lat'][:]
nlon = np.size(lon)
nlat = np.size(lat)

# results I need:
etaC_mat = np.nan*np.zeros((nlon, nlat))
etaN_mat = np.nan*np.zeros((nlon, nlat))
etaW_mat = np.nan*np.zeros((nlon, nlat))
mevD_mat = np.nan*np.zeros((nlon, nlat))
mevE_mat = np.nan*np.zeros((nlon, nlat))
etaQ_mat = np.nan*np.zeros((nlon, nlat))

nfeatures = np.size(features)

file_name_save = 'predicted_error_maps.nc'
load_already_saved = True

if not load_already_saved:
    for i in range(1, nlon-1):
        print(i)
        for j in range(1, nlat-1):

            Xpred = np.nan*np.ones(nfeatures)
            for k in range(nfeatures):
                Xpred[k] = ncres[features[k]][i,j]
            Xpred = np.reshape(Xpred, (1, nfeatures))
            etaC_mat[i, j] = etaC_model.predict(Xpred, quantile=50)
            etaW_mat[i, j] = etaW_model.predict(Xpred, quantile=50)
            etaN_mat[i, j] = etaN_model.predict(Xpred, quantile=50)
            # correct values and compute extremes:
            NYd = ncres['NYd'][i, j, :]
            CYd = ncres['CYd'][i, j, :]
            WYd = ncres['WYd'][i, j, :]
            NYe = NYd/(etaN_mat[i, j] + 1)
            CYe = CYd/(etaC_mat[i, j] + 1)
            WYe = WYd/(etaW_mat[i, j] + 1)
            mevD_mat[i, j] = down.mev_quant(Fi, 9.0*np.mean(CYd), NYd, CYd, WYd,
                                      thresh=cfun.pixelkwargs['thresh'])[0]
            mevE_mat[i, j] = down.mev_quant(Fi, 9.0*np.mean(CYe), NYe, CYe, WYe,
                                         thresh=cfun.pixelkwargs['thresh'])[0]
            etaQ_mat[i, j] = (mevD_mat[i, j] - mevE_mat[i, j])/ mevE_mat[i, j]
    # write the results to output::
    dset = xr.Dataset({'etaC':(['x', 'y'], etaC_mat),
                       'etaW':(['x', 'y'], etaW_mat),
                       'etaN':(['x', 'y'], etaN_mat),
                       'etaQ':(['x', 'y'], etaQ_mat)},
                       coords={'lon': (['x'], lon),
                               'lat': (['y'], lat)})
    dset.to_netcdf(os.path.join(cfun.outdir_data, file_name_save), mode='w')


dset2 = xr.open_dataset(os.path.join(cfun.outdir_data, file_name_save))

# dset2['etaC'].values
# correct yearly parameter values:

# the filter out values below water and with esaS < 25 Km


def worldmap(grid_data=None, boxlat=None, boxlon=None, label=None,
             scatter=False, ax=None, df=None, dfkey=None,
             vmin=0, vmax=1000, nb=50.0, sb=22.0,
             wb=-130.0, eb=-60.0, maskocean = True, colorbar=True,
             return_cs = False, logscale = False, quantilerror = False):
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
        if not logscale:
            cs = m.pcolormesh(x, y, newdata, cmap = 'jet', vmin=vmin, vmax=vmax)
                              # norm=mpl.colors.LogNorm() )
        else:
            cs = m.pcolormesh(x, y, newdata, cmap = 'jet', vmin=vmin, vmax=vmax,
            norm=mpl.colors.LogNorm() )
    if not scatter:
        if colorbar:
            if not logscale:
                if not quantilerror:
                    cbar = m.colorbar(cs,location='bottom',pad="8%")
                                  # norm=mpl.colors.LogNorm() )
                else:
                    cbar = m.colorbar(cs,location='bottom',pad="8%", ticks=[-0.5, 0, 0.5, 1])
                    cbar.set_ticklabels([r'$< -0.5$', '0', '0.5', r'$> 1$'])  # vertically oriented colorbar
            else:

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



# datamat1 = etaC_mat
# datamat2 = etaN_mat
# datamat3 = etaW_mat
# datamat4 = etaQ_mat
PLOTVARS = [r'$\eta_C$', r'$\eta_W$', r'$\eta_N$', r'$\eta_Q$']
datamat1 = dset2['etaC'].values
datamat2 = dset2['etaW'].values
datamat3 = dset2['etaN'].values
datamat4 = dset2['etaQ'].values

# mask sea and esa_s < 25
melev = ncres['melev'][:]
esa_s = ncres['esa_s'][:]
mymask = np.logical_not(np.logical_and(melev > 0, esa_s > 25))

datamat1[mymask] = np.nan
datamat2[mymask] = np.nan
datamat3[mymask] = np.nan
datamat4[mymask] = np.nan

nb = 50
sb = 22
wb = -130
eb = -60
maskocean = False

fig, axes = plt.subplots(2, 2, figsize=(12,9))
axes[0, 0].text(0.05, 0.05, r'(a)',
             transform=axes[0, 0].transAxes, size=20)
m1, cs1 = worldmap(grid_data=datamat1,
                  boxlat=lat,
                  ax=axes[0, 0],
                  boxlon=lon,
                  label=PLOTVARS[0],
                  nb=nb, sb=sb, wb=wb, eb=eb,
                  vmin=np.nanmin(datamat1), vmax = np.nanmax(datamat1),
                  maskocean = maskocean, colorbar=True, return_cs=True)
axes[0,1].text(0.05, 0.05, r'(b)',
             transform=axes[0, 1].transAxes, size=20)
m2, cs2 = worldmap(grid_data=datamat2,
              boxlat=lat,
              ax=axes[0, 1],
              boxlon=lon,
              label=PLOTVARS[1],
              nb = nb, sb = sb, wb = wb, eb = eb,
              # vmin=np.nanmin(datamat2), vmax=np.nanmax(datamat2),
                   vmin=np.nanmin(datamat2), vmax=np.nanmax(datamat2),
              maskocean = maskocean, colorbar=True, return_cs = True)

axes[1,0].text(0.05, 0.05, r'(c)',
               transform=axes[1, 0].transAxes, size=20)
m3, cs3 = worldmap(grid_data=datamat3,
                  boxlat=lat,
                  ax=axes[1, 0],
                  boxlon=lon,
                  label=PLOTVARS[2],
                  nb=nb, sb=sb, wb=wb, eb=eb,
                  vmin=np.nanmin(datamat3), vmax = np.nanmax(datamat3),
                  maskocean = maskocean, colorbar=True, return_cs=True)
axes[1,1].text(0.05, 0.05, '(d)',
             transform=axes[1, 1].transAxes, size=20)
m4, cs4 = worldmap(grid_data=datamat4,
              boxlat=lat,
              ax=axes[1, 1],
              boxlon=lon,
              label=PLOTVARS[3],
              nb = nb, sb = sb, wb = wb, eb = eb,
              vmin=np.nanmin(-0.5), vmax=np.nanmax(1.0),
              maskocean = maskocean, colorbar=True, return_cs = True,
              logscale = False, quantilerror=True)
              # Add colorbar, make sure to specify tick locations to match desired ticklabels

# plot common colorbar::
# cbar = m2.colorbar(cs, location='bottom', pad="8%",
#                    norm=mpl.colors.LogNorm())
# cbar.set_label('Annual maximum daily rainfall [mm]')
plt.tight_layout()
plt.savefig(os.path.join(cfun.outplot,
                         'predicted_error_maps.png'), dpi=600)
plt.show()
