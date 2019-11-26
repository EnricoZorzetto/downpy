
import downscale as down
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import mevpy as mev
import conusfun as cfun
down.matplotlib_update_settings()
import seaborn as sns



#########################    MAIN    ###########################


# load data from cluster complete analysis:: (eps/alp < 25 are removed here)
df = cfun.load_results_df(csvname='dfres_cluster_30580.csv')
df.to_csv( os.path.join(cfun.outdir_data, 'dfres_complete_only.csv'))
ncres = cfun.load_results_netcdf(ncname='ncres_cluster_30580.nc')

# df = pd.read_csv(os.path.join(cfun.outdir_data,
#                               'dfres_laptop_30580.csv'), index_col=0)
#
#
# dfg = pd.read_csv(os.path.join(cfun.outdir_data,
#                               'dfres_gauges_1451.csv'), index_col=0)
#
# ncres = xr.open_dataset(os.path.join(cfun.outdir_data,
#                                      'ncres_laptop_30580.nc'))

list(ncres.variables)
lats = ncres.coords['lat'].values
lons = ncres.coords['lon'].values

COMPP = ncres['complete_pixel'].values
LESS2 = ncres['gauge_nmin_less_than_2'].values
TLESS2 = ncres['tmpa_nmin_less_than_2'].values

MEVG = ncres['mev_g']
MEVS = ncres['mev_s']
MEVD = ncres['mev_d']

MEVGvals = MEVG.values
MEVGcomp = MEVGvals[COMPP]
MEVGless2 = MEVGvals[LESS2]

MEVSvals = MEVS.values
MEVScomp = MEVSvals[COMPP]
MEVSless2 = MEVSvals[LESS2]

MEVDvals = MEVD.values
MEVDcomp = MEVDvals[COMPP]
MEVDless2 = MEVDvals[LESS2]


Tfull = np.logical_and(COMPP, TLESS2)
Gfull = np.logical_and(COMPP, LESS2)
print(np.size(MEVDvals[Tfull]))
print(np.size(MEVGvals[Gfull]))
# print(np.size(MEVGvals[Gfull]))

print(np.size(LESS2[LESS2]))
print(np.size(TLESS2[TLESS2]))

plt.figure()
plt.imshow(ncres['mev_s_all'].values[:,:,0])
plt.show()

print('number complete pixels = {}'.format(np.size(MEVGcomp)/4))
print('number pixels less than 2 = {}'.format(np.size(MEVGless2)/4))
print('number complete pixels = {}'.format(np.size(MEVDcomp)/4))
print('number pixels less than 2 = {}'.format(np.size(MEVDless2)/4))
print('number complete pixels = {}'.format(np.size(MEVScomp)/4))
print('number pixels less than 2 = {}'.format(np.size(MEVSless2)/4))


# ESACOND = ncres['esa_d'].values > 25.0
# COMPP = np.logical_and(COMPP, ESACOND)
# comMEVG = MEVGvals[COMPP]
#
#
# MEVG = ncres['mev_g']
# MEVD = ncres['mev_d']
# MEVS = ncres['mev_s']
#
# NYG = ncres['']
#
# MEVGvals = MEVG.values
# MEVDvals = MEVD.values
# MEVSvals = MEVS.values
#
# nonnan = np.size(MEVGvals[~np.isnan(MEVGvals)])
# nonnan2 = np.size(MEVDvals[~np.isnan(MEVDvals)])
#
# plt.figure()
# plt.plot(np.ravel(MEVGvals), np.ravel(MEVDvals), 'o')
# plt.show()
#
# ncomplete = df.shape[0]
# print('num complete pixels = {}'.format(ncomplete))
#
# comMEVG = MEVGvals[COMPP]
# comMEVD = MEVDvals[COMPP]
# comMEVS = MEVSvals[COMPP]
# print(np.shape(comMEVG))
# print(np.shape(comMEVD))
# print(np.shape(comMEVS))
#
# print('number of nans = {}'.format(np.size(comMEVG[np.isnan(comMEVG)])))
# print('number of nans = {}'.format(np.size(comMEVD[np.isnan(comMEVD)])))
# print('number of nans = {}'.format(np.size(comMEVS[np.isnan(comMEVS)])))



