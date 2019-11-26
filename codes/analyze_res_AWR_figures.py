
import os
import conusfun as cfun
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import downscale as down
down.matplotlib_update_settings()
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib as mpl

# read data output from CONUS analysis
ncres = cfun.load_results_netcdf(ncname='ncres_cluster_30580.nc')
list(ncres.variables)
df = cfun.load_results_df(csvname='dfres_cluster_30580.csv')
dfg = pd.read_csv(os.path.join(cfun.outdir_data,
                   'dfres_gauges_1451.csv'), index_col=0)

# plot 4x4 maps with extremes gridded and scatter plot
mytr = 50
#
# plt.figure()
# plt.plot(df['Ns'], df['Nd'], 'o')
# plt.plot(df['Ns'], df['Ns'], 'k')
# plt.show()


boxlat = ncres.coords['lat'].values
boxlon = ncres.coords['lon'].values
nlon = np.size(boxlon)
nlat = np.size(boxlat)

# def conus_map_data(grid_data, boxlat, boxlon, **kwargs):
elev = ncres['melev'].values
mev_data0 = ncres['mev_s_all'].sel(TR=mytr).values
gev_data0 = ncres['gev_s_all'].sel(TR=mytr).values




# apply mask
mev_data = np.zeros((nlon, nlat))
gev_data = np.zeros((nlon, nlat))
nan_data = np.zeros((nlon, nlat))*np.nan
for i in range(nlon):
    for j in range(nlat):
        if elev[i,j] > -50:
            mev_data[i,j]=mev_data0[i,j]
            gev_data[i,j]=gev_data0[i,j]
        else:
            mev_data[i,j] = np.nan
            gev_data[i,j] = np.nan

print('mev max val = {}'.format( np.max(mev_data[~np.isnan(mev_data)])))
print('gev max val = {}'.format( np.max(gev_data[~np.isnan(gev_data)])))
print('mev min val = {}'.format( np.min(mev_data[~np.isnan(mev_data)])))
print('gev min val = {}'.format( np.min(gev_data[~np.isnan(gev_data)])))
#
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_axes([0.1,0.1,0.8,0.8])

def conusmap(grid_data=None, boxlat=None, boxlon=None, label=None,
             scatter=False, ax=None, df=None, dfkey=None,
             vmin=20, vmax=500, colorbar=True):
    m = Basemap(projection='merc',
                resolution='l',
                # get these from conusdata
                llcrnrlon = -130,   # lower right cornet lat
                llcrnrlat = 22,
                urcrnrlon = -60,  # upper right corner
                urcrnrlat = 50,
                ax=ax)  # latitude of the true scale
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
    np.shape(y)
    newdata = maskoceans(maplon, maplat, grid_data)
    # if grid:
    cs = m.pcolormesh(x, y, newdata, cmap = 'jet', vmin=vmin, vmax=vmax,
                      norm=mpl.colors.LogNorm() ## mod
                      )
    if not scatter:
        if colorbar:
            cbar = m.colorbar(cs,location='bottom',pad="8%",
                              norm=mpl.colors.LogNorm() ## mod
                              )
            cbar.set_label(label, fontsize=18)

            cbar.set_ticks([20, 50, 100, 200, 500])
            cbar.set_ticklabels([20, 50, 100, 200, 500])
    if scatter:
        sc = m.scatter(df['LON'].values, df['LAT'].values,
                               marker='o', c= df[dfkey].values, s=5,
                       zorder=10 , latlon=True, cmap='jet',
                       vmin=vmin, vmax=vmax, norm=mpl.colors.LogNorm()) ## mod
        if colorbar:
            cbar2 = m.colorbar(sc,location='bottom',pad="8%")
            cbar2.set_label(label, fontsize=18)
            cbar2.set_ticks([20, 50, 100, 200, 500])
            cbar2.set_ticklabels([20, 50, 100, 200, 500])

    m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='w', lakes=True)
    return m
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16,9))

axes[0,0].set_title('MEV Quantiles Tr = {}'.format(mytr), fontsize=18)
axes[0,0].text(0.05, 0.05, '(a)', transform=axes[0,0].transAxes, size=18)
m1 = conusmap(grid_data=mev_data, boxlat=boxlat, ax=axes[0,0],
              boxlon=boxlon, colorbar=True,
              label='Maximum annual daily rainfall [mm]')

axes[0,1].set_title('GEV Quantiles Tr = {}'.format(mytr), fontsize=18)
axes[0,1].text(0.05, 0.05, '(b)', transform=axes[0,1].transAxes, size=18)
m2 = conusmap(grid_data=gev_data, boxlat=boxlat, ax=axes[0,1],
              boxlon=boxlon, colorbar=True,
              label='Maximum annual daily rainfall [mm]')

# axes[1,0].set_title('MEV Quantiles Tr = {}'.format(mytr))
axes[1,0].text(0.05, 0.05, '(c)', transform=axes[1,0].transAxes, size=18)
m3 = conusmap(grid_data=nan_data, boxlat=boxlat, ax=axes[1,0],
              boxlon=boxlon, label='Maximum annual daily rainfall [mm]', df=dfg,
              scatter=True, dfkey='mev_{}'.format(mytr))

# axes[1,1].set_title('GEV Quantiles Tr = {}'.format(mytr))
axes[1,1].text(0.05, 0.05, '(d)', transform=axes[1,1].transAxes, size=18)
m4 = conusmap(grid_data=nan_data, boxlat=boxlat, ax=axes[1,1],
              boxlon=boxlon, label='Maximum annual daily rainfall [mm]', df=dfg,
              scatter=True, dfkey='gev_{}'.format(mytr))

plt.tight_layout()

# axes[0,0].get_position()
# axes[0,1].get_position()
# axes[1,0].get_position()
# axes[1,1].get_position()


# axes[0,0].set_position([0.05, 0.53, 0.41, 0.38])
# axes[0,1].set_position([0.53, 0.53, 0.41, 0.38])
# axes[1,0].set_position([0, 0.08, 0.51, 0.42])
# axes[1,1].set_position([0.48, 0.08, 0.51, 0.42])
plt.savefig(os.path.join(cfun.outplot, 'maps_evd.png'), dpi=300)
plt.show()

# figure 2 - correlation downscaling vs Longitude
fig, axes = plt.subplots(figsize=(12, 6.5), ncols=2, nrows=1)
axes[0].plot(df['clon'], df['esa_s'], 'or', markersize = 3,
             label=r'QPEs $\epsilon^{(L)} / \alpha^{(L)}$')
axes[0].plot(df['clon'], df['esa_d'], 'ob', markersize = 3,
             label=r'downscaled $\epsilon / \alpha$')
axes[0].plot(df['clon'], df['esa_g'], 'og', markersize = 3,
             label=r'gauges $\epsilon_g / \alpha_g$')
axes[0].legend()
axes[0].set_xlabel('Longitude [West]')
axes[0].set_ylabel(r'Correlation kernel spatial scale  [km]')
axes[1].plot(df['clon'], df['gam_s'], 'or', markersize = 3,
             label=r'QPEs $\gamma_{0,L}$')
axes[1].plot(df['clon'], df['gam_d'], 'ob', markersize = 3,
             label=r'downscaled $\gamma_{0,d}$')
axes[1].plot(df['clon'], df['gam_g'], 'og', markersize = 3,
             label=r'gauge $\gamma_{0,g}$')
axes[1].legend()
axes[1].set_xlabel('Longitude [West]')
axes[1].set_ylabel(r'Variance reduction factor $\gamma_0$ [-]')
plt.tight_layout()
fig.savefig(os.path.join(cfun.outplot, 'down_corr_long.png'), dpi=300)
plt.show()



# figure 3 - errors in the Weibull parameters
# first remove cells with epsilon/alpha < 25:
etaC = ncres['etaC'].values
etaW = ncres['etaW'].values
etaN = ncres['etaN'].values
etaGAM = ncres['etaGAM'].values
esa = ncres['esa_d'].values
# esa_cond = np.logical_and( np.isnan(esa), esa < 25.0)
esa_cond =  esa < 25.0
etaC[esa_cond] = np.nan
etaW[esa_cond] = np.nan
etaN[esa_cond] = np.nan
etaGAM[esa_cond] = np.nan
np.size(etaC[~np.isnan(etaC)])

fig, axes = plt.subplots(2,2, figsize=(12,8))


# axes[1,1].set_title(r'Variance reduction factor $\gamma_{d}$')
axes[0,0].text(0.05, 0.05, '(a)', transform=axes[0,0].transAxes, size=18)
m2 = conusmap(grid_data=ncres['etaGAM'].values, boxlat=boxlat, ax=axes[0,0],
              boxlon=boxlon, label=r'$\eta_{\gamma}$',
              vmin = np.nanmin(ncres['etaGAM'].values),
              vmax = np.nanmax(ncres['etaGAM'].values))


# axes[1,0].set_title('MEV Quantiles Tr = {}'.format(mytr))
axes[0, 1].text(0.05, 0.05, '(b)', transform=axes[0, 1].transAxes, size=18)
m3 = conusmap(grid_data=ncres['etaN'].values, boxlat=boxlat, ax=axes[0,1],
              boxlon=boxlon, label=r'$\eta_{N}$',
              vmin = np.nanmin(ncres['etaN'].values),
              vmax = np.nanmax(ncres['etaN'].values))

# vmin=-0.25,
# vmax=0.25)
# axes[0].set_title('MEV Quantiles Tr = {}'.format(mytr))
axes[1,0].text(0.05, 0.05, '(c)', transform=axes[1,0].transAxes, size=18)
m1 = conusmap(grid_data=ncres['etaC'].values, boxlat=boxlat, ax=axes[1,0],
              boxlon=boxlon, label=r'$\eta_{C}$',
              vmin = np.nanmin(ncres['etaC'].values),
              vmax = np.nanmax(ncres['etaC'].values))

# axes[0].set_title('GEV Quantiles Tr = {}'.format(mytr))
axes[1,1].text(0.05, 0.05, '(d)', transform=axes[1,1].transAxes, size=18)
# axes[0].text(0.15, 0.15, 'TEMPORARY FIGURE', transform=axes[0,1].transAxes, size=28)
m2 = conusmap(grid_data=ncres['etaW'].values, boxlat=boxlat, ax=axes[1,1],
              boxlon=boxlon, label=r'$\eta_{w}$',
              vmin = np.nanmin(ncres['etaW'].values),
              vmax = np.nanmax(ncres['etaW'].values))


plt.tight_layout()
plt.savefig(os.path.join(cfun.outplot, 'maps_NCW.png'), dpi=300)
plt.show()

# figure 4: scatter plots for NCW
# Nd = ncres['Nd'].values[~np.isnan(ncres['Nd'].values)]
# Cd = ncres['Cd'].values[~np.isnan(ncres['Cd'].values)]
# Wd = ncres['Wd'].values[~np.isnan(ncres['Wd'].values)]
# Ng = ncres['Ng'].values[~np.isnan(ncres['Ng'].values)]
# Cg = ncres['Cg'].values[~np.isnan(ncres['Cg'].values)]
# Wg = ncres['Wg'].values[~np.isnan(ncres['Wg'].values)]

Ns = df['Ns']
Cs = df['Cs']
Ws = df['Ws']

Nd = df['Nd']
Cd = df['Cd']
Wd = df['Wd']
Ng = df['Ng']
Cg = df['Cg']
Wg = df['Wg']
Gg = df['gam_g']
Gd = df['gam_d']

# plt.figure()
# plt.plot(Cs, Cd, 'o')
# plt.plot(Cs, Cs, 'k')
# plt.show()
#
#
# plt.figure()
# plt.plot(Ws, Wd, 'o')
# plt.plot(Ws, Ws, 'k')
# plt.show()
#
#
# plt.figure()
# plt.plot(Ns, Nd, 'o')
# plt.plot(Ns, Ns, 'k')
# plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (8, 8))

xy = np.vstack([Gd, Gg])
z = gaussian_kde(xy)(xy)
axes[0, 0].scatter(Gg, Gd,
                   c=z, s=20, edgecolor='', cmap = 'jet')
axes[0, 0].text(0.05, 0.90, '(a)', transform=axes[0, 0].transAxes, size=18)
amin = min(np.min(Gg), np.min(Gd))
amax = max(np.max(Gg), np.max(Gd))
axes[0, 0].plot([amin, amax],[amin, amax], 'k')
axes[0, 0].set_xlabel('Gauge $\gamma_{0,g}$')
axes[0, 0].set_ylabel('Downscaled $\gamma_{0,d}$')
axes[0, 0].set_xlim([0.5, 1])
axes[0, 0].set_ylim([0.5, 1])
axes[0, 0].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])

xy = np.vstack([Nd, Ng])
z = gaussian_kde(xy)(xy)
axes[0, 1].scatter(Ng, Nd,
  c=z, s=20, edgecolor='', cmap = 'jet')
axes[0, 1].text(0.05, 0.90, '(b)', transform=axes[0, 1].transAxes, size=18)
amin = min(np.min(Ng), np.min(Nd))
amax = max(np.max(Ng), np.max(Nd))
# amax = max(np.max(resC['y_test']), np.max(resC['pred_qrf']))
axes[0, 1].plot([amin, amax],[amin, amax], 'k')
axes[0, 1].set_xlabel('Gauge $N_{0,g}$')
axes[0, 1].set_ylabel('Downscaled $N_{0,d}$')
axes[0, 1].set_xlim([0, 180])
axes[0, 1].set_ylim([0, 180])
axes[0, 1].set_xticks([0, 50, 100, 150])
axes[0, 1].set_yticks([0, 50, 100, 150])

xy = np.vstack([Wd, Wg])
z = gaussian_kde(xy)(xy)
axes[1, 1].scatter(Wg, Wd,
  c=z, s=20, edgecolor='', cmap = 'jet')
axes[1, 1].text(0.05, 0.90, '(d)', transform=axes[1, 1].transAxes, size=18)
amin = min(np.min(Wg), np.min(Wd))
amax = max(np.max(Wg), np.max(Wd))
# amax = max(np.max(resC['y_test']), np.max(resC['pred_qrf']))
axes[1, 1].plot([amin, amax],[amin, amax], 'k')
axes[1, 1].set_xlabel('Gauge $w_{0,g}$')
axes[1, 1].set_ylabel('Downscaled $w_{0,d}$')
axes[1, 1].set_xlim([0.4, 1.2])
axes[1, 1].set_ylim([0.4, 1.2])
axes[1, 1].set_xticks([0.5, 0.7, 0.9, 1.1])
axes[1, 1].set_yticks([0.5, 0.7, 0.9, 1.1])

xy = np.vstack([Cd, Cg])
z = gaussian_kde(xy)(xy)
axes[1, 0].scatter(Cg, Cd,
  c=z, s=20, edgecolor='', cmap = 'jet')
axes[1, 0].text(0.05, 0.90, '(c)', transform=axes[1, 0].transAxes, size=18)
amin = min(np.min(Cg), np.min(Cd))
amax = max(np.max(Cg), np.max(Cd))
axes[1, 0].plot([amin, amax],[amin, amax], 'k')
axes[1, 0].set_xlabel('Gauge $C_{0,g}$')
axes[1, 0].set_ylabel('Downscaled $C_{0,d}$')
axes[1, 0].set_xlim([0, 21])
axes[1, 0].set_ylim([0, 21])
axes[1, 0].set_xticks([0, 5, 10, 15, 20])
axes[1, 0].set_yticks([0, 5, 10, 15, 20])

plt.tight_layout()
plt.savefig(os.path.join(cfun.outplot, 'down_NCW.png'), dpi=300)
plt.show()

