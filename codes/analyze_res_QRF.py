
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



#########################    MAIN    ###########################

# load data (eps/alp < 25 are removed here)
df = cfun.load_results_df(csvname='dfres_cluster_30580.csv')
df.to_csv( os.path.join(cfun.outdir_data, 'dfres_complete_only.csv'))
# predictor variables to be used in the analysis:

features = ['Cs', 'Ws', 'Ns', 'gam_s', 'melev', 'selev']
featurenames = [r'$C_L$', r'$W_L$', r'$N_L$',
                r'$\gamma_0$', r'$\mu_{e}$', r'$\sigma_{e}$']


# features = ['Cs', 'Ns', 'melev', 'selev']
# featurenames = [r'$C_L$', r'$N_L$', r'$\mu_{e}$', r'$\sigma_{e}$']
normalize_features = True

# normalize features:
if normalize_features:
    for fi in features:
        df[fi] = (df[fi] - np.mean(df[fi])) / np.std(df[fi])

# tunable parameters:
min_dist = 100 # 0 or 100
nshuffles = 20
test_size = 0.5
# max_features = 'sqrt' # suggested?
max_features = None # default
test_size_name = str(test_size).replace('.', '')
Tr = 50 # must be one of the Tr for which I saved the gauges, for EV analysis
min_samples_split = 10
n_estimators = 2000
# to have the same reshuffling for all 3 quantities to predict:
seeds = np.random.randint(1, 10000, size=nshuffles)


def remove_neighbours(df, min_dist=100):
    '''------------------------------------------------------------------------
    Given a dataframe with lat / lon values (clat, clon)
    keep only values (rows) at a minimum distance min_dist between them
    in order to reduce the effect of spatial correlation in the analysis
    min_dist = minimum distance in Km between two cell centers
    ------------------------------------------------------------------------'''
    # min_dist = 0.27
    # df['uncorrelated'] = np.ones(nrows, dtype = bool)
    nrows = df.shape[0]
    uncorr = np.ones(nrows, dtype = bool)
    for ii in range(nrows):
        # print(ii)
        for jj in range(ii + 1, nrows):
            if (uncorr[ii] == True) and (uncorr[jj] == True):
                dist_ij = down.haversine(df['clat'][ii], df['clat'][jj],
                                         df['clon'][ii], df['clon'][jj])
                if dist_ij < min_dist:
                    uncorr[jj] = False
    df2 = df.copy()
    df2 = df2.iloc[uncorr]
    print('sample size after removing correlated '
                   'sites = {}'.format(df2.shape[0]))
    return df2


def pred_error(dfr, label, features, seeds,*,
               min_samples_split=10, nshuffles=10,
               n_estimators=1000, test_size = 0.33, min_dist=100,
               max_features='none', plotmaps = True):
    ''' Predict error components using a quantile regression forest model '''

    # initialize arrays with prediction and test values:
    nX = np.size(features)
    Y_TEST   = []
    D_TEST   = []
    T_TEST   = []
    LOWER    = []
    UPPER    = []
    MEDIAN   = []
    EXPECTED = []
    LAT_TEST = []
    LON_TEST = []
    QRF_IMPS = np.zeros((nX, nshuffles))
    RF_IMPS = np.zeros((nX, nshuffles))

    for irs in range(nshuffles):

        # df = shuffle(df) # first reshuffle order of rows
        df0 = dfr.copy()
        df0 = df0.sample(frac=1, replace=False,
                       random_state=seeds[irs]).reset_index(drop=True)
        df = remove_neighbours(df0, min_dist=min_dist)



        X = df[features]
        # Xnames = list(X.columns)
        X = np.array(X)
        Yname = 'eta{}'.format(label) # label = error in the parameter C
        Y = df[Yname].values
        D = df['{}d'.format(label)].values # downscaled values to correct
        T = df['{}g'.format(label)].values # ground truth value
        Lat = df['clat'].values  # latitude value
        Lon = df['clon'].values  # longitiude value


        X_train, X_test, Y_train, Y_test, D_train, D_test, T_train, T_test, \
            Lat_train, Lat_test, Lon_train, Lon_test = train_test_split(
            X, Y, D, T, Lat, Lon, test_size=test_size, shuffle=False)


        if irs < 3 and plotmaps == True:
            plt.figure()
            plt.plot(dfr.clat, dfr.clon, '.')
            plt.plot(Lat_test, Lon_test, 'or')
            plt.plot(Lat_train, Lon_train, 'ob')
            plt.savefig(os.path.join(cfun.outplot, 'stats',
                                     'qrf_gen_{}_{}'.format(irs, label)))
            plt.close()

        # fit quantile regression forest
        rfqr = RandomForestQuantileRegressor(
                min_samples_split=min_samples_split,
                n_estimators=n_estimators, max_features=max_features)
        rfqr.fit(X_train, Y_train)
        upper = rfqr.predict(X_test, quantile=75)
        lower = rfqr.predict(X_test, quantile=25)
        median = rfqr.predict(X_test, quantile=50)
        qrf_imps = rfqr.feature_importances_
        # print(qrf_imps)

        # Fit random forest
        rfr = RandomForestRegressor(min_samples_split=min_samples_split,
                n_estimators=n_estimators, max_features=max_features)
        rfr.fit(X_train, Y_train)
        expected = rfqr.predict(X_test)
        rf_imps = rfr.feature_importances_
        # print(rf_imps)

        Y_TEST   = np.concatenate((Y_TEST, Y_test))
        D_TEST   = np.concatenate((D_TEST, D_test))
        T_TEST   = np.concatenate((T_TEST, T_test))
        UPPER    = np.concatenate((UPPER, upper))
        LOWER    = np.concatenate((LOWER, lower))
        MEDIAN   = np.concatenate((MEDIAN, median))
        EXPECTED = np.concatenate((EXPECTED, expected))
        LAT_TEST   = np.concatenate((LAT_TEST, Lat_test))
        LON_TEST   = np.concatenate((LON_TEST, Lon_test))

        QRF_IMPS[:, irs] = qrf_imps
        RF_IMPS [:, irs] = rf_imps
        # print(QRF_IMPS)

    MEAN_QRF_IMPS = np.mean(QRF_IMPS, axis=1)
    MEAN_RF_IMPS = np.mean(RF_IMPS, axis=1)
    CORR_QRF = D_TEST/(MEDIAN + 1)
    CORR_RF = D_TEST/(EXPECTED + 1)

    res = {'pred_qrf': MEDIAN, 'pred_rf': EXPECTED,
           'upper': UPPER, 'lower': LOWER,
           'y_test': Y_TEST, 'd_test': D_TEST, 't_test': T_TEST,
           'lat_test': LAT_TEST, 'lon_test': LON_TEST,
           'corr_qrf': CORR_QRF, 'corr_rf': CORR_RF,
           'qrf_imps': QRF_IMPS,
           'rf_imps': RF_IMPS,
           'mean_qrf_imps': MEAN_QRF_IMPS,
           'mean_rf_imps': MEAN_RF_IMPS
           }
    return res


resC = pred_error(df, 'C', features, seeds,
      nshuffles=nshuffles,
      min_samples_split=min_samples_split,
      n_estimators=n_estimators,
      test_size=test_size, min_dist=min_dist,
                  max_features=max_features)
resW = pred_error(df, 'W', features, seeds,
      nshuffles=nshuffles,
      min_samples_split=min_samples_split,
      n_estimators=n_estimators,
      test_size=test_size, min_dist=min_dist,
      max_features = max_features)
resN = pred_error(df, 'N', features, seeds,
      nshuffles=nshuffles,
      min_samples_split=min_samples_split,
      n_estimators=n_estimators,
      test_size=test_size, min_dist=min_dist,
      max_features = max_features)




# now for each TEST PIXEL extract the yearly values of the parameters
# and compute MEV quantiles
# ncres = cfun.load_results_netcdf(ncname='ncres_laptop_32.nc')
ncres = cfun.load_results_netcdf(ncname='ncres_cluster_30580.nc')
list(ncres.variables)
lats = ncres.coords['lat'].values
lons = ncres.coords['lon'].values

# x0 = 100.0
Fi = 1 - 1/Tr
ntest = np.size(resC['lat_test'])
ntest1 = np.size(resN['lat_test'])
ntest2 = np.size(resW['lat_test'])
print(ntest, ntest1,ntest2)
MEV_E = np.zeros(ntest) # mev with corrected parameters
MEV_S = np.zeros(ntest) # mev with corrected parameters
MEV_D = np.zeros(ntest) # mev with corrected parameters
MEV_G = np.zeros(ntest) # mev with corrected parameters
NE = np.zeros(ntest) # only for testing
CE = np.zeros(ntest) # only for testing
WE = np.zeros(ntest) # only for testing

Ng_test = np.zeros(ntest)
Cg_test = np.zeros(ntest)
Wg_test = np.zeros(ntest)

# plt.figure()
# plt.plot(resC['lon_test'], resC['lat_test'], 'o')
# plt.show()

print('number of test pixels = {}'.format(ntest))
for i in range(ntest):

    mylat = lats[np.argmin( np.abs(lats-resN['lat_test'][i]))]
    mylon = lons[np.argmin( np.abs(lons-resN['lon_test'][i]))]

    # extract downscaled yearly parameters:
    NYd = ncres['NYd'].sel(lat=mylat, lon=mylon).values
    CYd = ncres['CYd'].sel(lat=mylat, lon=mylon).values
    WYd = ncres['WYd'].sel(lat=mylat, lon=mylon).values


    # # extract rain gauge global parameters for comparison
    Ng_test[i] = ncres['Ng'].sel(lat=mylat, lon=mylon).values
    Cg_test[i] = ncres['Cg'].sel(lat=mylat, lon=mylon).values
    Wg_test[i] = ncres['Wg'].sel(lat=mylat, lon=mylon).values

    # correct yearly values: ('exact')
    NYe = NYd/(resN['pred_qrf'][i]+1)
    CYe = CYd/(resC['pred_qrf'][i]+1)
    WYe = WYd/(resW['pred_qrf'][i]+1)

    Nd = ncres['Nd'].sel(lat=mylat, lon=mylon).values
    Cd = ncres['Cd'].sel(lat=mylat, lon=mylon).values
    Wd = ncres['Wd'].sel(lat=mylat, lon=mylon).values
    Ne = Nd/(resN['pred_qrf'][i]+1)
    Ce = Cd/(resC['pred_qrf'][i]+1)
    We = Wd/(resW['pred_qrf'][i]+1)

    MEV_D[i] = down.mev_quant(Fi, 9.0*np.mean(CYd), NYd, CYd, WYd,
                             thresh=cfun.pixelkwargs['thresh'])[0]
    MEV_E[i] = down.mev_quant(Fi, 9.0*np.mean(CYe), NYe, CYe, WYe,
                             thresh=cfun.pixelkwargs['thresh'])[0]
    MEV_G[i] = ncres['mev_g'].sel(lat=mylat, lon=mylon, TR=Tr)
    MEV_S[i] = ncres['mev_s'].sel(lat=mylat, lon=mylon, TR=Tr)

# compute relative errors in estimated quantiles:
DOWN_ERROR = (MEV_D - MEV_G)/MEV_G
CORR_ERROR = (MEV_E - MEV_G)/MEV_G
TMPA_ERROR = (MEV_S - MEV_G)/MEV_G




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
axes[0].text(0.10, 0.95, '(a)', transform=axes[0].transAxes, size=18)
axes[0].scatter(MEV_G, MEV_S, marker='o', color= 'r',
            edgecolor='k', s=50, label=r'TMPA quantiles $\hat{h}_L$')
axes[0].scatter(MEV_G, MEV_D, marker='o', color= 'b',
            edgecolor='k', s=50, label=r'downscaled quantiles $\hat{h}_{0,d}$')
axes[0].scatter(MEV_G, MEV_E, marker='o', color= 'g',
            edgecolor='k', s=50, label=r'corrected quantiles $\hat{h}_{0,c}$')
# ylims = plt.gca().get_ylim()
axes[0].set_xlim([0, 350])
axes[0].set_ylim([0, 350])
axes[0].plot(MEV_G, MEV_G, 'k')
plt.suptitle('rainfall quantiles Tr = {}'.format(Tr))
# plt.legend()
axes[0].set_xlabel('gauge quantiles [mm]')
axes[0].set_ylabel('TMPA rainfall quantiles [mm]')

sns.distplot(TMPA_ERROR, hist=False, kde=True,
             vertical=True,
             ax = axes[1],
             bins=int(180/10), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}, label = r'TMPA $\hat{h}_{L}$')

sns.distplot(DOWN_ERROR, hist=False, kde=True,
             vertical=True,
             ax = axes[1],
             bins=int(180/10), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}, label = r'downscaled $\hat{h}_{0,d}$')


sns.distplot(CORR_ERROR, hist=False, kde=True,
             vertical=True,
             ax = axes[1],
             bins=int(180/10), color = 'green',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}, label = r'corrected $\hat{h}_{0,c}$')

# sns.distplot(MEV_G, hist=True, kde=False,
#              norm_hist=True,
#              vertical=True,
#              ax = axes[1],
#              bins=int(180/10), color = 'black',
#              hist_kws={'edgecolor':'black', 'alpha':0.4},
#              kde_kws={'linewidth': 4}, label = 'gauges')
axes[1].text(0.10, 0.95, '(b)', transform=axes[1].transAxes, size=18)
axes[1].legend()
# axes[1].set_ylim([0, 350])
axes[1].set_xlabel('frequency density')
# axes[1].set_ylabel('rainfall quantiles [mm]')
axes[1].set_ylabel('quantile relative error')
fig.savefig(os.path.join(cfun.outplot,
                            'quantile_corrected_rainfallpdf_{}_testsize_{}.png'.format(
                                min_dist, test_size_name)), dpi=300)
plt.show()


psize = 12
fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (12, 8.2))
xy = np.vstack([resC['y_test'], resC['pred_qrf']])
z = gaussian_kde(xy)(xy)
axes[0,0].scatter(resC['y_test'], resC['pred_qrf'],
                  c=z, s=50, edgecolor='', cmap = 'jet')
axes[0,0].text(0.05, 0.90, '(a)', transform=axes[0,0].transAxes, size=psize)
amin = min(np.min(resC['y_test']), np.min(resC['pred_qrf']))
amax = max(np.max(resC['y_test']), np.max(resC['pred_qrf']))
axes[0,0].plot([amin, amax],[amin, amax], 'k')
axes[0,0].set_xlabel('test $\eta_C$')
axes[0,0].set_ylabel('predicted $\eta_C$')

xy = np.vstack([resW['y_test'], resW['pred_qrf']])
z = gaussian_kde(xy)(xy)
axes[0,1].scatter(resW['y_test'], resW['pred_qrf'],
                  c=z, s=50, edgecolor='', cmap = 'jet')
axes[0,1].text(0.05, 0.90, '(b)', transform=axes[0,1].transAxes, size=psize)
amin = min(np.min(resW['y_test']), np.min(resW['pred_qrf']))
amax = max(np.max(resW['y_test']), np.max(resW['pred_qrf']))
axes[0,1].plot([amin, amax],[amin, amax], 'k')
axes[0,1].set_xlabel('test $\eta_W$')
axes[0,1].set_ylabel('predicted $\eta_W$')

xy = np.vstack([resN['y_test'], resN['pred_qrf']])
z = gaussian_kde(xy)(xy)
axes[0,2].scatter(resN['y_test'], resN['pred_qrf'],
                  c=z, s=50, edgecolor='', cmap = 'jet')
axes[0,2].text(0.05, 0.90, '(c)', transform=axes[0, 2].transAxes, size=psize)
amin = min(np.min(resN['y_test']), np.min(resN['pred_qrf']))
amax = max(np.max(resN['y_test']), np.max(resN['pred_qrf']))
axes[0,2].plot([amin, amax],[amin, amax], 'k')
axes[0,2].set_xlabel('test $\eta_N$')
axes[0,2].set_ylabel('predicted $\eta_N$')

xy = np.vstack([resC['t_test'], resC['corr_qrf']])
z = gaussian_kde(xy)(xy)
axes[1,0].scatter(resC['t_test'], resC['corr_qrf'],
                  c=z, s=50, edgecolor='', cmap = 'jet')
axes[1,0].text(0.05, 0.90, '(d)', transform=axes[1,0].transAxes, size=psize)
amin = min(np.min(resC['t_test']), np.min(resC['corr_qrf']))
amax = max(np.max(resC['t_test']), np.max(resC['corr_qrf']))
axes[1,0].plot([amin, amax],[amin, amax], 'k')
axes[1,0].set_ylabel('$C_{0,c} \quad [mm]$')
axes[1,0].set_xlabel('$C_{0,g} \quad [mm]$')

xy = np.vstack([resW['t_test'], resW['corr_qrf']])
z = gaussian_kde(xy)(xy)
axes[1,1].scatter(resW['t_test'], resW['corr_qrf'],
                  c=z, s=50, edgecolor='', cmap = 'jet')
axes[1,1].text(0.05, 0.90, '(e)', transform=axes[1,1].transAxes, size=psize)
amin = min(np.min(resW['t_test']), np.min(resW['corr_qrf']))
amax = max(np.max(resW['t_test']), np.max(resW['corr_qrf']))
axes[1,1].plot([amin, amax],[amin, amax], 'k')
axes[1,1].set_ylabel('$W_{0,c} \quad [-]$')
axes[1,1].set_xlabel('$W_{0,g} \quad [-]$')

xy = np.vstack([resN['t_test'], resN['corr_qrf']])
z = gaussian_kde(xy)(xy)
axes[1,2].scatter(resN['t_test'], resN['corr_qrf'],
                  c=z, s=50, edgecolor='', cmap = 'jet')
axes[1,2].text(0.05, 0.90, '(f)', transform=axes[1, 2].transAxes, size=psize)
amin = min(np.min(resN['t_test']), np.min(resN['corr_qrf']))
amax = max(np.max(resN['t_test']), np.max(resN['corr_qrf']))
axes[1,2].plot([amin, amax],[amin, amax], 'k')
axes[1,2].set_ylabel(' $N_{0,c} \quad [days]$')
axes[1,2].set_xlabel('$N_{0,g} \quad [days]$')
plt.tight_layout()
# if keep_min_dist:
fig.savefig( os.path.join(cfun.outplot,
        'rf_corrected_par_{}_testsize_{}.png'.format(
        min_dist, test_size_name)), dpi=300)
plt.show()



# violin plots of predictor importances

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

# pos = [1, 2, 3, 4, 5, 6]
npos = np.size(features)
pos = list(np.arange(1, npos + 1))


imp_list_N = [resN['qrf_imps'][i, :] for i in range(npos)]
imp_list_C = [resC['qrf_imps'][i, :] for i in range(npos)]
imp_list_W = [resW['qrf_imps'][i, :] for i in range(npos)]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 9))

parts0 = axes[0].violinplot(imp_list_N, pos, points=100, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
axes[0].set_title(r'Error in the number of events $\eta_N$', fontsize = 20)
axes[0].set_ylim([0, 0.8])
axes[0].set_ylabel('Importance')
axes[0].text(0.03, 0.85, '(a)', transform=axes[0].transAxes, size=18)



parts1 = axes[1].violinplot(imp_list_C, pos, points=100, vert=True, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
axes[1].set_title(r'Error in the Weibull scale parameter $\eta_C$', fontsize=20)
axes[1].set_ylim([0, 0.8])
axes[1].set_ylabel('Importance')
axes[1].text(0.03, 0.85, '(b)', transform=axes[1].transAxes, size=18)


parts2 = axes[2].violinplot(imp_list_W, pos, points=100, vert=True, widths=0.6,
                      showmeans=False, showextrema=False, showmedians=True)
axes[2].set_title(r'Error in the Weibull shape parameter $\eta_W$', fontsize=20)
axes[2].set_ylim([0, 0.8])
axes[2].set_ylabel('Importance')
axes[2].text(0.03, 0.85, '(c)', transform=axes[2].transAxes, size=18)

parts = [parts0, parts1, parts2]
col_parts = ['#D43F3A', '#226e19', '#3c2e78']
dat_parts = [imp_list_N, imp_list_C, imp_list_W]

for ip, pp in enumerate(parts):
    vp = pp['cmedians']
    vp.set_edgecolor('black')
    vp.set_linewidth(2)

    for pc in pp['bodies']:
        pc.set_facecolor(col_parts[ip])
        pc.set_edgecolor('black')
        pc.set_alpha(1)


    for ipl, pl, in enumerate(pos):
        data = dat_parts[ip][ipl]
        sorted_data = np.sort(data)

        quartile1, medians, quartile3 = np.percentile(
                            data, [25, 50, 75], axis=0)
        whiskersMin, whiskersMax = adjacent_values(
                            sorted_data, quartile1, quartile3)

        axes[ip].vlines(pos[ipl], quartile1, quartile3,
                        color='k', linestyle='-', lw=5)
        axes[ip].vlines(pos[ipl], whiskersMin, whiskersMax,
                        color='k', linestyle='-', lw=1)
for ax in axes.flat:
    ax.set_xticks(pos)
    ax.set_xticklabels(featurenames)

fig.subplots_adjust(hspace=0.4)
fig.savefig( os.path.join(cfun.outplot,
            'rf_importance_par_{}_testsize_{}.png'.format(
            min_dist, test_size_name)), dpi=300)
plt.show()

