# compute maps of global extremes using MEV and GEV
# read yearly files and read all data for each year
# reading raw data and computing C, W, N , and Maxima year by year
# save them in array 1440x400x366
# simply as a double check

import os
import numpy as np
import downscale as down
import h5py
from pyhdf.SD import SD, SDC # works in the Cluster
from scipy.special import gamma


outdir_data  = os.path.join('..', 'output', 'pixel_stats')
datadir  = os.path.join('..', 'data', 'tmpa_raw_data')
outname = "tmpa_mev_global_param.hdf5"
outname2 = "tmpa_mev_global_quants.hdf5"

if not os.path.exists(outdir_data):
    os.makedirs(outdir_data)

filenames   = sorted([f for f in os.listdir(datadir) if f.endswith('.HDF')],
                     key = lambda name: name[5:13]+name[14:16])

# TODO: get these quantities from conusfun
TR = np.array([10, 20, 50, 100])
Fi = 1 - 1 / TR
thresh = 1  # threshold for computing excesses over threshold = ordinary events
nmax_miss = 36 # max number missing / year
min_n_years = 5  # min number of non-missing daily totals in any year
Nt = 365

compute_par = False
if compute_par:
    years_all = np.array([ int(s[5:9]) for s in filenames])
    years = np.unique(years_all)
    nyears = np.size(years)
    dates_all = np.array([ int(s[5:13]) for s in filenames])
    lats = np.arange(-49.875, 49.876, 0.25) # south to North
    lons = np.arange(-179.875, 179.876, 0.25) # West to East
    nlat = np.size(lats)
    nlon = np.size(lons)

    # initialize arrays for Annual Maxima and MEV Yearly parameter values:
    MAX = np.ones((nlon, nlat, nyears))*np.nan
    C = np.ones((nlon, nlat, nyears))*np.nan
    W = np.ones((nlon, nlat, nyears))*np.nan
    N = np.ones((nlon, nlat, nyears))*np.nan
    
    for ii in range(nyears):
        print('year = ', years[ii])
        year_ii = years[ii]
        # list all files in the given year
        yearly_files = [file for file in filenames
                        if int(file[5:9]) == year_ii]
        yearly_dates = np.unique(np.array([
               int(file[5:13]) for file in yearly_files]))
        ndates = np.size(yearly_dates)
        print('ndates = ', ndates)
        if ndates > Nt - nmax_miss:
            yearmat = np.zeros((nlon, nlat, ndates))
            # get number of files per day
            for jj in range(ndates):
                print(jj)
                date_jj = yearly_dates[jj]
                # read file and save prcp in array
                daily_files = [file for file in filenames
                               if int(file[5:13]) == date_jj]
                # ndfiles = len(daily_files)
                if len(daily_files) !=8:
                    print('WARNING: skip day, missing data!')
                else:
                    dailymat = np.ones((nlon, nlat, 8))*np.nan
                    # read prcp - all daily files::
                    for kk in range(8):
                        # read and write data:
                        fullname = os.path.join(datadir, daily_files[kk])
                        hdf = SD(fullname, SDC.READ)
                        prcpmat = hdf.select('precipitation')  # read matrix of rainfall estimates
                        dailymat[:, :, kk] = prcpmat[:] 
                    
                    for ix in range(nlon):
                        for iy in range(nlat):
                            sample0 = dailymat[ix, iy, :]
                            non_miss = np.logical_and(sample0 >=0,
                                                      ~np.isnan(sample0))
                            sample = sample0[non_miss]
                            # compute daily totals only if no missing values.
                            if np.size(sample) == 8:
                                yearmat[ix, iy, jj] = np.sum(sample)*3 # daily accumulations
                            else:
                                yearmat[ix, iy, jj] = -9999
                                
            for ix in range(nlon):
                for iy in range(nlat):
                    data0 = yearmat[ix, iy, :]
                    enough_data_mask = np.logical_and(
                             data0 > -0.1, np.logical_not(np.isnan(data0)))
                    data = data0[enough_data_mask]
                    excesses = data[data > thresh] - thresh
                    nexcesses = np.size(excesses)
                    if np.size(data) > Nt - nmax_miss:
                        MAX[ix, iy, ii] = np.max(data)
                        if nexcesses == 0:
                            N[ix, iy, ii] = 0
                            C[ix, iy, ii] = 1e-9
                            W[ix, iy, ii] = 1.0
                        elif nexcesses < 3:
                            N[ix, iy, ii] = 1
                            C[ix, iy, ii] = np.mean(excesses)/gamma(1+1/0.7)
                            W[ix, iy, ii] = 0.7
                        else:
                            N[ix, iy, ii], C[ix, iy, ii], W[ix, iy, ii] \
                                = down.wei_fit(excesses)
                    else: # Flag years with too many missing data
                        N[ix, iy, ii]   = -9999.0
                        C[ix, iy, ii]   = -9999.0
                        W[ix, iy, ii]   = -9999.0
                        MAX[ix, iy, ii] = -9999.0

        else: # not enough dates in current year, for entire dataset
            N[:, :, ii] = -9999.0*np.ones((nlon, nlat))
            C[:, :, ii] = -9999.0*np.ones((nlon, nlat))
            W[:, :, ii] = -9999.0*np.ones((nlon, nlat))
            MAX[:, :, ii] = -9999.0*np.ones((nlon, nlat))

    # save data:
    with h5py.File(os.path.join(outdir_data, outname), "w") as f:
        f.create_dataset("N", data = N, dtype='int32')
        f.create_dataset("C", data = C, dtype='f')
        f.create_dataset("W", data = W, dtype='f')
        f.create_dataset("Max", data = MAX,  dtype='f')
        f.create_dataset("lat", data = lats,  dtype='f')
        f.create_dataset("lon", data =lons,  dtype='f')
        f.create_dataset("years", data = years,  dtype='int32')

# read saved data::
with h5py.File(os.path.join(outdir_data, outname), "r") as fr:
    print(list(fr.keys()))
    Nmat = fr['N'][:]
    Cmat = fr['C'][:]
    Wmat = fr['W'][:]
    Maxima = fr['Max'][:]
    lats = fr['lat'][:]
    lons = fr['lon'][:]
    vyears = fr['years'][:]

nlon = np.size(lons)
nlat = np.size(lats)
ntr = len(TR)
qmev = np.ones((nlon, nlat, ntr))*np.nan
qgev = np.ones((nlon, nlat, ntr))*np.nan
num_complete_years = np.ones((nlon, nlat), dtype=int)*np.nan
# for ix in range(800, 801):
for ix in range(nlon):
    print(ix)
    for iy in range(nlat):
        # for iy in range(298, 299):
        # print(iy)
        # The following bool must be the same for the 4 variables: N, C, W, MAX
        maskN = Nmat[ix, iy, :] > -9000.0
        num_years_avail = np.size(maskN[maskN])
        num_complete_years[ix, iy] = num_years_avail
        if num_years_avail >= min_n_years:
            myN = Nmat[ix, iy, :][maskN]
            myC = Cmat[ix, iy, :][maskN]
            myW = Wmat[ix, iy, :][maskN]
            myMax = Maxima[ix, iy, :][maskN]
            (csi, psi, mu) = down.gev_fit_lmom(myMax)
            # x0 = 8*np.mean(myC) + 8*np.std(myC)
            for it in range(ntr):
                # x0 = np.mean(myC)*( -np.log((1-Fi[it])/np.mean(myN)))**(1/np.mean(myW))
                Fi0 = 0.5
                x0 = np.mean(myC)*( -np.log((1-Fi0**(1/np.mean(myN)))))**(1/np.mean(myW))
                qmev_quant = down.mev_quant(Fi[it], x0, myN, myC, myW,
                                                 thresh=thresh)
                if np.logical_not(qmev_quant[1]): # keep only if converging
                    qmev[ix, iy, it] = qmev_quant[0]
                qgev[ix, iy, it] = down.gev_quant(Fi[it], csi, psi, mu)

            # Fi0 = 0.50
            # x0 = np.mean(myC) * (-np.log((1 - Fi0) / np.mean(myN))) ** (1 / np.mean(myW))
            # qmev[ix, iy, :] = down.mev_quant(Fi, x0, myN, myC, myW,
            #                                   thresh=thresh)[0]
            # qgev[ix, iy, :] = down.gev_quant(Fi, csi, psi, mu)
        # else:
        #     qmev[ix, iy, :] = np.ones(ntr)*np.nan
        #     qgev[ix, iy, :] = np.ones(ntr)*np.nan


with h5py.File(os.path.join(outdir_data, outname2), "w") as f:
    f.create_dataset("qmev", data = qmev, dtype='f')
    f.create_dataset("qgev", data = qgev, dtype='f')
    f.create_dataset("num_complete_years", data =
                             num_complete_years, dtype='int32')
    f.create_dataset("Tr", data = TR,  dtype='int32')
    f.create_dataset("lat", data = lats,  dtype='f')
    f.create_dataset("lon", data =lons,  dtype='f')
    f.create_dataset("nmax_miss", data =nmax_miss,  dtype='f')
    f.create_dataset("thresh", data =thresh,  dtype='f')
    f.create_dataset("min_n_complete_years", data =min_n_years,  dtype='int32')
