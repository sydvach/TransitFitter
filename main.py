from planet_fit import *
import pandas as pd
import numpy as np
# import minimint 
from astropy.io import fits

rsun = 69634000000.
msun = 1.989*10**33
G = 6.67*10**-8

filters = ['Tycho_B','Tycho_V', "2MASS_J","2MASS_H","2MASS_Ks","Gaia_G_DR2Rev","Gaia_BP_EDR3", 'Gaia_RP_EDR3']
ii = minimint.Interpolator(filters)

def load_lcf(file_path, instrument='TESS'):
    if instrument == 'TESS':
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1) # open light curve file
        t = data[:,0]
        f = data[:,1]
        ferr = data[:,2]
        return t, f, ferr
    
    if instrument == 'CHEOPS':
        obs = fits.open(file_path)[1].data  
        obs = obs[obs['STATUS'] == 0.]
        
        df = {'t':obs['BJD_TIME'] - 2457000., 'y':obs['FLUX'] / np.nanmedian(obs['FLUX']), 'yerr':obs['FLUXERR'] / np.nanmedian(obs['FLUX']),\
                'bg': obs['BACKGROUND'], 'contam': obs['CONTA_LC'], 'smear': obs['SMEARING_LC'],\
                'phi': obs['ROLL_ANGLE'], 'dx': obs['CENTROID_X'], 'dy': obs['CENTROID_Y']}
        return df


def load_sed_mags(file_path):
    data = np.genfromtxt(file_path,invalid_raise=False)
#     print(data)
    obs_mags = data[:, 1]
    obs_mags_unc = data[:, 2]
    return obs_mags, obs_mags_unc

    
mags, mag_unc =load_sed_mags('toi5358_data/mags')

# limb darkening coefficents from Claret 2017 (TESS)
# u1 = 0.46256916800000014 +/- 0.02098389464298218
# u2 = 0.18296777100000003 +/- 0.015062629770115143 
limb_darkening_coefficients_TESS = [0.46, 0.18]  

# limb darkening coefficents from Claret 2017 Kepler band (similar to CHEOPS)
# u1 = 0.61467663 +/- 0.02709113508222016
# u2 = 0.11798228799999999 +/- 0.021730601977051994
limb_darkening_coefficients_CHEOPS = [0.61, 0.12]  


theta0 = [0.736, #mstar (m_sun) 0
          120, #age  (myr) 1
          7.24, #plx (mas) 2
          0.108, #feh 3
          0.001, #tess jitter 4
          0.01, #cheops_jitter 5
          2459450.138346 - 2457000, #t0 6
          2.6599176523685, #per 7
          0.0402805, #rprstar 8
          9.07, #arstar 9 
          89, #inc 10
          0., #ecosw 11
          0.] #esinw 12

bounds = [[0.2, 2], #mstar
          [50,200], #age
          [7.24-1., 7.24+1.],#plx
          [.108-0.1, .174+.1],#metalixity
          [0., .5],# bounds for jitter term instrument 1
         [0., .5],# bounds for jitter term instrument 2
    [theta0[6] - 0.5, theta0[6] + 0.5],# bounds on t0
         [theta0[7] - 0.25, theta0[7] + 0.25],# bounds on period
         [0., 0.2],# bounds on rp/rstar
         [0, 20], # bounds on a/rstar
         [88,90], # bounds on inclination
         [-1, 1],# bounds for sqrt(e)cos(w)
         [-1, 1],# bounds for sqrt(e)sin(w)
         ] # bounds for jitter term instrument 2

t,y,yerr = load_lcf('toi5358_data/46631742.csv')
# s1 = keplersplinev2.keplersplinev2(t[t<t[0]+26], y[t<t[0]+26], bkspace=0.9)
tess = {'inst_name': 'TESS', 't':t, 'y':y 'yerr': yerr, 'limb_dark': limb_darkening_coefficients_TESS
}

cheops1 = load_lcf('toi5358_data/toi5358_cheops_obs1.fits', 'CHEOPS')
cheops1['inst_name'] = 'CHEOPS'
cheops1['limb_dark'] = limb_darkening_coefficients_CHEOPS


cheops2 = load_lcf('toi5358_data/toi5358_cheops_obs2.fits', 'CHEOPS')
cheops2['inst_name'] = 'CHEOPS'
cheops2['limb_dark'] = limb_darkening_coefficients_CHEOPS

observations = [tess, cheops1, cheops2]

labels=['mstar', 'age', 'plx', 'feh', 'jitter_TESS', 'jitter_CHEOPS','t0_1', 'per_1', 'rprstar_1', 'arstar_1', 'inc_1', 'sqrt(e)cos(w)_1', 'sqrt(e)sin(w)_1']

limb_darkening_coefficients = [limb_darkening_coefficients_TESS, limb_darkening_coefficients_CHEOPS]
transit_fitter = TransitFit(observations, 
                            mags, mag_unc,
                               limb_darkening_coefficients, labels)

nwalkers = 50
derived_params, uncertainties = transit_fitter.fit_transit(nwalkers, \
                           theta0=theta0, \
                           bounds=bounds)


for i in range(len(labels)):
    print(labels[i], derived_params[i], uncertainties[0][i], '/', uncertainties[1][i])
#     print()
