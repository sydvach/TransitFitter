import batman
import numpy as np
# import minimint
import emcee
from scipy.optimize import minimize
import pickle 
import keplersplinev2
import pandas as pd
import matplotlib.pyplot as plt

class TransitFit:
    def __init__(self, t, y, yerr, limb_darkening_coeff, ndim=7):
        # store tess data in self object
        self.t = t # time [bjd - 2457000]
        self.y = y # normalized flux -- right now this is flattened, come back and make it detrend simultaneously 
        self.yerr = yerr # flux err 
        self.limb_darkening_coeff = limb_darkening_coeff # fix limbdarkening params from 
        self.ndim = ndim
        
    def transit_model(self, theta):
        model_flux = np.ones_like(self.t)
#         print(len(theta), 'here are the theta values',theta)
        
        n_planets = len(theta) // self.ndim
        theta = np.reshape(theta, (n_planets, -1))
        
        for n in range(n_planets):
            planet_params = theta[n]
            params = batman.TransitParams()
            params.t0 = planet_params[0]
            params.per = planet_params[1]
            params.rp = planet_params[2]
            params.a = planet_params[3]
            params.inc = planet_params[4]
            params.ecc = np.sqrt(planet_params[5]**2 + planet_params[6]**2)
            params.w = np.arctan2(planet_params[6], planet_params[5])
            params.limb_dark = 'quadratic'
            params.u = self.limb_darkening_coeff
            
            tmod = batman.TransitModel(params, self.t)
            model_flux *= tmod.light_curve(params)
            
        return model_flux
    
    def ln_like(self, theta):
        model = self.transit_model(theta)
        chisq = np.nansum((self.y - model) **2 / self.yerr**2) # come back and add jitter term 
        return -0.5 * chisq
    
    def ln_prior(self, theta, bounds):
        # check if the values are within the defined bounds
        # see if i can make it to where the user can chose to provide bounds and if not they code automatically sets bounds that are the limits of physically realizable values
        for th, bound in zip(theta, bounds):
            if not bound[0] < th < bound[1]:
#                 print(bound[0], th, bound[1])
                return -np.inf
        return 0.
        
    def ln_posterior(self, theta, bounds):
        lp = self.ln_prior(theta, bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_like(theta)
    
    
    def fit_transit(self, nwalkers, theta0, bounds, nsteps=500, burn_in=250):
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.ln_posterior, args=(bounds,))

        # run mcmc
        sampler.run_mcmc(theta0, nsteps, progress=True)

        # get chains and flatten them
        samples = sampler.get_chain(discard=burn_in)

        plt.figure()
        for i in range(self.ndim):
            plt.subplot(self.ndim, 1, i + 1)
            plt.plot(samples[:,:, i], alpha=0.3)
            plt.ylabel(f"Param {i}")
        plt.xlabel("Step")
        plt.show()
        
        samples = sampler.get_chain(discard=burn_in, flat=True)
        
        # Calculate median and percentiles
        derived_params = np.median(samples, axis=0)
        uncertainties = np.percentile(samples, [16, 84], axis=0) - derived_params

        return derived_params, uncertainties
        
data = pd.read_csv('toi5358_data/46631742.csv')
t = data['Time (BTJD)'].values
y = data['SAP_FLUX_DEBLEND'].values
yerr = data['PDCSAP_FLUX_ERR'].values

limb_darkening_coefficients = [0.4, 0.3]  # Replace with actual coefficients

theta0 = [2459450.138346 - 2457000, 2.6599176523685, 0.0402805, 9.07, 89, 0., 0.]
bounds = [[theta0[0] - 0.5, theta0[0] + 0.5],# bounds on t0
         [theta0[1] - 0.5, theta0[1] + 0.5],# bounds on period
         [0., 0.2],# bounds on rp/rstar
         [0, 20], # bounds on a/rstar
         [88,90], # bounds on inclination
         [-1, 1],# bounds for sqrt(e)cos(w)
         [-1, 1]]# bounds for sqrt(e)sin(w)

nwalkers = 25

theta0 = np.tile(theta0, (nwalkers, 1)) + 1e-4 * np.random.rand(nwalkers, len(theta0))


s1 = keplersplinev2.keplersplinev2(t[t<t[0]+26], y[t<t[0]+26], bkspace=0.8)
transit_fitter = TransitFit(t[t<t[0]+26],
                               y[t<t[0]+26] / s1,
                               yerr[t<t[0]+26], 
                               limb_darkening_coefficients)

derived_params, uncertainties = transit_fitter.fit_transit(nwalkers, \
                           theta0=theta0, \
                           bounds=bounds)


labels=['t0', 'per', 'rp/rstar', 'a/rstar', 'inc (deg)', 'sqrt(e)cos(w)', 'sqrt(e)sin(w)']

for i in range(len(labels)):
    print(labels[i], derived_params[i], uncertainties[0][i], '/', uncertainties[1][i])