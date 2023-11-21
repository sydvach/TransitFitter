import batman
import numpy as np
import minimint
import emcee
from scipy.optimize import minimize
import pickle 
import keplersplinev2
import pandas as pd
import matplotlib.pyplot as plt
import detrend
from sed_model import sed_model
from batman_model import batman_model


rsun = 69634000000.
msun = 1.989*10**33
G = 6.67*10**-8

filters = ['Tycho_B','Tycho_V', "2MASS_J","2MASS_H","2MASS_Ks","Gaia_G_DR2Rev","Gaia_BP_EDR3", 'Gaia_RP_EDR3']
ii = minimint.Interpolator(filters)



class TransitFit:
    def __init__(self, observations, 
                 obs_mags, obs_mags_err, 
                 limb_darkening_coeff, labels, num_planets =1):

        # take in a dictionary of data from multiple instruments 
        self.observations = observations
        
        # fix limbdarkening params from clare (2017)-- T for tess and Kp for CHEOPS
        self.limb_darkening_coeff = limb_darkening_coeff 
       
        # incorportate SED FIT
        self.obs_mags = obs_mags
        self.obs_mags_err = obs_mags_err
        self.labels = labels
        self.n_planets = num_planets
        
        print('self has been assigned', self)
        
    def transit_model(self, theta, i):
        return batman_model(self, theta, i)
    
    def sed_fit(self,theta):
        sim_mags, dist_mod = sed_model(self, theta)
        chisq = np.nansum(((sim_mags + dist_mod) - self.obs_mags) ** 2 / self.obs_mags_err ** 2)
        return chisq
       
    def ln_like(self, theta):

        chisq = np.array([])
        for i in range(len(self.observations)):
            y_obs = self.observations[i]['y']
            yerr_obs = self.observations[i]['yerr']
            model = self.transit_model(theta, i)
            
            inst_name = self.observations[i]['inst_name']
            jitter = theta[self.labels.index(f'jitter_'+inst_name)]
            chisq_transit = np.nansum((y_obs - model) **2 / np.sqrt(yerr_obs**2 + theta[-1]**2)**2)
            if np.abs(chisq_transit) < 0.0001:
                return -np.inf
            
            chisq = np.append(chisq, chisq_transit)
            
        chisq_sed = self.sed_fit(theta)
        chisq = np.append(chisq, chisq_sed)
        chisq = np.nansum(chisq)
        if np.abs(chisq) < 0.0001:
            return -np.inf
        return -0.5 * np.nansum(chisq)
    
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
        
        ndim = len(theta0)
#         print('we are here, about to make a matrix for theta0', theta0)
        theta0 = np.tile(theta0, (nwalkers, 1)) + 1e-4 * np.random.rand(nwalkers, ndim)
        self.ndim = ndim
#         print('about to run the sampler')
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
        
        # add calculate teff and rstar and rp

        return derived_params, uncertainties
    
    
