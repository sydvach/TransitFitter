import numpy as np
import batman
from scipy.optimize import minimize
import keplersplinev2
import detrend

def batman_model(self, theta, i):
#         for i in range(n_obs):
    model_flux = np.ones_like(self.observations[i]['t'])
    
    for n in range(self.n_planets):
        params = batman.TransitParams()
        params.t0 = theta[self.labels.index(f't0_{n+1}')]
        params.per = theta[self.labels.index(f'per_{n+1}')]
        params.rp = theta[self.labels.index(f'rprstar_{n+1}')]
        params.a = theta[self.labels.index(f'arstar_{n+1}')]
        params.inc = theta[self.labels.index(f'inc_{n+1}')]
        params.ecc = np.sqrt(theta[self.labels.index(f'sqrt(e)cos(w)_{n+1}')]**2 + theta[self.labels.index(f'sqrt(e)sin(w)_{n+1}')]**2)
        params.w = np.arctan2(theta[self.labels.index(f'sqrt(e)sin(w)_{n+1}')], theta[self.labels.index(f'sqrt(e)cos(w)_{n+1}')])
        params.limb_dark = 'quadratic'
        params.u = self.observations[i]['limb_dark']
        
        tmod = batman.TransitModel(params, self.observations[i]['t'])
        model_flux *= tmod.light_curve(params)

        if self.observations[i]['inst_name'] == 'CHEOPS':

            residuals = self.observations[i]['y'] - model_flux 

            trend = detrend.cheops_trend_func(self.observations[i]['t'], self.observations[i], residuals)
#             print('cheops trend',trend)
            model_flux += trend

        if self.observations[i]['inst_name'] == 'TESS':
            
            model_flux = model_flux

    return model_flux