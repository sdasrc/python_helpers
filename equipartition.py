# ## Equations and Variables needed to calculate Equipartition Parameters
# Equations from [Odea & Owen (1987)](http://adsabs.harvard.edu/doi/10.1086/165182)
# 1. Radio Luminosity:
# \begin{equation}
# L_{rad} = 1.2\times 10^{27}~D^2~S_0~\nu_0^{-\alpha}~(1+z)^{-(1+\alpha)}~(\nu_{up}^{1+\alpha}~-~\nu_{low}^{1+\alpha})~{(1+\alpha)}^{-1}~ergs~s^{-1}.
# \end{equation} <br>
# where z is the galaxy redshift, D is the luminosity distance to the source (Mpc), $S_0$ is the flux density (Jy) at a fiducial frequency $\nu_0$ (Hz; we take it to be 4.7 GHz), $\alpha$ is the spectral index $(S~\propto~\nu^\alpha)$, and $\nu_{up}$ and $\nu_{low}$ are the upper and lower frequency cutoffs (in Hz).<br>

# 2. Magnetic field at minimum pressure:
# \begin{equation}
# B_{min} = {[2\pi~(1+\eta)~C_{12}~L_{rad}~{(V\sigma)}^{-1}]}^{2/7}~G.
# \end{equation}
# <br>
# 3. Particle energy (electrons and protons) at minimum pressure:
# \begin{equation}
# E_{min} = {[V\sigma~{(2\pi)}^{-1}]}^{3/7}~{[L_{rad}~(1+\eta)~C_{12}]}^{4/7}~ergs.
# \end{equation}<br>
# where $\eta$ is the ratio of relativistic proton to relativistic electron energy, $V$ is the source volume, $C_{12}$ is a constant which depends on the spectral index and frequency cutoffs (see Pacholczyk 1970), and $\sigma$ is the volume filling factor.<br>
# 4. The minimum pressure:
# \begin{equation}
# P_{min} = {(2\pi)}^{-3/7}~\frac{7}{12}~{[L_{rad}~C_{12}~(1+\eta)~{(V\sigma)}^{-1}]}^{4/7}~dynes~{cm}^{-2}.
# \end{equation}
# <br>
# 5. The Particle Lifetime ($t_e$):<br>
# The electron lifetime at a frequency $\nu~(Hz)$ due to synchrotron losses in a magnetic field of strength $B~(G)$ and inverse Compton losses off the microwave background (with equivalent magnetic field $B_R$) is given by:

# \begin{equation}
# t_e = \frac{2.6\times10^{4}~B_{min}^{1/2}}{(B_{min}^2 + B_R^2)~{[(1+z)\nu]}^{1/2}}
# \end{equation}
# where
# \begin{equation}
# B_R = 4~(1 + z)^2 \mu G.
# \end{equation}
# We take $\nu$ = break frequency = 1.6 GHz for this calculation.

import numpy as np
from IPython.display import display, Math
class EquipartitionCalculator:
    '''
    S +/- dS      : radio flux density in Jy at ...
    v             : frequency in Hz
    a +/- da      : spectral index, S \propto v^a
    z             : redshift
    Dmpc          : luminosigmaty distance in Mpc
    breakfreq     : break frequency in Hz (default 1.6 GHz)
    vu, vl        : the upper and lower frequency cutoffs (in Hz), 
                    default :: 15 GHz and 100 MHz respectively.
    n             : ion to electron densigmaty ratio (default 1)
    Vol +/- dVol  : volume occupied by source in cm3
    sigma         : volume filling factor (default 1)
    c12           : constant defined in Pacholczyk (1970)
    '''
    def __init__(self, Dmpc, z, v, Vol, dVol, a, da, S, dS, c12, 
                 n=1, sigma=1, v_l=100e6, v_u=15e9, breakfreq=144e6,
                 verbose=False):
        self.Dmpc = Dmpc
        self.z = z
        self.v = v
        self.v_u = v_u
        self.v_l = v_l
        self.Vol = Vol
        self.dVol = dVol
        self.a = a
        self.da = da
        self.S = S
        self.dS = dS
        self.n = n
        self.sigma = sigma
        self.c12 = c12
        self.breakfreq = breakfreq
        self.verbose = verbose

    def calc_lrad(self):
        K_La = 1.2e27 * self.Dmpc * self.Dmpc * self.S
        K_v1 = self.v ** (-self.a)
        K_v2 = (1. + self.z) ** (-1. - self.a)
        K_v3 = self.v_u ** (1. + self.a) - self.v_l ** (1. + self.a)
        K_v4 = (1. + self.a) ** -1
        Lrad = K_La * K_v1 * K_v2 * K_v3 * K_v4
        dLrad_da = -(K_La * K_v1 * K_v2 * K_v3 * (1. + self.a) ** -2) \
                  + (K_La * K_v1 * K_v2 * ((self.v_u ** (1. + self.a)) * np.log(self.v_u) - (self.v_l ** (1. + self.a)) * np.log(self.v_l)) * K_v4) \
                  - (K_La * K_v1 * ((1. + self.z) ** (-1. - self.a)) * np.log(1. + self.z) * K_v3 * K_v4) \
                  - (K_La * (self.v ** (-self.a)) * np.log(self.v) * K_v2 * K_v3 * K_v4)
        dLrad_dS = 1.2e27 * self.Dmpc * self.Dmpc * K_v1 * K_v2 * K_v3 * K_v4
        dLrad = np.sqrt((dLrad_da ** 2) * (self.da ** 2) + (dLrad_dS ** 2) * (self.dS ** 2))
        return Lrad, dLrad

    def calc_Bmin(self):
        K_B = (2 * np.pi * (1. + self.n) * self.c12 * (self.sigma ** -1)) ** (2. / 7.)
        Lrad, dLrad = self.calc_lrad()
        Bmin = K_B * (Lrad ** (2. / 7.)) * (self.Vol ** (-2. / 7.))
        dB_dL = K_B * (self.Vol ** (-2. / 7.)) * (2. / 7.) * (Lrad ** (-5. / 7.))
        dB_dVol = K_B * (Lrad ** (2. / 7.)) * (-2. / 7.) * (self.Vol ** (-9. / 7.))
        dB = np.sqrt((dB_dL ** 2) * (dLrad ** 2) + (dB_dVol ** 2) * (self.dVol ** 2))
        return Bmin, dB

    def calc_Emin(self):
        K_E = (self.sigma * (2 * np.pi) ** -1) ** (3. / 7.) * ((1. + self.n) * self.c12) ** (4. / 7.)
        Lrad, dLrad = self.calc_lrad()
        Emin = K_E * self.Vol ** (3. / 7.) * Lrad ** (4. / 7.)
        dE_dVol = K_E * (3. / 7.) * self.Vol ** (-4. / 7.) * Lrad ** (4. / 7.)
        dE_dL = K_E * (4. / 7.) * self.Vol ** (3. / 7.) * Lrad ** (-3. / 7.)
        dE = np.sqrt((dE_dL ** 2) * (dLrad ** 2) + (dE_dVol ** 2) * (self.dVol ** 2))
        return Emin / self.Vol, dE / self.Vol

    def calc_Pmin(self):
        K_P = (2 * np.pi) ** (-3. / 7.) * (7. / 12.) * (self.c12 * (1 + self.n) * (self.sigma) ** -1) ** (4. / 7.)
        Lrad, dLrad = self.calc_lrad()
        Pmin = K_P * self.Vol ** (-4. / 7.) * Lrad ** (4. / 7.)
        dP_dVol = K_P * (-4. / 7.) * self.Vol ** (-11. / 7.) * Lrad ** (4. / 7.)
        dP_dL = K_P * (4. / 7.) * self.Vol ** (-4. / 7.) * Lrad ** (-3. / 7.)
        dP = np.sqrt((dP_dL ** 2) * (dLrad ** 2) + (dP_dVol ** 2) * (self.dVol ** 2))
        return Pmin, dP

    def calc_te(self):
        Br = 4.0E-6 * ((1.0 + self.z) ** 2)
        Bmin, dB = 2.77e-6, 2.77e-7
        print(Bmin)
        n3 = 2.6e4 * np.sqrt(Bmin)
        d3 = (Bmin * Bmin + Br * Br) * np.sqrt((1.0 + self.z) * self.breakfreq)
        t3 = (n3 / d3)
        t3_6 = t3 / 1.e6
        t03 = 2.6e4 / ((1. + self.z) * self.breakfreq) ** 0.5
        dt3 = t03 * dB * (0.5 * (Bmin ** (-0.5)) * ((Bmin ** 2 + Br ** 2) ** (-1)) + 2. * Bmin * (Bmin ** (0.5)) * ((Bmin ** 2 + Br ** 2) ** (-2)))
        dt3_6 = dt3 / 1.e6
        return t3_6, dt3_6
    
    def get_values(self): 
        
        Lrad, dLrad = self.calc_lrad()
        Bmin, dB = self.calc_Bmin()
        Emin, dE = self.calc_Emin()
        Pmin, dP = self.calc_Pmin()
        t3_6, dt3_6 = self.calc_te()
        
        if self.verbose:        
            txt = "\mathrm{{{2}}} = {0:.4e}~~~~\pm~~~~{1:.4e}"
            txt = txt.format(Lrad, dLrad, 'L_\mathrm{rad}~[ergs~s^{-1}]~~~~~~~~')
            display(Math(txt))

            txt = "\mathrm{{{2}}} = {0:.4e}~~~~\pm~~~~{1:.4e}"
            txt = txt.format(Bmin, dB, 'B_\mathrm{min,P}~[G]~~~~~~~~~~~~~~~')
            display(Math(txt))

            txt = "\mathrm{{{2}}} = {0:.4e}~~~~\pm~~~~{1:.4e}"
            txt = txt.format(Emin, dE, 'E_\mathrm{min,P}~[ergs]~~~~~~~~~~~')
            display(Math(txt))

            txt = "\mathrm{{{2}}} = {0:.4e}~~~~\pm~~~~{1:.4e}"
            txt = txt.format(Pmin, dP, 'P_\mathrm{min}~[dynes~cm^{-2}]~')
            display(Math(txt))

            txt = "\mathrm{{{2}}} = {0:.4e}~~~~\pm~~~~{1:.4e}"
            txt = txt.format(t3_6, dt3_6, 't_\mathrm{IC}~[Myrs]~~~~~~~~~~~~~~~~~~~~')
            display(Math(txt))
        
        return { 'L_rad' : {'val': Lrad, 'unc': dLrad, 'units' : 'ergs s^-1' },
            'B_min' : {'val': Bmin, 'unc': dB, 'units' : 'G' },
            'E_min' : {'val': Emin, 'unc': dE, 'units' : 'ergs' },
            'P_min' : {'val': Pmin, 'unc': dP, 'units' : 'dynes cm^-2' },
            'T_IC' : {'val': t3_6, 'unc': dt3_6, 'units' : 'Myrs' }  }
    