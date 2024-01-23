# Provided to Eli Schwat by Christopher Cox on January 22, 2024
import numpy as np

def calc_z0(z, Cd, zL):
    """Calculate roughness length using the methodology of Andreas et al. 2010.
    Andreas, E. L., Persson, P. O. G., Grachev, A. A., Jordan, R. E., Horst, T. W., Guest, P. S., & Fairall, C. W. (2010). Parameterizing Turbulent Exchange over Sea Ice in Winter. Journal of Hydrometeorology, 11(1), 87–104. https://doi.org/10.1175/2009JHM1102.1

    Args:
        z (_type_): height of instrument (m)
        Cd (_type_): drag coefficient
        zL (_type_): stability parameter z/L
    """
    k = 0.4
    L = z / zL
    # stability function for stable state from Grachev (2007) https://doi.org/10.1007/s10546-007-9177-6 
    sma = 1 + (6.5 * zL * (1+zL)**(1/3)) / (1.3 + zL); # Psi 

    # stability function for unstable state from Paulson (1970) https://doi.org/10.1175/1520-0450(1970)009<0857:TMROWS>2.0.CO;2
    # the real is just because the equation is only valid for unstable and this is vectorized, see below
    x = np.real((1 - 16*zL)**(0.25)) # assumes gamma = 16
    smp = 2*np.log((1+x)/2) + np.log((1+x**2)/2) - 2*np.arctan(x)+np.pi/2

    # select amongst smp and sma depending on stability. just approx using thermodynamic temp
    sm = sma; 
    ii = np.argwhere(zL < 0)
    sm[ii] = smp[ii]

    # Andreas et al. (2010) eq 3.5
    z0 = z * np.exp(-(k*Cd**-0.5 + sm*(z/L)))
    
    return z0