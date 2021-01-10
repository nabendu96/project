"""
creating the k array for the power spectrum plot

Something like this will work: python3 karr.py ./spec_z3.000_0_512.dat 512 5000 karr_512.txt

n_pixels is the number of grids 
n_los is the number of sightlines
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import sys

def get_taus(filename, n_pixels, n_los):
    #norm = 0.9179910109957352        # fiducial 
    #norm = 0.8957986327622899        # non-power law
    #norm = 0.4931288366537928        # 10*T_0
    #norm = 0.7835412796289518        # 2*T_0
    #norm = 0.6490402406402864        # 4*T_0
    #norm = 0.5760575572902173        # 6*T_0
    #norm = 0.5280126690283625        # 8*T_0
    #norm = 0.6436081830025281        # 2*slope
    #norm = 0.875788437770983         # 1.2*slope
    #norm = 0.8266538422620232        # 1.4*slope
    #norm = 0.770737346393422         # 1.6*slope
    #norm = 0.7089911633630724        # 1.8*slope
    #norm = 0.9174169704210932        # -ve power at low temperature
    #norm = 0.4924037489101563        # 10*T_0 and -ve power at low temperature
    #norm = 0.9286380805401929         # fiducial with 512 grids
    #norm = 0.524817614661789          # 10*T_0 with 512 grids
    with open(filename, 'rb') as f:
        for i in range(n_los):
            n = np.fromfile(f, dtype=np.int64, count=1)
            redshift = np.fromfile(f, dtype=np.float64, count=1)
            boxsize = np.fromfile(f, dtype=np.float64, count=1)
            velscale = np.fromfile(f, dtype=np.float64, count=1)
            time = np.fromfile(f, dtype=np.float64, count=1)
            xpos = np.fromfile(f, dtype=np.float64, count=1)
            ypos = np.fromfile(f, dtype=np.float64, count=1)
            dirflag = np.fromfile(f, dtype=np.int64, count=1)
            tau = np.fromfile(f, dtype=np.float64, count=n_pixels)
            temp = np.fromfile(f, dtype=np.float64, count=n_pixels)
            vpec = np.fromfile(f, dtype=np.float64, count=n_pixels)
            rho = np.fromfile(f, dtype=np.float64, count=n_pixels)
            rhoneutral = np.fromfile(f, dtype=np.float64, count=n_pixels)
            nhi = np.fromfile(f, dtype=np.float64, count=n_pixels)
            denstemp = np.fromfile(f, dtype=np.float64, count=n_pixels)
            n2 = np.fromfile(f, dtype=np.int64, count=n_pixels)
            assert(n==n2)
    return velscale

filename = sys.argv[1].strip()
n_pixels = int(sys.avgv[2].strip())
n_los = int(sys.avgv[3].strip())
outfile = sys.avgr[4].strip
velscale = get_taus(filename, n_pixels, n_los)

vaxis = velscale*np.arange(0, n_pixels)/n_pixels

#k array
fnew = open(outfile, 'w')
dx = vaxis[1] - vaxis[0]
karr = np.empty(int(n_pixels/2)+1, dtype=np.float64)
for m in range(int(n_pixels/2)+1):
    karr[m] = m/(n_pixels*dx)
    karr[m] = 2*np.pi*karr[m]
    fnew.write("%f \n" %karr[m])
fnew.close()
