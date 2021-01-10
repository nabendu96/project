"""
Creating the power spectrum array for the power spectrum plot.
Something like this will work: python3 ps.py ./spec_z3.000_0.dat 256 5000 0.9179910109957352 Parr_0.txt
n_pixels is the number of grids
n_los is the number of sightlines
norm is set using mean flux = 0.6670 at z = 3 (https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.2067B/abstract: Table 2 of this paper has measured value of the mean flux at 2 < z < 5)
"""

import numpy as np
import sys

def get_P(filename, n_pixels, n_los, norm):
    taus = np.empty((n_pixels, n_los), dtype=np.float64)
    flux = np.empty((n_pixels, n_los), dtype=np.float64)
    dft = np.empty((n_pixels, n_los), dtype=complex)
    P = np.empty((int(n_pixels/2+1),n_los), dtype=np.float64)
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
            taus[:,i] = tau
            taus[:,i] -= taus[:,i].min()
            flux[:,i] = np.exp(-norm*taus[:,i]) 
            mean_flux = np.mean(flux[:,i], axis=0)
            flux[:,i] = (flux[:,i] - mean_flux)/mean_flux     #flux contrast
            
            #dft of flux contrast
            for k in range(n_pixels):
                s = 0
                for l in range(n_pixels):
                    s = s + flux[l,i]*np.exp(2j*np.pi*l*k/n_pixels)
                dft[k,i]=s
            vaxis = velscale*np.arange(0, n_pixels)/n_pixels
            dx = vaxis[1] - vaxis[0]
            dft = dx*dft/velscale
            
            #power spectrum using Periodogram estimate
            P[0,i] = (np.abs(dft[0,i]))**2
            for p in range(int((n_pixels/2)-1)):
                P[p+1,i] = (np.abs(dft[p+1,i])**2 + np.abs(dft[n_pixels-p-1,i])**2)
            P[int(n_pixels/2),i] = (np.abs(dft[int(n_pixels/2),i]))**2
            
            temp = np.fromfile(f, dtype=np.float64, count=n_pixels)
            vpec = np.fromfile(f, dtype=np.float64, count=n_pixels)
            rho = np.fromfile(f, dtype=np.float64, count=n_pixels)
            rhoneutral = np.fromfile(f, dtype=np.float64, count=n_pixels)
            nhi = np.fromfile(f, dtype=np.float64, count=n_pixels)
            denstemp = np.fromfile(f, dtype=np.float64, count=n_pixels)
            n2 = np.fromfile(f, dtype=np.int64, count=1)
            assert(n==n2)

    return P,velscale

filename = sys.argv[1].strip()
n_pixels = int(sys.argv[2].strip())
n_los = int(sys.argv[3].strip())
norm = float(sys.argv[3].strip())
outfile = sys.argv[4].strip()

P,velscale = get_P(filename, n_pixels, n_los, norm)

f = open(outfile, 'w')
P_ave = np.empty(int(n_pixels/2+1))
for a in range(int(n_pixels/2+1)):
    P_ave[a] = velscale*np.mean(P[a,:])
    f.write("%f \n" %P_ave[a])
f.close()
