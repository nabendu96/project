import numpy as np
from scipy import optimize as op
import sys

def get_taus(filename, n_pixels, n_los):

    taus = np.empty((n_pixels, n_los), dtype=np.float64)

    with open(filename, 'rb') as f:
        for i in range(n_los): 
            n = np.fromfile(f, dtype=np.int64, count=1)[0]
            redshift = np.fromfile(f, dtype=np.float64, count=1)
            boxsize = np.fromfile(f, dtype=np.float64, count=1)
            velscale = np.fromfile(f, dtype=np.float64, count=1)
            time = np.fromfile(f, dtype=np.float64, count=1)
            xpos = np.fromfile(f, dtype=np.float64, count=1)
            ypos = np.fromfile(f, dtype=np.float64, count=1)
            dirflag = np.fromfile(f, dtype=np.int64, count=1)
            tau = np.fromfile(f, dtype=np.float64, count=n)
            taus[:,i]= tau
            temp = np.fromfile(f, dtype=np.float64, count=n)
            vpec = np.fromfile(f, dtype=np.float64, count=n)
            rho = np.fromfile(f, dtype=np.float64, count=n)
            rhoneutral = np.fromfile(f, dtype=np.float64, count=n)
            nhi = np.fromfile(f, dtype=np.float64, count=n)
            denstemp = np.fromfile(f, dtype=np.float64, count=n)
            n2 = np.fromfile(f, dtype=np.int64, count=1)
            assert(n==n2)

    return taus

specfile = sys.argv[1]
print(specfile)
n_pixels = 2048
n_los = 5000 
taus = get_taus(specfile, n_pixels, n_los)

f_measured = float(sys.argv[2])

def f(a):
    
    f_sim_mean = np.mean(np.exp(-a*taus))
    return f_measured - f_sim_mean 

r = op.bisect(f, 0.01, 100.0)

print('r=',r)

fbar = np.mean(np.exp(-r*taus))
print('fbar=', fbar)
