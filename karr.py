"""
Creating the k array for the power spectrum plot. For a given grid number you have to create only one k array.

Something like this will work: python3 karr.py ./spec_z3.000_0_512.dat 512 5000 karr_512.txt

n_pixels is the number of grids 
n_los is the number of sightlines
"""

import numpy as np
import sys

def get_velscale(filename, n_pixels, n_los):
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
            n2 = np.fromfile(f, dtype=np.int64, count=1)
            assert(n==n2)
    return velscale

filename = sys.argv[1].strip()
n_pixels = int(sys.argv[2].strip())
n_los = int(sys.argv[3].strip())
outfile = sys.argv[4].strip()

velscale = get_velscale(filename, n_pixels, n_los)

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
