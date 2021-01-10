import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

def get_taus(filename, n_pixels, n_los):
    #norm = 0.9179910109957352
    #norm = 0.8957986327622899
    #norm  = 0.4931288366537928      # 108T_0
    #norm = 0.7835412796289518       # 2*T_0 
    #norm = 0.6490402406402864       # 4*T_0
    #norm = 0.5760575572902173       # 6*T_0
    #norm = 0.5280126690283625       # 8*T_0
    #norm = 0.6436081830025281       # 2*slope
    #norm = 0.875788437770983         # 1.2*slope
    #norm = 0.8266538422620232        # 1.4*slope
    #norm = 0.770737346393422         # 1.6*slope
    #norm = 0.7089911633630724        # 1.8*slope
    #norm = 0.9174169704210932
    #norm = 0.4924037489101563
    #print(norm)
    #norm = 0.9286380805401929
    #norm = 0.524817614661789
    norm = 1.0886294626531978
    taus = np.empty((n_pixels, n_los), dtype=np.float64)
    flux = np.empty((n_pixels, n_los), dtype=np.float64)
    dft = np.empty((n_pixels, n_los), dtype=complex)
    P = np.empty((int(n_pixels/2+1),n_los), dtype=np.float64)

    with open(filename, 'rb') as f:
        for i in range(n_los): 
            n = np.fromfile(f, dtype=np.int64, count=1)
            #print(n)
            redshift = np.fromfile(f, dtype=np.float64, count=1)
            #print(redshift)
            boxsize = np.fromfile(f, dtype=np.float64, count=1)
            velscale = np.fromfile(f, dtype=np.float64, count=1)
            time = np.fromfile(f, dtype=np.float64, count=1)
            xpos = np.fromfile(f, dtype=np.float64, count=1)
            ypos = np.fromfile(f, dtype=np.float64, count=1)
            dirflag = np.fromfile(f, dtype=np.int64, count=1)
            tau = np.fromfile(f, dtype=np.float64, count=1024)
            taus[:,i] = tau
            taus[:,i] -= taus[:,i].min()
            flux[:,i] = np.exp(-norm*taus[:,i]) 
            mean_flux = np.mean(flux[:,i], axis=0)
            flux[:,i] = (flux[:,i] - mean_flux)/mean_flux
            for k in range(n_pixels):
                s = 0
                for l in range(n_pixels):
                    s = s + flux[l,i]*np.exp(2j*np.pi*l*k/n_pixels)
                dft[k,i]=s
            vaxis = velscale*np.arange(0,1024.)/1024.
            dx = vaxis[1] - vaxis[0]
            dft = dx*dft/velscale
            
            #power spectrum using Periodogram estimat
            P[0,i] = (np.abs(dft[0,i]))**2
            for p in range(int((n_pixels/2)-1)):
                P[p+1,i] = (np.abs(dft[p+1,i])**2 + np.abs(dft[n_pixels-p-1,i])**2)
            P[int(n_pixels/2),i] = (np.abs(dft[int(n_pixels/2),i]))**2
            
            temp = np.fromfile(f, dtype=np.float64, count=1024)
            vpec = np.fromfile(f, dtype=np.float64, count=1024)
            rho = np.fromfile(f, dtype=np.float64, count=1024)
            rhoneutral = np.fromfile(f, dtype=np.float64, count=1024)
            nhi = np.fromfile(f, dtype=np.float64, count=1024)
            denstemp = np.fromfile(f, dtype=np.float64, count=1024)
            n2 = np.fromfile(f, dtype=np.int64, count=1)
            print(n,n2)
            assert(n==n2)

    return P,velscale

P,velscale = get_taus('./spec_z3.000_0_1024.dat', 1024, 5000)
#P1,velscale = get_taus('./spec_z3.000_1.dat', 256, 5000,1)

f = open('Parr_2_1024.txt', 'w')
n_pixels = 1024
P_ave = np.empty(int(n_pixels/2+1))
for a in range(int(n_pixels/2+1)):
    P_ave[a] = velscale*np.mean(P[a,:])
    f.write("%f \n" %P_ave[a])
f.close()
#P_ave = P_ave*velscale

"""
P_ave1 = np.empty(int(n_pixels/2+1))
for b in range(int(n_pixels/2+1)):
    P_ave1[b] = np.mean(P1[b,:])
P_ave1 = P_ave1*velscale
"""

#print(P_ave)


vaxis = velscale*np.arange(0,1024.)/1024.

#k array
fnew = open('karr3.txt', 'w')
dx = vaxis[1] - vaxis[0]
karr = np.empty(int(n_pixels/2)+1, dtype=np.float64)
for m in range(int(n_pixels/2)+1):
    karr[m] = m/(n_pixels*dx)
    karr[m] = 2*np.pi*karr[m]
    fnew.write("%f \n" %karr[m])
fnew.close()
#karr=2*np.pi*karr


"""
fn = 1/(2*dx)
kn = 2*np.pi*fn
kf = 4*np.pi*fn/n_pixels
"""

"""
plt.loglog(karr, karr*P_ave/np.pi, label=r'ps')
plt.loglog(karr, karr*P_ave1/np.pi, label=r'modified ps')
plt.xlim(karr.min(),karr.max())
plt.xlabel('k [km$^{-1}$s]',size='xx-large')
plt.ylabel('$\Delta^2_\mathrm{F,1D}$',size='xx-large')
#plt.text(10**(-2.6), 10**(-3.2), r'Nyquist wavenumber = 0.3561 km$^{-1}$s', fontsize=10)
#plt.text(10**(-2.6), 10**(-3.5), r'fundamental wavenumber = 0.0028 km$^{-1}$s', fontsize=10)
plt.legend()
plt.savefig('ps1.pdf', bbox_inches='tight')
plt.show()
"""


"""
class spectrum():

    def __init__(self, z, specfile, n_los=5000, n_pixels=512):

        self.taus = get_taus(specfile, n_pixels, n_los)
        self.n_los = n_los
        self.n_pixels = n_pixels

        return

    def rsightline(self, seed):

        # Bug: assumes n_pixels = 512 or 1024 or 2048 

        if self.n_pixels == 512: 
        
            np.random.seed(seed)
            l1 = np.random.randint(n_los)
            tau1 = self.taus[:,l1]

            l2 = np.random.randint(n_los)
            pidx = np.random.randint(n_pixels)
            tau2 = self.taus[:,l2]
            np.roll(tau2, pidx)

            l3 = np.random.randint(n_los)
            pidx = np.random.randint(n_pixels)
            tau3 = self.taus[:,l3]
            np.roll(tau3, pidx)

            tau = np.concatenate((tau1, tau2, tau3[:256]))

        if self.n_pixels == 1024: 

            np.random.seed(seed)
            l1 = np.random.randint(n_los)
            tau1 = self.taus[:,l1]

            l2 = np.random.randint(n_los)
            pidx = np.random.randint(n_pixels)
            tau2 = self.taus[:,l2]
            np.roll(tau2, pidx)

            tau = np.concatenate((tau1, tau2[:256]))

        if self.n_pixels == 2048: 

            np.random.seed(seed)
            l1 = np.random.randint(n_los)
            tau = self.taus[:640,l1]

        return tau 
    

    def tau_eff(self, n=2000):

        # Bug: assumes n_pixels = 512 or 1024.

        if self.n_pixels == 512: 

            taueffs = []
            for i in range(n):
                l1 = np.random.randint(self.n_los)
                tau1 = self.taus[:,l1]

                l2 = np.random.randint(self.n_los)
                pidx = np.random.randint(self.n_pixels)
                tau2 = self.taus[:,l2]
                np.roll(tau2, pidx)

                l3 = np.random.randint(self.n_los)
                pidx = np.random.randint(self.n_pixels)
                tau3 = self.taus[:,l3]
                np.roll(tau3, pidx)

                tau = np.concatenate((tau1, tau2, tau3[:256]))
                mean_transmission = np.mean(np.exp(-tau)) 
                taueffs.append(-np.log(mean_transmission))

            self.taueffs = taueffs

        if self.n_pixels == 1024: 

            taueffs = []
            for i in range(n):
                l1 = np.random.randint(self.n_los)
                tau1 = self.taus[:,l1]

                l2 = np.random.randint(self.n_los)
                pidx = np.random.randint(self.n_pixels)
                tau2 = self.taus[:,l2]
                np.roll(tau2, pidx)

                tau = np.concatenate((tau1, tau2[:256]))
                mean_transmission = np.mean(np.exp(-tau)) 
                taueffs.append(-np.log(mean_transmission))

            self.taueffs = taueffs

        if self.n_pixels == 2048: 

            taueffs = []
            for i in range(n):
                l1 = np.random.randint(self.n_los)
                tau = self.taus[:640,l1]
                mean_transmission = np.mean(np.exp(-tau)) 
                taueffs.append(-np.log(mean_transmission))

            self.taueffs = taueffs
            
        return

    def cdftau(self, taueffs):

        bs = np.linspace(1.0, 8.0, 1000)
        h, b = np.histogram(taueffs, bins=bs, density=True)
        dx = b[1]-b[0]
        h = np.cumsum(h)*dx
        h = np.concatenate((np.array([0.0]),h))

        return b, h 
        
        
    def tau_cdf(self):

        if not hasattr(self, 'taueffs'):
            self.tau_eff() 

        self.cdf =  self.cdftau(self.taueffs)

        return 
        
    
    def cdf_with_spread(self):

        nbins = 1000
        ncdfs = 50
        nlos = 25 
        cdfs = np.empty((nbins,ncdfs), dtype=np.float64)
        for k in range(ncdfs):
            self.tau_eff(n=nlos)
            b, h = self.cdftau(self.taueffs)
            cdfs[:,k]=h

        self.b = b 
        self.cdfs_coeval=cdfs

        return
"""
