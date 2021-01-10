import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
"""
file=open('Parr_0.txt','r')         
P_ave=[]
for line in file:
    P_ave.append(float(line))
P_ave=np.array(P_ave)
"""

file1=open('Parr_0_512.txt','r')         
P_ave=[]
for line1 in file1:
    P_ave.append(float(line1))
P_ave=np.array(P_ave)

file2=open('Parr_2.txt','r')         
P_ave2=[]
for line2 in file2:
    P_ave2.append(float(line2))

file3=open('Parr_2_512.txt','r')         
P_ave3=[]
for line3 in file3:
    P_ave3.append(float(line3))
P_ave3=np.array(P_ave3)

"""
file4=open('Parr_33.txt','r')         
P_ave4=[]
for line4 in file4:
    P_ave4.append(float(line4))
P_ave4=np.array(P_ave4)

file5=open('Parr_34.txt','r')         
P_ave5=[]
for line5 in file5:
    P_ave5.append(float(line5))
P_ave5=np.array(P_ave5)
"""

file6=open('karr_512.txt','r')         
karr=[]
for line6 in file6:
    karr.append(float(line6))
karr=np.array(karr)

file7=open('karr.txt','r')
karr2=[]
for line7 in file7:
    karr2.append(float(line7))
karr2=np.array(karr2)

#print(karr*P_ave4/np.pi-karr*P_ave2/np.pi)

fig, ax = plt.subplots(constrained_layout=True)

plt.loglog(karr, karr*P_ave/np.pi, label=r'fiducial simulation')
plt.loglog(karr2, karr2*P_ave2/np.pi, label=r'$10T_0$, n_grid = 256')
plt.loglog(karr, karr*P_ave3/np.pi, label=r'$10T_0$, n_grid = 512')
#plt.loglog(karr, karr*P_ave4/np.pi, label=r' $1.6 \times$ slope')
#plt.loglog(karr, karr*P_ave5/np.pi, label=r'$1.8 \times$ slope')
#plt.loglog(karr,karr*P_ave1/np.pi, label=r'$2.0 \times$ slope')
#plt.xlim(karr[1],karr.max())
plt.xlabel('k [km$^{-1}$s]',size='xx-large')
plt.ylabel('$\Delta^2_\mathrm{F,1D}$',size='xx-large')

"""
x = np.array([0.002803, 0.003499, 0.004479, 0.005632, 0.00708, 0.008945, 0.0113, 0.01425, 0.01794, 0.02259, 0.02838, 0.03573, 0.04501, 0.05666, 0.07132, 0.08978, 0.113, 0.1423, 0.1792, 0.2255, 0.2839, 0.3574])
y = np.array([0.03103, 0.03522, 0.04468, 0.04717, 0.05932, 0.05596, 0.05733, 0.06204, 0.06517, 0.05688, 0.05032, 0.04715, 0.03812, 0.02728, 0.01872, 0.01121, 0.00548, 0.002266, 0.0009699, 0.0003949, 0.0002304, 0.0001749])
yerr = np.array([0.009415, 0.006892, 0.008296, 0.006015, 0.007473, 0.006969, 0.005735, 0.006429, 0.005423, 0.003983, 0.002825, 0.002449, 0.002463, 0.001632, 0.001295, 0.00106, 0.0005469, 0.0003068, 0.000172, 9.197e-05, 3.592e-05, 2.872e-05])
"""

#plotting power spectrum
#plt.loglog(karr, karr*P_ave/np.pi, label=r'Our model')
#plt.errorbar(x,y,yerr=yerr,fmt='.k',capsize=5, label=r'Walther et al. 2018')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(karr2[1], karr2.max())
#ax.set_ylim(ymin, ymax)
#plt.xlim(karr[1],karr.max())
#plt.xlabel('k [km$^{-1}$s]',size='xx-large')
#plt.ylabel('$\Delta^2_\mathrm{F,1D}$',size='xx-large')
#plt.text(10**(-2.6), 10**(-3.2), r'Nyquist wavenumber = 0.3561 km$^{-1}$s', fontsize=10)
#plt.text(10**(-2.6), 10**(-3.5), r'fundamental wavenumber = 0.0028 km$^{-1}$s', fontsize=10)
plt.legend()
#plt.savefig('flux_ps.pdf')
#plt.savefig('flux_ps.pdf', bbox_inches='tight')
#plt.show()


ax2 = ax.twiny()
#ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlim(1.0/karr2[1],1.0/karr2.max())
#ax2.set_ylim(ymin, ymax)
ax2.set_xlabel('$\lambda = 1/$k [km$/$s]',size='xx-large')
#ax2.tick_params('x', which='major', length=7, width=1)
#ax2.tick_params('x', which='minor', length=3, width=1)
        
#plt.text(10**(-2.6), 10**(-3.2), r'Nyquist wavenumber = 0.3561 km$^{-1}$s', fontsize=10)
#plt.text(10**(-2.6), 10**(-3.5), r'fundamental wavenumber = 0.0028 km$^{-1}$s', fontsize=10)
#plt.legend()
plt.savefig('ps_20_512_2.pdf', bbox_inches='tight')
plt.show()
