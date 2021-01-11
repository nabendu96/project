import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

filename = '/scratch/snx3000/gkulkarn/L160N2048/output_final/cube_T_2048_instantanne_19.npy'
T = np.load(filename, mmap_mode='r')
filename = '/scratch/snx3000/gkulkarn/L160N2048/output_final/cube_D_2048_instantanne_19.npy'
D = np.load(filename, mmap_mode='r')
Dbar = np.mean(D) 

# Scatter plot 
fig = plt.figure(figsize=(7, 7), dpi=100)
ax = fig.add_subplot(1, 1, 1)
plt.minorticks_on()
ax.tick_params('both', which='major', length=7, width=1, direction='in', top='on', right='on')
ax.tick_params('both', which='minor', length=3, width=1, direction='in', top='on', right='on')
ax.tick_params('x', which='major', pad=8)
temp = T[10,:,:].flatten()
dens = D[10,:,:].flatten()/Dbar
plt.scatter(dens, temp, rasterized=True, s=6)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1.0e0, 1.0e4)
plt.xlim(1.0e-1, 1.0e1)
plt.xlabel(r'$\log_{10}(\Delta_\mathrm{gas})$')
plt.ylabel(r'$T$ [K]')
plt.text(0.9, 0.9, '20--512, $z=19.6$', horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
plt.savefig('trho_L20N512_z19p6_scatter.pdf', bbox_inches='tight')

# 2d histogram 
fig = plt.figure(figsize=(9, 7), dpi=100)
ax = fig.add_subplot(1, 1, 1)
plt.minorticks_on()
ax.tick_params('both', which='major', length=7, width=1, direction='in', top='on', right='on')
ax.tick_params('both', which='minor', length=3, width=1, direction='in', top='on', right='on')
ax.tick_params('x', which='major', pad=8)
temp = T[:,:,1024].flatten()
dens = D[:,:,1024].flatten()/Dbar
log10temp = np.log10(temp)
log10dens = np.log10(dens)
tbins = np.linspace(0.5,4.5,num=500)
dbins = np.linspace(-2,3,num=500)
H, xedges, yedges = np.histogram2d(log10dens, log10temp, bins=(dbins,tbins), normed=True)
H = H.T
x = (xedges[1:]+xedges[:-1])/2
y = (yedges[1:]+yedges[:-1])/2
x, y = np.meshgrid(x, y)
log10H = np.log10(H)
plt.scatter(x, y, rasterized=True, s=4, c=log10H, marker='s', vmin=-4, vmax=1)
plt.ylim(3, 5)
plt.xlim(-2, 3)
plt.yticks((3,4,5))
cb = plt.colorbar()
cb.set_label(r"$\log_{10}\mathrm{PDF}$", labelpad=10)
plt.xlabel(r'$\log_{10}\Delta_\mathrm{gas}$')
plt.ylabel(r'$\log_{10}(T/\mathrm{K})$')
plt.text(0.02, 0.95, '160--2048 (mono-freq), $z=6.6$', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
plt.savefig('trho_L160N2048_8jun_z6p6_hist.pdf', bbox_inches='tight')


