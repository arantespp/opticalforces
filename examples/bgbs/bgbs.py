from math import tan
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cmplt

sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import *

wl = 632.8e-9
k = 9.93e6
krho = 4.07e4
R = 3.5e-3

# bessel gauss beam superpositon
bgbs = BesselGaussBeamSuperposition()
bgbs.wavelength = wl
bgbs.k = k
bgbs.krho = krho
bgbs.R = R
bgbs.N = 23
bgbs.q = 0

with open("beam-parameters.txt", 'w') as f:
    f.write(str(bgbs))

Rmax = bgbs.R
Zmax = bgbs.z_max

rho = np.linspace(-1.25*Rmax*1e3, 1.25*Rmax*1e3, 501)
z = np.linspace(0, 1.25*Zmax, 250)

heigth = 4.5

# plot real apperture
plt.figure(1, figsize=(heigth*1.618, heigth))

plt.plot(rho, [bgbs.psi(Point(abs(rho*1e-3), 0, 0)).real
	              for rho in rho], '-')
axes = plt.gca()
axes.set_xlim([-1.25*Rmax*1e3, 1.25*Rmax*1e3])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'$\rho$(mm)', fontsize=14)
plt.ylabel(r'$\mathbb{Re}\{\Psi(\rho)\}$', fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('real-apperture.png')

# plot intensity 2D
plt.figure(2, figsize=(heigth*1.618, heigth))

plt.plot(z, [bgbs.intensity(Point(0, 0, z)) for z in z], '-')
plt.axvline(x=Zmax, color='r', linestyle='--', linewidth=0.75, label=r'$Z_{max}$')
axes = plt.gca()
axes.set_xlim([0, 1.25*Zmax])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel('z(m)', fontsize=14)
plt.ylabel('I(z)', fontsize=14)
plt.legend(fontsize=14, loc=1)
plt.grid()
plt.tight_layout()
plt.savefig('perfil2D.png')

# plot RS intensity
plt.figure(3, figsize=(heigth*1.618, heigth))

z = np.linspace(0.125*Zmax, 1.25*Zmax, 800)

plt.plot(z, [bgbs.RSI(Point(0, 0, z), Rmax) for z in z], '-')
plt.axvline(x=Zmax, color='r', linestyle='--', linewidth=0.75, label=r'$Z_{max}$')
axes = plt.gca()
axes.set_xlim([0, 1.25*Zmax])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel('z(m)', fontsize=14)
plt.ylabel('I(z)', fontsize=14)
plt.legend(fontsize=14, loc=1)
plt.grid()
plt.tight_layout()
plt.savefig('RS-perfil2D.png')

# plot 3D intensity
plt.figure(4, figsize=(heigth*1.618, heigth))
axes = plt.gca(projection='3d')

Rmax = 0.25*bgbs.R*1e3
Zmax = 1.25*bgbs.z_max

rho = np.linspace(-Rmax, Rmax, 101)
z = np.linspace(0, Zmax, 100)

RHO, Z = np.meshgrid(rho, z)

def inty(rho, z):
	return bgbs.intensity(Point(rho*1e-3, 0, z, 'cilin'))

vinty = np.vectorize(inty)

INTY = vinty(RHO, Z)

axes.plot_surface(RHO, Z, INTY, rcount=1000, ccount=1000, alpha=0.9)
#cset = axes.contourf(RHO, Z, INTY, zdir='z', offset=0, cmap=cmplt.coolwarm)
#cset = axes.contourf(RHO, Z, INTY, zdir='x', offset=-Rmax, cmap=cmplt.coolwarm)
#cset = axes.contourf(RHO, Z, INTY, zdir='y', offset=Zmax, cmap=cmplt.coolwarm)

axes.set_xlabel(r'$\rho$(cm)', fontsize=16)
#ax.set_xlim(-rho_max*10e6, rho_max*10e6)
axes.set_ylabel('z(m)', fontsize=16)
#ax.set_ylim(0, z_max*10e6)
axes.set_zlabel(r'I($\rho$, z)', fontsize=16)
#ax.set_zlim(0, inty_max)

axes.view_init(elev=35, azim=-35)

plt.savefig('perfil3D.png')