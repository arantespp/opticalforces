from math import tan
import sys
import numpy
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cmplt
import time

sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import FrozenWave, Point
from particle import SphericalParticle

# ----- beam definition -----

L=50e-3

fw = FrozenWave(vacuum_wavelength=1064e-9,
                medium_refractive_index=1.33,
                L=L,
                N=15)

fw.Q = 0.9995*fw.wavenumber

def ref_func(z):
    if abs(z) < 0.1*L:
        return 1
    else:
        return 0

fw.reference_function = ref_func
fw.electric_field_direction = [1, 0, 0]

with open('beam-parameters.txt', 'w') as file:
    file.write(str(fw))

# ----- plot 2D -----
z = numpy.linspace(-L/2, L/2, 251)

plt.figure(1, figsize=(5*1.618, 5))

plt.plot([z*1e3 for z in z], [fw.intensity(Point([0, 0, z])) for z in z],
         label='Frozen-wave')
plt.plot([z*1e3 for z in z], [fw.reference_function(z) for z in z],
         label='Reference')
plt.legend(fontsize=12, loc=1)
ax = plt.gca()
ax.set_xlim([-L/2*1e3, L/2*1e3])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'z(mm)', fontsize=14)
plt.ylabel('I(z)', fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('perfil2D.png')
plt.show()

# ----- plot 3D -----
rho = numpy.linspace(-250e-6, 250e-6, 151)
RHO, Z = numpy.meshgrid(rho, z)

def inty(rho, z):
    return fw.intensity(Point([rho, 0, z], 'cylindrical'))

vinty = numpy.vectorize(inty)

INTY = vinty(RHO, Z)

plt.figure(2, figsize=(5*1.618, 5))
ax = plt.gca(projection='3d')

RHO = [rho/10**-6 for rho in RHO]
Z = [z/10**-3 for z in Z]
#INTY = [value/10**INTY_order for value in INTY]
ax.plot_surface(RHO, Z, INTY, rcount=1000, ccount=1000, alpha=1, cmap=cmplt.coolwarm)
#cset = ax.contourf(RHO, Z, INTY, zdir='z', offset=0, cmap=cmplt.coolwarm)
#cset = ax.contourf(RHO, Z, INTY, zdir='x', offset=-Rmax, cmap=cmplt.coolwarm)
#cset = ax.contourf(RHO, Z, INTY, zdir='y', offset=Zmax, cmap=cmplt.coolwarm)

ax.set_xlabel(r'$\rho$ ($\mu$m)', fontsize=14)
ax.set_xlim(-250, 250)
ax.set_ylabel(r'z (mm)', fontsize=14)
ax.set_ylim(-25, 25)
zlabel = 'I(z)'
ax.set_zlabel(zlabel, fontsize=14)
#ax.set_zlim(0, inty_max)

ax.view_init(elev=35, azim=-35)

plt.savefig('perfil3D.png')
plt.show()

# ----- plot 3D - II -----
plt.figure(3, figsize=(5*1.618, 5))

ax = plt.gca(projection='3d')

ax.plot_surface(RHO, Z, INTY, rcount=1000, ccount=1000, alpha=0.5)
cset = ax.contourf(RHO, Z, INTY, zdir='z', offset=-.3, cmap=cmplt.gist_heat)
cset = ax.contourf(RHO, Z, INTY, zdir='x', offset=-250, cmap=cmplt.gist_heat)
cset = ax.contourf(RHO, Z, INTY, zdir='y', offset=25, cmap=cmplt.gist_heat)

ax.set_xlabel(r'$\rho$ (mm)', fontsize=14)
ax.set_xlim(-250, 250)
ax.set_ylabel(r'z ($\mu$m)', fontsize=14)
ax.set_ylim(-25, 25)
zlabel = 'I(z)'
ax.set_zlabel(zlabel, fontsize=14)
ax.set_zlim(-.3, 1.3)

ax.view_init(elev=30, azim=-35)

plt.savefig('perfil3D-II.png')
plt.show()
