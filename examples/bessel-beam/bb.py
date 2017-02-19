from math import tan
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import *

# wavelength
wl = 500e-9
krho = 2.39e4

# ideal bessel beam
ibb = BesselBeam()
ibb.nm = 1
ibb.wavelength = wl
ibb.krho = krho

with open("beam-parameters.txt", 'w') as f:
    f.write(str(ibb))

Rmax = 2e-3
Zmax = Rmax/tan(ibb.theta)

rho = np.linspace(-Rmax*1e3, Rmax*1e3, 501)

z = np.linspace(0.05, 0.1*Zmax, 200)
z = np.append(z, np.linspace(0.1*Zmax, 0.2*Zmax, 200))
z = np.append(z, np.linspace(0.2*Zmax, 0.3*Zmax, 100))
z = np.append(z, np.linspace(0.3*Zmax, 0.4*Zmax, 50))
z = np.append(z, np.linspace(0.4*Zmax, 0.5*Zmax, 30))
z = np.append(z, np.linspace(0.5*Zmax, 0.6*Zmax, 30))
z = np.append(z, np.linspace(0.6*Zmax, 0.7*Zmax, 20))
z = np.append(z, np.linspace(0.7*Zmax, 0.8*Zmax, 20))
z = np.append(z, np.linspace(0.8*Zmax, 1.2*Zmax, 20))

width = 4.5
plt.figure(1, figsize=(width*1.618, width))

plt.plot(rho, [ibb.intensity(Point(abs(rho*1e-3), 0, 0))
	              for rho in rho], '-')
axes = plt.gca()
axes.set_xlim([-Rmax*1e3, Rmax*1e3])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'$\rho$(mm)', fontsize=14)
plt.ylabel(r'I($\rho$)', fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('apperture.png')

plt.figure(2, figsize=(width*1.618, width))

plt.plot(z, [ibb.RSI(Point(0, 0, z), Rmax) for z in z], '-')
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
