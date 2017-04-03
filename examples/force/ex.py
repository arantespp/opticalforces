import math as ma
import cmath as cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm as cmplt
import time
from astropy.table import Table
import sys

sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import *
from force import *
# ===========================

# Speed of light.
c = 299792458
# Vacuum permittivity.
eps0 = 8.854187817e-12
pi = ma.pi

# ===========================

class gaussian(PlaneWave):

    def __init__(self):
        PlaneWave.__init__(self)
        self.name = 'gaussian-beam-zhang-E0y'

        self.params = {
            '_medium_refractive_index': 1,
            '_wavelength': 488e-9/1.33,
            '_omega0': 0.4e-6,
            '_P': 1e-3
        }

        self._medium_refractive_index = 1
        self.medium_refractive_index = 1
        self._wavelength = 488e-9/1.33
        self.wavelength = 488e-9/1.33
        self._omega0 = 0.4e-6
        self.omega0 = 0.4e-6
        self._P = 1e-3
        self.P = 1e-3
        self._wavenumber = 2*ma.pi*self._medium_refractive_index/self._wavelength
        self.wavenumber = self._wavenumber


    def psi(self, pt):

        wl = self._wavelength
        omega0 = self._omega0
        k = self._wavenumber
        P = self._P
        nm = self._medium_refractive_index

        #E0 = ma.sqrt(2*P*1.33/(pi*c))
        E0 = 1

        omega = omega0*ma.sqrt((1+((wl*pt.z)/(pi*omega0**2))**2))

        if pt.z != 0:
            r = pt.z*(1 + ( (pi*omega0**2)/(wl*pt.z) )**2 )
        else:
            r = 0

        delta_phi = ma.atan(wl*pt.z/(pi*omega0**2))

        if pt.z != 0:
            Phi = k*(pt.z + pt.rho**2/(2*r)) - delta_phi
        else:
            Phi = 0

        return (E0*(omega0/omega)*ma.exp(-(pt.rho/omega)**2)
                *cm.exp(1j*Phi))

    def wavenumber_direction(self, pt):
        wl = self._wavelength
        omega0 = self._omega0
        k = self._wavenumber
        P = self._P
        nm = self._medium_refractive_index

        if pt.z != 0:
            r = pt.z*(1 + ( (pi*omega0**2)/(wl*pt.z) )**2 )
        else:
            r = 0

        kx = k*pt.x/r
        ky = k*pt.y/r
        kz = (k*(1 - (2/(k*omega0)**2)*(1/(1 + ((pt.z*wl)/(pi*omega0**2))**2))
                 - (pt.x**2+pt.y**2)/(2*r**2)*(1-(pi*omega0**2/(wl*pt.z))**2)))

        k0 = [kx, ky, kz]

        k0 = k0/np.linalg.norm(k0)

        return k0


# ===========================
gb = gaussian()

#print(gb.wavelength)
#print(gb.nm)
#print(gb.k)
w0 = 0.4e-6
gb = GaussianBeam()
gb.vacuum_wavelength = 0.488e-6
gb.medium_refractive_index = 1.33
gb.waist_radius = w0
gb.electric_field_direction = [0, 1, 0]


#print(gb2.wavenumber_direction(Point([3e-4, 3e-5, 12e-4])))
#print(gb.wavenumber_direction_1(Point([3e-4, 3e-5, 12e-4])))

#exit()

Rp=6e-6

ptc = []

particle = SphericalParticle(medium_refractive_index=1.33)
particle.radius = Rp
particle.refractive_index = 1.6
particle.absorption_coefficient = 0.01e6
#ptc.append(particle)

particle = SphericalParticle(medium_refractive_index=1.33)
particle.radius = Rp
particle.refractive_index = 1.6
particle.absorption_coefficient = 0.1e6
#ptc.append(particle)

particle = SphericalParticle(medium_refractive_index=1.33)
particle.radius = Rp
particle.refractive_index = 1.6
particle.absorption_coefficient = 0.5e6
ptc.append(particle)

z=[]
#z.append(+4*Rp)
z.append(+1*Rp)
#z.append(-2*Rp)

# ===========================
numberPoints = 2**3 + 1
#numberPoints = 25+1
x_initial = -3*Rp
x_final = 3*Rp

def style(i):
    if i == 0:
        return '-'
    elif i == 1:
        return '--'
    elif i == 2:
        return '-.'
    else:
        return '-.-'

def lbl(i):
    if i == 0:
        return r'$\alpha$ = 0.001'
    elif i == 1:
        return r'$\alpha$ = 0.06'
    else:
        return r'$\alpha$ = 0.5'

plt.figure(1)

for i in range(len(z)):
    plt.subplot(len(z), 1, i+1)

    for j in range(len(ptc)):
        print('i: ', i+1, 'j: ', j+1)
        x = np.linspace(x_initial, x_final, numberPoints)

        fx = [ptc[j].geo_opt_force(gb, x, pi, z[i], 'fx', 'total') if x !=0 else 0 for x in x]
        plt.plot([x/Rp for x in x], fx, style(0), label='total')

        fx = [ptc[j].geo_opt_force(gb, x, pi, z[i], 'fx', 'incident') if x !=0 else 0 for x in x]
        plt.plot([x/Rp for x in x], fx, style(1), label='incident')

        fx = [ptc[j].geo_opt_force(gb, x, pi, z[i], 'fx', 'reflection') if x !=0 else 0 for x in x]
        plt.plot([x/Rp for x in x], fx, style(2), label='reflection')

        fx = [ptc[j].geo_opt_force(gb, x, pi, z[i], 'fx', 'transmission') if x !=0 else 0 for x in x]
        plt.plot([x/Rp for x in x], fx, style(3), label='transmission')

    plt.grid()

plt.legend(loc=1)

'''plt.figure(2)

for i in range(len(z)):
    plt.subplot(len(z), 1, i+1)

    for j in range(len(ptc)):
        print('i: ', i+1, 'j: ', j+1)
        x = np.linspace(x_initial, x_final, numberPoints)
        fx = [Force.geo_opt(gb2, ptc[j], Point([x,0,z[i]]), 'fx') if x !=0 else 0 for x in x]

        plt.plot([x/Rp for x in x], fx, style(j), label=lbl(j))

    plt.grid()

plt.legend(loc=1)'''
#title='asdasd'
#msg = ''
#msg += r'$\beta: 0$; '
#msg += r'$\alpha$:' + str(0.5) + ' '
#msg += r'$\omega_0$:' + str(0.4) + ' '
#msg += 'E=[1, 0, 0]'
#plt.title(msg)
#plt.savefig(title + '-' + str(ptc[j].absorption_coefficient*1e-6) + '-' + str(round(gb.omega0*1e6,2)) + '.png')
plt.show()
