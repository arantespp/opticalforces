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
        self.name = 'gaussian-beam'

        self.params = {
            '_beta': pi/4,
            '_nm': 1,
            '_wavelength': 488e-9/1.33,
            '_omega0': 0.2e-6,
            '_P': 1e-3
        }

        self._nm = 1
        self.nm = 1
        self._wavelength = 488e-9/1.33
        self.wavelength = 488e-9/1.33
        self._omega0 = 0.2e-6
        self.omega0 = 0.2e-6
        self._P = 1e-3
        self.P = 1e-3
        self._k = 2*ma.pi*self._nm/self._wavelength
        self.k = self._k


    def psi(self, pt):

        wl = self._wavelength
        omega0 = self._omega0
        k = self._k
        P = self._P
        nm = self._nm

        E0 = ma.sqrt(2*P/(pi*omega0**2*nm*c))
        E0 = 1

        omega = omega0*ma.sqrt((1+((wl*pt.z)/(pi*omega0**2))**2))

        r = pt.z*(1 + ( (pi*omega0**2)/(wl*pt.z) )**2 )

        delta_phi = ma.atan(wl*pt.z/(pi*omega0**2))

        if pt.z != 0:
            Phi = k*(pt.z + pt.rho**2/(2*r)) - delta_phi
        else:
            Phi = 0

        return (E0*(omega0/omega)*ma.exp(-(pt.rho/omega)**2)
                *cm.exp(1j*Phi))

# ===========================
gb = gaussian()

#print(gb.wavelength)
#print(gb.nm)
#print(gb.k)

#exit()

Rp=6e-6

ptc = []

particle = SphericalParticle(nm=1.33)
particle.Rp = Rp
particle.np = 1.6
particle.alphap = 0.001e6
ptc.append(particle)

particle = SphericalParticle(nm=1.33)
particle.Rp = Rp
particle.np = 1.6
particle.alphap = 0.02e6
ptc.append(particle)

particle = SphericalParticle(nm=1.33)
particle.Rp = Rp
particle.np = 1.6
particle.alphap = 0.5e6
ptc.append(particle)

z=[]
z.append(-4*Rp)
z.append(-1*Rp)
z.append(+2*Rp)

#print(Point(-1.54740140e-05, -1.139921e-06, -6.78348481e-07))
#print(gb.psi(Point(-1.54740140e-05, -1.139921e-06, -6.78348481e-07)))
#exit()
# ===========================
numberPoints = 2**4 + 1
x_initial = -3*Rp
x_final = 3*Rp

plt.figure(1)

def style(i):
    if i == 0:
        return '-'
    elif i == 1:
        return '--'
    else:
        return '-.'

def lbl(i):
    if i == 0:
        return r'$\alpha$ = 0.001'
    elif i == 1:
        return r'$\alpha$ = 0.02'
    else:
        return r'$\alpha$ = 0.5'

for i in range(len(z)):
    plt.subplot(len(z), 1, i+1)

    for j in range(len(ptc)):
        print('i: ', i+1, 'j: ', j+1)
        # calculate first initial forces
        x = [x_initial, 0, x_final]
        fx = [Force._geo_opt(gb, ptc[j], Point(x,0,z[i]))[0] for x in x]

        # get forces on x range if them have alread been calculated
        def match(table, key_colnames):
            all_params = {}

            # Beam's parameters
            for param, value in gb.__dict__.items():
                if param[0] == '_':
                    all_params.update({param: value})

            # Particle's parameters
            for param, value in ptc[j].__dict__.items():
                if param[0] == '_':
                    all_params.update({param: value})

            pt = {'z':z[i]}

            for key in key_colnames:
                if key == 'fx' or key == 'fy' or key == 'fz':
                    continue

                if key != 'x' and key != 'y' and key != 'z':
                    if round_sig(table[key][0]) != round_sig(all_params[key]):
                        return False
                elif key == 'z':
                    if round_sig(table[key][0]) != round_sig(z[i]):
                        return False
                else:
                    pt[key] = round_sig(table[key][0])

            if (pt['x'] >= x_initial and pt['x'] <= x_final):
                return True
            else:
                return False

        table = Table.read(gb.name + '-geo-opt.fits')

        tableg = table.group_by(table.colnames)

        table_match = tableg.groups.filter(match)

        t = Table(names=('x', 'fx'))

        for data in table_match:
            t.add_row([data['x'], data['fx']])

        t = t.group_by('x')

        x = list(t['x'])
        fx = [Force._geo_opt(gb, ptc[j], Point(x, 0, z[i]))[0] for x in x]

        # add points in x to lenght x reaches numberPoints
        while len(x) < numberPoints:
            # find max delta index
            max_delta = 0
            i_max_delta = 0
            for k in range(len(x)-1):
                if x[k+1] - x[k] > max_delta:
                    max_delta = x[k+1] - x[k]
                    i_max_delta = k
            # insert a number in x at max interval distance
            x.insert(i_max_delta+1, (x[i_max_delta]+x[i_max_delta+1])/2)
            fx = [Force._geo_opt(gb, ptc[j], Point(x, 0, z[i]))[0] for x in x]

        plt.plot([x/Rp for x in x], fx, style(j), label=lbl(j))

    plt.grid()

plt.legend(loc=1)
plt.show()


'''rho_max = 10*gb.params['omega0']
z_max = 2e-6

rho = np.linspace(-rho_max, rho_max, 50)
z = np.linspace(0, z_max, 50)

plt.figure(1)

plt.subplot(211)
plt.plot(rho*10e6, [gb.intensity(Point(rho,0,0,'cilin')) for rho in rho])
plt.grid()

plt.subplot(212)
plt.plot(z*10e6, [gb.intensity(Point(0,0,z)) for z in z])
plt.grid()

fig = plt.figure()
ax = fig.gca(projection='3d')

inty = [[gb.intensity(Point(rho,0,z,'cilin')) for rho in rho] for z in z]
inty_max = max(max(inty))
rho = [[rhoi*10e6 for rhoi in rho] for rhoj in rho]
z = [[zj*10e6 for zi in z] for zj in z]

ax.plot_surface(rho, z, inty, rstride=9, cstride=8, alpha=1)
#cset = ax.contourf(rho, z, inty, zdir='z', offset=0, cmap=cmplt.coolwarm)
#cset = ax.contourf(rho, z, inty, zdir='x', offset=-rho_max*10e6, cmap=cmplt.coolwarm)
#cset = ax.contourf(rho, z, inty, zdir='y', offset=z_max*10e6, cmap=cmplt.coolwarm)

ax.set_xlabel('rho')
#ax.set_xlim(-rho_max*10e6, rho_max*10e6)
ax.set_ylabel('z')
#ax.set_ylim(0, z_max*10e6)
ax.set_zlabel('I')
#ax.set_zlim(0, inty_max)

plt.show()'''

'''z = np.linspace(0.05, 1.25*Zmax, 51)
rho = np.linspace(-Rmax*1e3, Rmax*1e3, 21)

INTY = [[ibb.intensity(Point(rho*1e-3, 0, z, 'cilin')) for rho in rho] 
    for z in z]
RHO, Z = np.meshgrid(rho, z)

plt.figure(1, figsize=(width*1.618, width))
axes = plt.gca(projection='3d')

axes.plot_surface(RHO, Z, INTY)

axes.set_xlabel('rho')
#ax.set_xlim(-rho_max*10e6, rho_max*10e6)
axes.set_ylabel('z')
#ax.set_ylim(0, z_max*10e6)
axes.set_zlabel('I')
#ax.set_zlim(0, inty_max)'''