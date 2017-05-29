from math import exp
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

def ref_func(z):
    if abs(z) < 0.2*L:
        return exp(50*z)
    else:
        return 0

fwx = FrozenWave(vacuum_wavelength=1064e-9,
                medium_refractive_index=1.33,
                L=L,
                N=15)

fwx.Q = 0.9995*fwx.wavenumber
fwx.reference_function = ref_func
fwx.electric_field_direction = [1, 0, 0]
fwx.name = 'fwx-%s' % str(fwx.electric_field_direction)

with open('beam-parameters-fwx.txt', 'w') as file:
    file.write(str(fwx))

fwy = FrozenWave(vacuum_wavelength=1064e-9,
                medium_refractive_index=1.33,
                L=L,
                N=15)

fwy.Q = 0.9995*fwy.wavenumber
fwy.reference_function = ref_func
fwy.electric_field_direction = [0, 1, 0]
fwy.name = 'fwy-%s' % str(fwy.electric_field_direction)

with open('beam-parameters-fwy.txt', 'w') as file:
    file.write(str(fwy))

# ----- particle definition
ptc = SphericalParticle()
ptc.radius = 17.5e-6
ptc.medium_refractive_index = 1.33

# ----- figure 1 - total fx to 4 differents np particles -----
z = numpy.linspace(-15e-3, +15e-3, 2**8+1)

plt.figure(0, figsize=(5*1.618, 5))

def style(i):
    if i == 0:
        return '-'
    elif i == 1:
        return '--'
    elif i == 2:
        return '-'
    else:
        return '-.'

def color(i):
    if i == 0:
        return 'blue'
    elif i == 1:
        return 'red'
    elif i == 2:
        return 'green'
    else:
        return 'magenta'

np = [0.950*1.33, 1.005*1.33, 1.010*1.33, 1.2*1.33]
#np = [1.2*1.33]

for n_index, n in enumerate(np):
    print(n_index, n)
    ptc.refractive_index = n
    ptc.name = 'SP-nrel-%s' % str(ptc.refractive_index/1.33)
    forcex = []
    forcey = []
    z_graph = []

    if n == 0.950*1.33:
        factor = 1.3
    elif n == 1.010*1.33:
        factor = 19
    elif n == 1.005*1.33:
        factor = 60
    else:
        factor = 1

    for i in range(len(z)):
        print(i, len(z)-1)
        #if abs(z[i]) > 0.2*L:
            #ffy = 0
            #ffx = 0
        #else:
        beam_pos = Point([0, 0, z[i]], 'cartesian')
        ffy = ptc.geo_opt_force(fwy, beam_pos, 'fz', 'total')
            #ffx = ptc.geo_opt_force(fwx, beam_pos, 'fz', 'total')

        #print(z[i], ffy)

        forcey.append(factor*ffy)
        #forcex.append(ffx)
        z_graph.append(z[i]*1e3)

    label = r'$n_{rel}$=%s (x%s)' % (str(n/1.33), str(factor))
    plt.plot(z_graph, forcey, style(0), color=color(n_index), label=label)
    #plt.plot(z_graph, forcex, style(n_index), label='[1, 0, 0]')

ax = plt.gca()
#ax.set_xlim([-300, 300])
'''ax.annotate('Q', fontsize=14, xy=(-43.5, 0), xytext=(-150, 1e-11),
              arrowprops=dict(facecolor='black', width=0.1, shrink=1,
                              headwidth=5),)
ax.annotate('P', fontsize=14, xy=(-50.6, 0), xytext=(-200, 1e-11),
              arrowprops=dict(facecolor='black', width=0.1, shrink=1,
                              headwidth=5),)'''
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'$z_0$ (mm)', fontsize=14)
plt.ylabel(r'$F_z(z_0$)', fontsize=14)
plt.title('Force Total')
plt.legend(fontsize=12, loc=1)
plt.grid()
plt.tight_layout()
plt.savefig('all-4-ptc-nrel.png')

# ---- 4 kind forces of all ptc nrel
'''for n_index, n in enumerate(np):
    plt.figure(n_index+1, figsize=(5*1.618, 5))

    ptc.refractive_index = n

    force_total = [ptc.geo_opt_force(fwx, Point([0, 0, z]), 'fz', 'total')
                   if abs(z) <= 0.2*L else 0 for z in z]
    plt.plot(z_graph, force_total, '-', label='total')

    force_incid = [ptc.geo_opt_force(fwx, Point([0, 0, z]), 'fz', 'incident')
                   if abs(z) <= 0.2*L else 0 for z in z]
    plt.plot(z_graph, force_incid, '-.', label='incident')

    force_reflec = [ptc.geo_opt_force(fwx, Point([0, 0, z]), 'fz', 'reflection')
                    if abs(z) <= 0.2*L else 0 for z in z]
    plt.plot(z_graph, force_reflec, '--', label='reflection')

    force_trans = [ptc.geo_opt_force(fwx, Point([0, 0, z]), 'fz', 'transmission')
                   if abs(z) <= 0.2*L else 0 for z in z]
    plt.plot(z_graph, force_trans, '-', label='transmission')

    ax = plt.gca()
    ax.set_xlim([-300, 300])
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xlabel(r'$z_0(\mu$m)', fontsize=14)
    plt.ylabel(r'$F_z(z$)', fontsize=14)
    plt.title(r'$n_{rel} = %s$' % str(ptc.refractive_index/1.33))
    plt.legend(fontsize=12, loc=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig('all-kind-force-nrel-%s.png' % str(ptc.refractive_index/1.33))
'''
plt.show()
