from math import tan
import sys
import numpy
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cmplt
import time

sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import *
from force import *

# ========================== #

L=1e-3

fw = FrozenWave(vacuum_wavelength=1064e-9,
                medium_refractive_index=1.33,
                L=L,
                N=15)

fw.Q = 0.95*fw.wavenumber

def ref_func(z):
    if abs(z) < 0.1*L:
        return 1
    else:
        return 0

fw.reference_function = ref_func
fw.electric_field_direction = [1, 1, 0]

'''with open('beam-parameters.txt', 'w') as file:
    file.write(str(fw))

print(fw)

z = np.linspace(-L/2, L/2, 250)

plt.figure(1, figsize=(4.5*1.618, 4.5))

plt.plot([z*1e6 for z in z], [fw.intensity(Point([0, 0, z])) for z in z])
axes = plt.gca()
axes.set_xlim([-L/2*1e6, L/2*1e6])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'z($\mu$m)', fontsize=14)
plt.ylabel('I(z)', fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('perfil2D.png')
plt.show()

exit()'''

# ========================== #

ptc = SphericalParticle()
ptc.radius = 17.5e-6
ptc.medium_refractive_index = 1.33


power = 5
z = numpy.linspace(-300e-6, +300e-6, 2**power+1)

plt.figure(1, figsize=(4.5*1.618, 4.5))

#np = [1.2*1.33, 1.010*1.33, 1.005*1.33, 0.950*1.33]
np = [1.2*1.33]

for n in np:
    ptc.refractive_index = n
    force = []
    z_graph = []

    for i in range(len(z)):
        #if i >= 32:
            #break
        print(i, len(z)-1)
        if abs(z[i]) > 0.2*L:
            ff = 0
        else:
            ptc_pos = Point([0, 0, z[i]], 'cartesian')
            ff = ptc.geo_opt_force(fw, 0, 0, z[i], 'fx')

        print(z[i], ff)

        if n == 1.010*1.33:
            ff *= 5
        elif n == 1.005*1.33:
            ff *= 10

        force.append(ff)
        z_graph.append(z[i]*1e6)

    plt.plot(z_graph, force, label=n/1.33)

axes = plt.gca()
axes.set_xlim([-300, 300])
#axes.set_xlim([-150, 150])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'z($\mu$m)', fontsize=14)
plt.ylabel(r'$F_z(z$)', fontsize=14)
plt.legend(fontsize=14, loc=1)
plt.grid()
plt.tight_layout()
#plt.savefig('fw-fz-geo-opt.png')
plt.show()
