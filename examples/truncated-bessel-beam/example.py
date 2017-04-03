from math import tan
import sys
import numpy
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cmplt
import time

sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import BesselGaussBeamSuperposition, Point
from force import SphericalParticle

# ----- beam definition -----
tbb = BesselGaussBeamSuperposition()
tbb.electric_field_direction = [1, 0, 0]
tbb.medium_refractive_index = 1.33
tbb.vacuum_wavelength = 1064e-9
tbb.aperture_radius = 1e-3
tbb.zmax = 10e-3
tbb.q = 0
tbb.N = 21

with open('beam-parameters.txt', 'w') as file:
    file.write(str(tbb))

z = numpy.linspace(0, 1.25*tbb.zmax, 250)

plt.figure(1, figsize=(4.5*1.618, 4.5))

plt.plot([z*1e3 for z in z], [tbb.intensity(Point([0, 0, z])) for z in z])
axes = plt.gca()
axes.set_xlim([0, 1.25*tbb.zmax*1e3])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'z(mm)', fontsize=14)
plt.ylabel('I(z)', fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('perfil2D.png')
plt.show()

# ----- particle definition
ptc = SphericalParticle()
ptc.radius = 17.5e-6
ptc.medium_refractive_index = 1.33

# ----- figure 1 - total fz to 4 differents np particles -----
plt.figure(1, figsize=(4.5*1.618, 4.5))

np = [1.2*1.33, 1.010*1.33, 1.005*1.33, 0.950*1.33]

z = numpy.linspace(0, 1.25*tbb.zmax, 2**7 + 1)

for n in np:
    ptc.refractive_index = n
    force = []
    z_graph = []

    for i in range(len(z)):
        print(i, len(z)-1)
        ptc_pos = Point([0, 0, z[i]], 'cartesian')
        ff = ptc.geo_opt_force(tbb, 0, 0, -z[i], 'fz', 'total')

        print(z[i], ff)

        if n == 1.010*1.33:
            ff *= 5
        elif n == 1.005*1.33:
            ff *= 10

        force.append(ff)
        z_graph.append(z[i]*1e3)

    plt.plot(z_graph, force, label=n/1.33)

axes = plt.gca()
axes.set_xlim([0, 1.25*tbb.zmax*1e3])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'z(mm)', fontsize=14)
plt.ylabel(r'$F_z(z$)', fontsize=14)
plt.legend(fontsize=14, loc=1)
plt.grid()
plt.tight_layout()
plt.savefig('tbb-fz-geo-opt.png')
plt.show()
