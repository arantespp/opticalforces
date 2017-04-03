import numpy as np
from math import pi
import matplotlib.pyplot as plt
import sys
import copy
sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import GaussianBeam
from force import SphericalParticle

# ----- beam definition -----
gbeam = GaussianBeam()

gbeam.vacuum_wavelength = 0.488e-6
gbeam.medium_refractive_index = 1.33
gbeam.spot = 0.4e-6
gbeam.electric_field_direction = [0, 1, 0]

with open('beam-parameters.txt', 'w') as file:
    file.write(str(gbeam))

# ----- particle definition -----
ptc = []

particle_radius = 6e-6
particle = SphericalParticle(medium_refractive_index=1.33)
particle.radius = particle_radius
particle.refractive_index = 1.6

particle.absorption_coefficient = 0.001e6
ptc.append(copy.copy(particle))

particle.absorption_coefficient = 0.02e6
ptc.append(copy.copy(particle))

particle.absorption_coefficient = 0.5e6
ptc.append(copy.copy(particle))

# ----- particle initical position -----
z0=[]
z0.append(+4*particle_radius)
z0.append(+1*particle_radius)
z0.append(-2*particle_radius)

# ----- force -----
def style(i):
    if i == 0:
        return '--'
    elif i == 1:
        return '-.'
    elif i == 2:
        return '-'
    else:
        return '-'

plt.figure(1, figsize=(1.3*4.5, 4.5))

for index_z, z in enumerate(z0):
    #plt.subplot(len(z0), 1, index_z+1)

    for index_ptc in range(len(ptc)):
        print('z: ', index_z+1, 'ptc: ', index_ptc+1)
        x = np.linspace(-3*particle_radius, 3*particle_radius, 2**6 + 1)
        fx = [ptc[index_ptc].geo_opt_force(gbeam, x, pi, z, 'fx','total')*10e11
              if x !=0 else 0 for x in x]

        plt.plot([x/particle_radius for x in x], fx, style(index_ptc),
                 label=str(ptc[index_ptc].absorption_coefficient*1e-6))

    axes = plt.gca()
    axes.set_xlim([-3, 3])
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xlabel(r'x/$R_p$', fontsize=14)
    plt.ylabel(r'$F_x(x)\times 10^{-11}$', fontsize=14)
    if index_z == 0:
        plt.legend(fontsize=14, loc=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig('z' + str(index_z) + '-' + __file__.replace('.py', '.png'))
    plt.gcf().clear()
