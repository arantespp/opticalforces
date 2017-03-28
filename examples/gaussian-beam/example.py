import numpy as np
from math import pi
import matplotlib.pyplot as plt
import sys
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
particle_radius = 6e-6
ptc = SphericalParticle(medium_refractive_index=1.33)
ptc.radius = particle_radius
ptc.refractive_index = 1.6
ptc.absorption_coefficient = 0.5e6

# ----- force -----
def style(i):
    if i == 0:
        return '-'
    elif i == 1:
        return '--'
    elif i == 2:
        return '-.'
    else:
        return '-'

x0 = np.linspace(-3*particle_radius, 3*particle_radius, 2**8+1)
phi0 = pi
z0 = particle_radius

fx_total = [ptc.geo_opt_force(gbeam, x, phi0, z0, 'fx', 'total')
            for x in x0]

fx_incident = [ptc.geo_opt_force(gbeam, x, phi0, z0, 'fx', 'incident')
               for x in x0]

fx_reflection = [ptc.geo_opt_force(gbeam, x, phi0, z0, 'fx', 'reflection')
                 for x in x0]

fx_transmission = [ptc.geo_opt_force(gbeam, x, phi0, z0, 'fx', 'transmission')
                   for x in x0]

plt.figure(1, figsize=(4.5*1.618, 4.5))

x_axis = [x/particle_radius for x in x0]
plt.plot(x_axis, fx_total, style(0), label='total')
plt.plot(x_axis, fx_incident, style(1), label='incident')
plt.plot(x_axis, fx_reflection, style(2), label='reflection')
plt.plot(x_axis, fx_transmission, style(3), label='transmission')

axes = plt.gca()
axes.set_xlim([-3, 3])

plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'$x_0(\mu$m)', fontsize=14)
plt.ylabel(r'$F_x(x_0$)', fontsize=14)
plt.legend(fontsize=12, loc=1)
plt.grid()
plt.tight_layout()

#plt.savefig(title + '-' + str(ptc[j].absorption_coefficient*1e-6) + '-' + str(round(gb.omega0*1e6,2)) + '.png')
plt.show()
