from math import tan
import sys
import numpy as np
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

'''fw = FrozenWave(wavelength=1064e-9*1.33,
				nm=1.33,
				L=L,
				N=15,
				beta=0)

fw.Q = 0.95*fw.k

def ref_func(z):
	if 0.4*L <= z and z <= 0.6*L:
		return 1
	else:
		return 0

fw.ref_func = ref_func'''

fw = FrozenWave(wavelength=1064e-9,
				nm=1.33,
				L=L,
				N=20,
				beta=0)

fw.Q = 7.7274e6

def ref_func(z):
	if 0.25*L <= z and z <= 0.5*L:
		return 1
	elif 0.75*L <= z and z <= 0.9*L:
		return 2
	else:
		return 0

fw.ref_func = ref_func

ptc = SphericalParticle(Rp=10*fw.wavelength, nm=1.33, np=0.8*1.33)

ptc_pos = Point(10e-6, -pi/2, 0.4*L, 'cilin')

theta = pi/180*108
phi = pi/180*0

thetai = ptc.incident_angle([0, 0, 1], theta, phi)
print(thetai*180/pi)
print(ptc.refracted_angle(thetai)*180/pi)
print(1.33*fw.intensity(ptc_pos+Point(ptc.Rp, theta, phi, 'spher'))/c)
print(ptc.Qkd([0, 0, 1], theta, phi))
print('d0: ' + str(ptc.d0([0, 0, 1], theta, phi)))

with open("beam-parameters.txt", 'w') as f:
    f.write(str(fw))

exit()

# plot intensity 2D
z = np.linspace(0*L, 1*L, 250)

plt.figure(1, figsize=(4.5*1.618, 4.5))

plt.plot([z*1e6 for z in z], [fw.intensity(Point(0, 0, z)) for z in z])
axes = plt.gca()
axes.set_xlim([0*L*1e6, 1*L*1e6])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'z($\mu$m)', fontsize=14)
plt.ylabel('I(z)', fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig('perfil2D.png')
#plt.show()

# ========================== #

ptc = SphericalParticle(Rp=10*fw.wavelength, nm=1.33, np=1.2*1.33)

rho = np.linspace(0, 100e-6, 2**3+1)

plt.figure(2, figsize=(4.5*1.618, 4.5))

#np = [1.2*1.33, 1.010*1.33, 1.005*1.33, 0.950*1.33]
np = [1.5*1.33, 0.75*1.33]

for n in np:
	ptc.np = n
	force = []

	for i in range(len(rho)):
		t0 = time.time()
		#if z[i] < 0.3*L or z[i] > 0.7*L:
		#	ff = 0
		#else:
		ff = Force._geo_opt(fw, ptc, Point(rho[i], 0, 0.4*L)).x

		#if n == 1.010*1.33:
	#		ff *= 5
	#	elif n == 1.005*1.33:
	#		ff *= 10
		
		force.append(ff)
		print(i, len(rho)-1, time.time() - t0)

	plt.plot([rho*1e6 for rho in rho], force, label=n/1.33)

	#fw.beta = pi/4
	#plt.plot([z*1e6-500 for z in z], 
	#	     [Force._geo_opt(fw, ptc, Point(0, 0, z))[2]
#		     for z in z])

	#fw.beta = pi/2
	#plt.plot([z*1e6-500 for z in z], 
#		     [Force._geo_opt(fw, ptc, Point(0, 0, z))[2]
#		     for z in z])

axes = plt.gca()
#axes.set_xlim([-300, 300])
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel(r'\rho($\mu$m)', fontsize=14)
plt.ylabel(r'I($\rho$)', fontsize=14)
plt.legend(fontsize=14, loc=1)
plt.grid()
plt.tight_layout()
#plt.savefig('perfil2D.png')
plt.show()
