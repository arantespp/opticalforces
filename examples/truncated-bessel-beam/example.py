from math import tan
import sys
import numpy
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cmplt
import time
import copy
import warnings

from astropy.modeling import models, fitting

sys.path.insert(0, '/home/arantespp/Dropbox/Mestrado/opticalforces/opticalforces')

from beam import BesselGaussBeamSuperposition, Point
from particle import SphericalParticle

# ===== User variables =====
figure_heigth = 5

npx0 = 2**7 + 1
npz0 = 2**7 + 1

field_directions = [[1, 0, 0]]
axicons_angles = [6]

nrel = [0.95, 1.01, 1.2]
radius = [18e-6]
abs_coeff = [0*0.5e6]

# ===== beam definition =====
beams = []
for params in list(itertools.product(field_directions, axicons_angles)):
    tbb = BesselGaussBeamSuperposition(medium_refractive_index=1.33,
                                       vacuum_wavelength=1064e-9)
    tbb.aperture_radius = 1e-3
    tbb.q = 0
    tbb.N = 21
    tbb.name = 'tbb-E0-%s-axic-%s' % (str(params[0]), str(params[1]))
    tbb.name = tbb.name.replace(' ', '')
    tbb.electric_field_direction = params[0]
    tbb.axicon_angle_degree = params[1]

    with open(tbb.name + '-parameters.txt', 'w') as file:
        file.write(str(tbb))

    beams.append(copy.copy(tbb))

    zmax = 4/3*tbb.zmax
    z = numpy.linspace(0, zmax, 250)

    # 2D plot
    print('Creating perfil 2D graphics...')
    plt.figure(1, figsize=(figure_heigth*1.618, figure_heigth))
    plt.plot([z*1e3 for z in z], [tbb.intensity(Point([0, 0, z])) for z in z])
    axes = plt.gca()
    axes.set_xlim([0, 4/3*tbb.zmax*1e3])
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xlabel(r'z(mm)', fontsize=14)
    plt.ylabel(r'$|\Psi(z)|^2$', fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.axvline(x=tbb.zmax*1e3, color='r', linestyle='--', linewidth=0.75)
    plt.savefig(tbb.name + '-perfil2D.png')
    plt.clf()

    # 3D plot
    print('Creating perfil 3D graphics...')
    rhomax = 20*2.4/tbb.transversal_wavenumber
    rho = numpy.linspace(-rhomax, +rhomax, 151)
    RHO, Z = numpy.meshgrid(rho, z)
    def inty(rho, z):
        return tbb.intensity(Point([rho, 0, z], 'cylindrical'))
    vinty = numpy.vectorize(inty)
    INTY = vinty(RHO, Z)
    plt.figure(1, figsize=(figure_heigth*1.618, figure_heigth))
    ax = plt.gca(projection='3d')
    RHO = [rho*1e6 for rho in RHO]
    Z = [z*1e3 for z in Z]
    #INTY = [value/10**INTY_order for value in INTY]
    ax.plot_surface(RHO, Z, INTY, rcount=1000, ccount=1000, alpha=1)
    #cset = ax.contourf(RHO, Z, INTY, zdir='z', offset=-.3, cmap=cmplt.gist_heat)
    #cset = ax.contourf(RHO, Z, INTY, zdir='x', offset=-rhomax*1e6, cmap=cmplt.gist_heat)
    #cset = ax.contourf(RHO, Z, INTY, zdir='y', offset=zmax*1e3, cmap=cmplt.gist_heat)
    ax.set_xlabel(r'$\rho$ ($\mu$m)', fontsize=14)
    ax.set_ylabel(r'z ($\mu$m)', fontsize=14)
    ax.set_zlabel(r'$|\Psi(z, \rho)|^2$', fontsize=14)
    ax.set_xlim(-rhomax*1e6, rhomax*1e6)
    ax.set_ylim(0, zmax*1e3)
    #ax.set_zlim(0, inty_max)
    ax.view_init(elev=35, azim=-35)
    plt.savefig(tbb.name + '-perfil3D.png')
    plt.clf()

# ===== particle definition =====
print('Creating particles...')

ptcs = []
for params in list(itertools.product(nrel, radius, abs_coeff)):
    ptc = SphericalParticle(medium_refractive_index=1.33)
    ptc.refractive_index = ptc.medium_refractive_index*params[0]
    ptc.radius = params[1]
    ptc.absorption_coefficient = params[2]

    ptc.name = 'SP-nrel-%s-R-%s-alpha-%s' % (str(params[0]), str(params[1]),
                                             str(params[2]))

    ptcs.append(copy.copy(ptc))

# ===== calculating fz forces with rho = 0 =====
print('Calculating fz forces with rho = 0...')

counter = 0
for beam in beams:
    for ptc in ptcs:
        counter += 1
        print(counter, len(ptcs)*len(beams))

        z0 = numpy.linspace(-4/3*beam.zmax, 0, npz0)

        total = [ptc.geo_opt_force(beam, Point([0, 0, z]), 'fz', 'total')
                 for z in z0]

        incid = [ptc.geo_opt_force(beam, Point([0, 0, z]), 'fz', 'incident')
                 for z in z0]

        reflec = [ptc.geo_opt_force(beam, Point([0, 0, z]), 'fz', 'reflection')
                  for z in z0]

        trans = [ptc.geo_opt_force(beam, Point([0, 0, z]), 'fz', 'transmission')
                 for z in z0]

        plt.figure(1, figsize=(figure_heigth*1.618, figure_heigth))

        plt.plot([z*1e3 for z in z0], total, '-', label='total', linewidth=2)
        plt.plot([z*1e3 for z in z0], incid, '--', label='incident')
        plt.plot([z*1e3 for z in z0], reflec, '-.', label='reflection')
        plt.plot([z*1e3 for z in z0], trans, '-', label='transmission')
        axes = plt.gca()
        axes.set_xlim([-4/3*beam.zmax*1e3, 0])
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.xlabel(r'$z_0$(mm)', fontsize=14)
        plt.ylabel(r'$F_z(z_0)$', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.axvline(x=-beam.zmax*1e3, color='k', linestyle='--', linewidth=0.75,
                    label=r'$Z_{max}$')
        plt.legend(fontsize=14, loc=2)
        plt.savefig('%s-%s-fz_z0-2D.png' % (beam.name, ptc.name))
        plt.clf()

# ===== calculating fz forces in x with z = zmax =====
print('Calculating fz forces in x with z = zmax...')

counter = 0
for beam in beams:
    for ptc in ptcs:
        counter += 1
        print(counter, len(ptcs)*len(beams))

        #x0_max = 10*ptc.radius
        x0_max = 3*ptc.radius
        x0 = numpy.linspace(-x0_max, x0_max, npx0)

        total = [ptc.geo_opt_force(beam, Point([abs(x), 0, beam.zmax]), 'fz', 'total')
                 for x in x0]

        incid = [ptc.geo_opt_force(beam, Point([abs(x), 0, beam.zmax]), 'fz', 'incident')
                 for x in x0]

        reflec = [ptc.geo_opt_force(beam, Point([abs(x), 0, beam.zmax]), 'fz', 'reflection')
                  for x in x0]

        trans = [ptc.geo_opt_force(beam, Point([abs(x), 0, beam.zmax]), 'fz', 'transmission')
                 for x in x0]

        plt.figure(1, figsize=(figure_heigth*1.618, figure_heigth))

        plt.plot([x*1e6 for x in x0], total, '-', label='total', linewidth=2)
        plt.plot([z*1e6 for z in x0], incid, '--', label='incident')
        plt.plot([z*1e6 for z in x0], reflec, '-.', label='reflection')
        plt.plot([z*1e6 for z in x0], trans, '-', label='transmission')
        axes = plt.gca()
        axes.set_xlim([-x0_max*1e6, x0_max*1e6])
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.xlabel(r'$x_0$($\mu$m)', fontsize=14)
        plt.ylabel(r'$F_z(z_0)$', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.legend(fontsize=14, loc=2)
        plt.savefig('%s-%s-fz_x0-2D.png' % (beam.name, ptc.name))
        plt.clf()

# ===== calculating fx forces in x with z = zmax =====
print('Calculating fx forces in x with z = zmax...')

counter = 0
for beam in beams:
    for ptc in ptcs:
        counter += 1
        print(counter, len(ptcs)*len(beams))

        #x0_max = 10*ptc.radius
        x0_max = 3*ptc.radius
        x0 = numpy.linspace(-x0_max, x0_max, npx0)

        total = [ptc.geo_opt_force(beam, Point([x, 0, beam.zmax]), 'fx', 'total')
                 for x in x0]

        incid = [ptc.geo_opt_force(beam, Point([x, 0, beam.zmax]), 'fx', 'incident')
                 for x in x0]

        reflec = [ptc.geo_opt_force(beam, Point([x, 0, beam.zmax]), 'fx', 'reflection')
                  for x in x0]

        trans = [ptc.geo_opt_force(beam, Point([x, 0, beam.zmax]), 'fx', 'transmission')
                 for x in x0]

        plt.figure(1, figsize=(figure_heigth*1.618, figure_heigth))

        plt.plot([x*1e6 for x in x0], total, '-', label='total', linewidth=1)
        plt.plot([x*1e6 for x in x0], incid, '--', label='incident')
        plt.plot([x*1e6 for x in x0], reflec, '-.', label='reflection')
        plt.plot([x*1e6 for x in x0], trans, '-', label='transmission')
        axes = plt.gca()
        axes.set_xlim([-x0_max*1e6, x0_max*1e6])
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.xlabel(r'$x_0$($\mu$m)', fontsize=14)
        plt.ylabel(r'$F_x(x_0)$', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.legend(fontsize=14, loc=2)
        plt.savefig('%s-%s-fx_x0-2D.png' % (beam.name, ptc.name))
        plt.clf()

# ===== calculating fz forces 3D =====
'''print('Calculating fz forces 3D...')

counter = 0
for beam in beams:
    for ptc in ptcs:
        counter += 1
        print(counter, len(ptcs)*len(beams))

        z0 = numpy.linspace(-4/3*beam.zmax, 0, npz0)

        #x0_max = 10*ptc.radius
        x0_max = 3*ptc.radius
        x0 = numpy.linspace(-x0_max, x0_max, npx0)

        X0, Z0 = numpy.meshgrid(x0, z0)

        def fz(x0, z0):
            return ptc.geo_opt_force(beam, Point([abs(x0), 0, z0]), 'fz', 'total')

        vfz = numpy.vectorize(fz)

        FZ = vfz(X0, Z0)

        plt.figure(1, figsize=(figure_heigth*1.618, figure_heigth))
        ax = plt.gca(projection='3d')

        X0 = [x*1e6 for x in X0]
        Z0 = [z*1e3 for z in Z0]
        #INTY = [value/10**INTY_order for value in INTY]
        ax.plot_surface(X0, Z0, FZ, rcount=1000, ccount=1000, alpha=1, cmap=cmplt.coolwarm)
        #cset = ax.contourf(X0, Z0, FZ, zdir='z', offset=0, cmap=cmplt.gist_heat)
        #cset = ax.contourf(X0, Z0, FZ, zdir='x', offset=-Rmax, cmap=cmplt.gist_heat)
        #cset = ax.contourf(X0, Z0, FZ, zdir='y', offset=Zmax, cmap=cmplt.gist_heat)
        ax.set_xlabel(r'$x_0$ ($\mu$ m)', fontsize=14)
        ax.set_ylabel(r'$z_0$ (mm)', fontsize=14)
        ax.set_zlabel(r'$F_z(z_0, x_0)$', fontsize=14)
        ax.set_xlim(-x0_max*1e6, x0_max*1e6)
        ax.set_ylim(-4/3*beam.zmax*1e3, 0)
        ax.view_init(elev=35, azim=-35)
        plt.savefig('%s-%s-fz-3D.png' % (beam.name, ptc.name))
        plt.clf()'''
