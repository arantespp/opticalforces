import math as ma
from math import pi
import cmath as cm
from numbers import Number
from functools import wraps
import time
from scipy.integrate import quad
import csv
import numpy
import os

from beam import Point

# Speed of light.
SPEED_OF_LIGHT = 299792458

def round_sig(num, sig=4):
    if isinstance(num, Number):
        if num < 0:
            num = -num
            return -round(num, sig-int(ma.floor(ma.log10(num)))-1)
        elif num > 0:
            return +round(num, sig-int(ma.floor(ma.log10(num)))-1)
        else:
            return num
    elif isinstance(num, list):
        return [round_sig(element) for element in num]

def timing(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        print('start:', time.strftime("%d %b %Y %H:%M:%S", time.localtime()))
        time0 = time.time()
        _func = func(*args, **kwargs)
        print('time:', time.time() - time0)
        return _func
    return wrapped

def check_database(regime):
    def force(func):
        @wraps(func)
        def wrapped(self, beam, beam_pos, force_dir, force_type):
            # Database's name
            database_name = self.name
            database_name += '-' + beam.name
            database_name += '-' + force_dir
            database_name += '-' + force_type
            database_name += '-' + regime
            database_name += '.csv'

            database_subdir = 'database'

            full_path = os.path.join(database_subdir, database_name)

            field_names = ['x', 'y', 'z', 'force']

            delimiter = ','

            _beam_pos = (str(round_sig(beam_pos.x)),
                         str(round_sig(beam_pos.y)),
                         str(round_sig(beam_pos.z)),)

            if os.path.isfile(full_path):
                with open(full_path) as database:
                    reader = csv.DictReader(database, delimiter=delimiter)
                    for row in reader:
                        if (row['x'], row['y'], row['z']) == _beam_pos:
                            return float(row['force'])
            else:
                os.makedirs(database_subdir)

            _force = func(self, beam, beam_pos, force_dir, force_type)

            with open(full_path, 'a+') as database:
                writer = csv.DictWriter(database, fieldnames=field_names)
                database.seek(0) #ensure you're at the start of the file..
                if not database.read(1):
                    writer.writeheader()
                writer.writerow({'x': _beam_pos[0],
                                 'y': _beam_pos[1],
                                 'z': _beam_pos[2],
                                 'force': _force,})

            return _force

        return wrapped

    return force


class SphericalParticle(object):
    params = ('_radius',
              '_refractive_index',
              '_medium_refractive_index',
              '_absorption_coefficient',)

    def __init__(self, **kwargs):
        self.name = 'spherical-particle'

        self._radius = None
        self._refractive_index = None
        self._medium_refractive_index = None
        self._absorption_coefficient = 0

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    def __str__(self):
        out = 'Particle parameters: \n'
        for param in self.params:
            out += '    '
            out += param + ': ' + str(self.__dict__[param]) + '\n'
        return out

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def refractive_index(self):
        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, value):
        self._refractive_index = value

    @property
    def medium_refractive_index(self):
        return self._medium_refractive_index

    @medium_refractive_index.setter
    def medium_refractive_index(self, value):
        self._medium_refractive_index = value

    @property
    def absorption_coefficient(self):
        return self._absorption_coefficient

    @absorption_coefficient.setter
    def absorption_coefficient(self, value):
        self._absorption_coefficient = value

    # --- Methods ----

    @staticmethod
    def normal(theta, phi):
        """ Vector normal to surface point where a single ray hits. """
        if theta == 0:
            return [0, 0, 1]
        elif theta == pi:
            return [0, 0, -1]
        else:
            return [ma.sin(theta)*ma.cos(phi), ma.sin(theta)*ma.sin(phi),
                    ma.cos(theta)]

    @staticmethod
    def ortonormal_ray_direction(ray_direction, normal):
        dot = numpy.dot(ray_direction, normal)
        if dot == 0:
            return normal

        d0 = [n-k for n, k in zip(normal, [dot*k for k in ray_direction])]

        if numpy.linalg.norm(d0) == 0:
            return [0, 0, 0]

        return [d/numpy.linalg.norm(d0) for d in d0]

    @staticmethod
    def incident_angle(ray_direction, normal):
        return ma.acos(-numpy.dot(ray_direction, normal))

    @staticmethod
    def refracted_angle(incident_angle, medium_refractive_index,
                        particle_refractive_index):
        return cm.asin(medium_refractive_index*ma.sin(incident_angle)
                       /particle_refractive_index).real

    @staticmethod
    def crossing_angle(ray_direction, normal, electric_field):
        """ Crossing angle between the polarization direction of the
        incident beam and the normal vector of the incident plane."""

        plane_normal = numpy.cross(ray_direction, normal)
        if numpy.linalg.norm(plane_normal) == 0:
            return 0

        div = (numpy.dot(electric_field, plane_normal)
               /numpy.linalg.norm(plane_normal))

        if abs(div) >= 1:
            return 0
        else:
            return ma.acos(abs(div))

    @staticmethod
    def parallel_reflectivity(incident_angle, refracted_angle):
        if incident_angle == 0:
            nrel = (self._refractive_index/self._medium_refractive_index)
            return (n-1)/(n+1)

        if refracted_angle == pi/2:
            return 1

        return (ma.tan(incident_angle - refracted_angle)
                / ma.tan(incident_angle + refracted_angle))**2

    @staticmethod
    def perpendicular_reflectivity(incident_angle, refracted_angle):
        if incident_angle == 0:
            nrel = (self._refractive_index/self._medium_refractive_index)
            return -(n-1)/(n+1)

        if refracted_angle == pi/2:
            return 1

        return (ma.sin(incident_angle - refracted_angle)
                / ma.sin(incident_angle + refracted_angle))**2

    @classmethod
    def reflectivity(cls, incident_angle, refracted_angle, crossing_angle):
        Rpa = cls.parallel_reflectivity(incident_angle, refracted_angle)
        Rpe = cls.perpendicular_reflectivity(incident_angle, refracted_angle)
        return Rpa*ma.sin(crossing_angle)**2 + Rpe*ma.cos(crossing_angle)**2

    @classmethod
    def parallel_trasmissivity(cls, incident_angle, refracted_angle):
        return 1 - cls.parallel_reflectivity(incident_angle, refracted_angle)

    @classmethod
    def perpendicular_trasmissivity(cls, incident_angle, refracted_angle):
        return 1 - cls.perpendicular_reflectivity(incident_angle,
                                                  refracted_angle)

    @classmethod
    def trasmissivity(cls, incident_angle, refracted_angle, crossing_angle):
        Tpa = cls.parallel_trasmissivity(incident_angle, refracted_angle)
        Tpe = cls.perpendicular_trasmissivity(incident_angle, refracted_angle)
        return Tpa*ma.sin(crossing_angle)**2 + Tpe*ma.cos(crossing_angle)**2

    def Qt(self, incident_angle, refracted_angle, reflectivity, trasmissivity,
           force_type):

        length = 2*self._radius*ma.cos(refracted_angle)
        internal_attenuation = ma.exp(-self._absorption_coefficient*length)

        if force_type == 'total':
            return (1 + reflectivity*cm.exp(-2j*incident_angle)
                    - trasmissivity**2*internal_attenuation
                    * cm.exp(-2j*(incident_angle-refracted_angle))
                    / (1+reflectivity*internal_attenuation
                       * cm.exp(+2j*refracted_angle)))
        elif force_type == 'incident':
            return 1
        elif force_type == 'reflection':
            return reflectivity*cm.exp(-2j*incident_angle)
        elif force_type == 'transmission':
            return (- trasmissivity**2*internal_attenuation
                    * cm.exp(-2j*(incident_angle-refracted_angle))
                    / (1+reflectivity*internal_attenuation
                       * cm.exp(+2j*refracted_angle)))

    @timing
    @check_database('geo-opt')
    def geo_opt_force(self, beam, beam_pos, force_dir, force_type='total'):
        """ Force that beam causes in a spherical particle in a deter-
        mined position in geometrical optics regime (particle radius is
        greater than 10 times beam's wavelenght).

        We used this paper to do the code:
        Zhang, Yanfeng, et al. "Influence of absorption on optical
        trapping force of spherical particles in a focused Gaussian
        beam." Journal of Optics A: Pure and Applied Optics 10.8
        (2008): 085001.

        """

        def dforce(theta, phi):
            # Beam particle surface: beam coordinates point that
            # match the point at theta and phi on particle surface.
            bps = Point(self.radius, theta, phi, 'spherical') - beam_pos

            rho, phi, z = bps.cylindrical()

            # Vector parallel to the direction of a single ray.
            k0 = beam.wavenumber_direction(rho, phi, z, 'cylindrical')

            n0 = self.normal(theta, phi)

            thetai = self.incident_angle(k0, n0)

            # Check if this sphere point is being illuminated
            if thetai >= pi/2:
                return 0

            thetar = self.refracted_angle(thetai,
                                          self._medium_refractive_index,
                                          self._refractive_index)

            d0 = self.ortonormal_ray_direction(k0, n0)

            E0 = beam.electric_field_direction(rho, phi, z, 'cylindrical')

            beta = self.crossing_angle(k0, n0, E0)

            reflectivity = self.reflectivity(thetai, thetar, beta)

            trasmissivity = self.trasmissivity(thetai, thetar, beta)

            Qt = self.Qt(thetai, thetar, reflectivity, trasmissivity, force_type)

            intensity = beam.intensity(rho, phi, z, 'cylindrical')

            dpower = intensity*ma.cos(thetai)

            _dforce = [(Qt.real*k + Qt.imag*d)*self._medium_refractive_index
                       * dpower/SPEED_OF_LIGHT for k, d in zip(k0, d0)]

            if force_dir == 'fx':
                return _dforce[0]
            elif force_dir == 'fy':
                return _dforce[1]
            elif force_dir == 'fz':
                return _dforce[2]
            else:
                return 0

        def quad_integration():
            epsrel = 1e-2
            epsabs = 1e-18
            limit = 999

            def theta_integral(phi):
                val, err = quad(lambda theta: dforce(theta, phi)*ma.sin(theta),
                                0, pi, epsabs=epsabs, epsrel=epsrel,
                                limit=limit)
                return val

            val, err = quad(theta_integral, 0, 2*pi, epsabs=epsabs, limit=limit,
                            epsrel=epsrel)

            return val

        #return quad_integration()
        return time.time()


if __name__ == '__main__':
    print("Please, visit: https://github.com/arantespp/opticalforces")

    from beam import VectorialFrozenWave, Point
    import pandas as pd

    def ref_func(z):
        if abs(z) < 0.35*0.1:
            return 1
        else:
            return 0

    vfw = VectorialFrozenWave()
    vfw.wavelength = 1064e-9
    vfw.medium_refractive_index = 1.33
    vfw.Q = 0.99*vfw.wavenumber
    vfw.N = 5
    vfw.L = 0.1
    vfw.reference_function = ref_func

    # ----- particle definition
    ptc = SphericalParticle()
    ptc.radius = 17.5e-6
    ptc.medium_refractive_index = 1.33
    ptc.refractive_index = 1.6

    #print(ptc.geo_opt_force(vfw, Point(0,0,0.01), 'fz', 'total'))

    a = pd.DataFrame(columns=('a', 'b'))
    a = a.append({'a':1,'b':2}, ignore_index=True)

    print(a)
