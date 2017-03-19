import math as ma
from math import pi
import cmath as cm
from numbers import Number
from functools import wraps
import time
from scipy.integrate import quad
from astropy.table import Table
import numpy

from beam import Point

import pyperclip

# Speed of light.
speed_of_light = 299792458

def round_sig(num, sig=4):
    if num < 0:
        num = -num
        return -round(num, sig-int(ma.floor(ma.log10(num)))-1)
    elif num > 0:
        return +round(num, sig-int(ma.floor(ma.log10(num)))-1)
    # num == 0
    else:
        return num

def timing(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        print('start: ', time.strftime("%d %b %Y %H:%M:%S", time.localtime()))
        time0 = time.time()
        _func = func(*args, **kwargs)
        print('end:', time.strftime("%d %b %Y %H:%M:%S", time.localtime()))
        print('time:', time.time() - time0)
        return _func
    return wrapped

def check_database(regime):
    def _force(func):
        @wraps(func)
        def wrapped(self, beam, ptc, beam_pos, force, *args, **kwargs):
            # Database's name
            database_name = beam.name + '-' + force + '-' + regime + '.fits'
            all_params = {}

            # Beam's parameters
            for param in beam.params:
                if isinstance(getattr(beam, param), Number) is True:
                    all_params.update({param: getattr(beam, param)})

            # Particle's parameters
            for param in ptc.params:
                if isinstance(getattr(ptc, param), Number) is True:
                    all_params.update({param: getattr(ptc, param)})

            # Particle position's parameters
            all_params.update({'x': beam_pos.x,
                               'y': beam_pos.y,
                               'z': beam_pos.z})

            # Round 'all_params' variables
            for param, value in all_params.items():
                all_params[param] = round_sig(value)

            # Params without forces. We need them to filter table.
            all_params_WF = all_params

            # Force's parameters
            all_params.update({'force': None})

            # Verify if exist a database and if currently params
            # have already been calculated for this beam. if there,
            # is no dabatase, create one.
            try:
                db = Table.read(database_name)

                # Match function used to filter database.
                def match(table, key_colnames):
                    for key in key_colnames:
                        if key == 'force':
                            continue

                        if table[key][0] != all_params_WF[key]:
                            return False

                    return True

                # If exist some value on table, search for match
                if len(db) != 0:

                    # sort by column's names except forces
                    key_names = db.colnames
                    key_names.remove('force')
                    db = db.group_by([key for key in key_names])

                    # Verify if currently forces have already been
                    # calculated.
                    db_match = db.groups.filter(match)

                    # If length db_match is greater than zero,
                    # means that at least one point have already
                    # been calculated.
                    if len(db_match) > 0:
                        return db_match['force'][0]

            except:
                # database create
                db_names = [name for name in all_params]
                db = Table(names=db_names)
                db.write(database_name)

            # Force (:obj:'Point')
            __force = func(self, beam, ptc, beam_pos, force, *args, **kwargs)

            all_params['force'] = __force

            data_rows = [all_params[key] for key in db.colnames]

            # Add 'all_params' in a last row on database
            db.add_row(data_rows)

            # Save new database
            db.write(database_name, overwrite=True)

            return __force

        return wrapped

    return _force


class SphericalParticle(object):
    params = ('_radius',
              '_refractive_index',
              '_medium_refractive_index',
              '_absorption_coefficient',)

    def __init__(self, **kwargs):
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
        plane_normal /= numpy.linalg.norm(plane_normal)

        return ma.acos(abs(numpy.dot(electric_field, plane_normal)))

    @staticmethod
    def parallel_reflectivity(incident_angle, refracted_angle):
        if incident_angle == refracted_angle:
            return 0
        if refracted_angle == pi/2:
            return 1
        return (ma.tan(incident_angle - refracted_angle)
                / ma.tan(incident_angle + refracted_angle))**2

    @staticmethod
    def perpendicular_reflectivity(incident_angle, refracted_angle):
        if incident_angle == refracted_angle:
            return 0
        if refracted_angle == pi/2:
            return 1
        return (ma.sin(incident_angle - refracted_angle)
                / ma.sin(incident_angle + refracted_angle))**2

    @classmethod
    def reflectivity(cls, incident_angle, refracted_angle, crossing_angle):
        Rpa = cls.parallel_reflectivity(incident_angle, refracted_angle)
        Rpe = cls.perpendicular_reflectivity(incident_angle, refracted_angle)
        return Rpa*ma.cos(crossing_angle)**2 + Rpe*ma.sin(crossing_angle)**2

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
        return Tpa*ma.cos(crossing_angle)**2 + Tpe*ma.sin(crossing_angle)**2

    def Qkd(self, incident_angle, refracted_angle, reflectivity, trasmissivity):

        length = 2*self._radius*ma.cos(refracted_angle)
        internal_attenuation = ma.exp(-self._absorption_coefficient*length)

        den = (1 + 2*reflectivity*internal_attenuation*ma.cos(2*refracted_angle)
               + (reflectivity*internal_attenuation)**2)

        if den != 0:
            Qk = (1 + reflectivity*ma.cos(2*incident_angle)
                  - trasmissivity**2*internal_attenuation
                  * (ma.cos(2*(incident_angle-refracted_angle))
                     + reflectivity*internal_attenuation
                     * ma.cos(2*incident_angle))/den)

            Qd = (0 + reflectivity*ma.sin(2*incident_angle)
                  - trasmissivity**2*internal_attenuation
                  * (ma.sin(2*(incident_angle-refracted_angle))
                     + reflectivity*internal_attenuation
                     * ma.sin(2*incident_angle))/den)
        else:
            Qk = (1 + reflectivity*ma.cos(2*incident_angle)
                  - internal_attenuation
                  * (ma.cos(2*(incident_angle-refracted_angle))
                     + reflectivity*internal_attenuation
                     * ma.cos(2*incident_angle)))

            Qd = (0 + reflectivity*ma.sin(2*incident_angle)
                  - internal_attenuation
                  * (ma.sin(2*(incident_angle-refracted_angle))
                     + reflectivity*internal_attenuation
                     * ma.sin(2*incident_angle)))

        return Qk, -Qd

    def radiation_pressure_ray(self, theta, phi, incident_ray, poynting,
            electric_field):
        if incident_ray == [0, 0, 0]:
            return [0, 0, 0]

        n0 = self.normal(theta, phi)
        incident_angle = self.incident_angle(incident_ray, n0)

        # Check if this sphere point is being illuminated
        if incident_angle >= pi/2:
            return [0, 0, 0]

        d0 = self.ortonormal_ray_direction(incident_ray, n0)
        refracted_angle = self.refracted_angle(incident_angle,
                                               self._medium_refractive_index,
                                               self._refractive_index)
        crossing_angle = self.crossing_angle(incident_ray, n0, electric_field)
        #crossing_angle = 0
        reflectivity = self.reflectivity(incident_angle, refracted_angle,
                                       crossing_angle)
        trasmissivity = self.trasmissivity(incident_angle, refracted_angle,
                                           crossing_angle)
        Qk, Qd = self.Qkd(incident_angle, refracted_angle, reflectivity,
                          trasmissivity)

        rad_press_ray = [Qk*k + Qd*d for k, d in zip(incident_ray, d0)]

        factor = self._medium_refractive_index/speed_of_light
        #factor = 1

        return [factor*rpr*poynting*ma.cos(incident_angle*0)
                for rpr in rad_press_ray]


class Force(object):
    def __init__(self):
        pass

    @classmethod
    @timing
    #@check_database('new-geo-opt-minus-kds')
    def geo_opt(cls, beam, ptc, beam_pos, force, epsrel=1e-1, limit=9999):
        """ Force that beam causes in a spherical particle in a deter-
        mined position in geometrical optics regime (particle radius is
        greater than 10 times beam's wavelenght).

        We used this paper to do the code:
        Zhang, Yanfeng, et al. "Influence of absorption on optical
        trapping force of spherical particles in a focused Gaussian
        beam." Journal of Optics A: Pure and Applied Optics 10.8
        (2008): 085001.

        Args:
            ptc (:obj:'Particle'): spherical particle.
            beam_pos (:obj:'Point'): point at which particle is placed.
                Our reference O: (0,0,0) is the beam's aperture
                center.

        Returns:
            A 'Point' containing the force vector.

        """

        # assert about required particle parameters
        assert isinstance(ptc.radius, Number), \
            ('Particle param radius not defined')

        assert isinstance(ptc.refractive_index, Number), \
            ('Particle param refractive_index not defined')

        assert isinstance(ptc.absorption_coefficient, Number), \
            ('Particle param absorption_coefficient not defined')

        # assert about geometric optics force approximation condition
        assert ptc.radius >= 9.9 * beam.wavelength, \
            ('Particle radius length less than 10 * wavelength')

        def radiation_pressure(theta, phi):
            """ Return each infinitesimal force contribuition on sphere
            surface as function of theta and phi at particle
            coordinates."""

            # Beam particle surface: beam coordinates point that
            # match the point at theta and phi on particle surface.
            bps = Point([ptc.radius, theta, phi], 'spherical')

            # Vector parallel to the direction of a single ray.
            k0 = beam.wavenumber_direction(bps - beam_pos)

            # Beam's power at particle surface
            poynting = beam.intensity(bps - beam_pos)

            # Electric field direction
            E0 = beam._electric_field_direction

            # Force of a single ray.
            radiation_pressure_ray = ptc.radiation_pressure_ray(theta, phi, k0,
                                                                poynting, E0)

            def test_msg():
                incident_ray_abs = 1
                n0 = ptc.normal(theta, phi)
                d0 = ptc.ortonormal_ray_direction(k0, n0)
                incident_angle = ptc.incident_angle(k0, n0)

                refracted_angle = ptc.refracted_angle(incident_angle,
                                                       ptc._medium_refractive_index,
                                                       ptc._refractive_index)
                crossing_angle = ptc.crossing_angle(k0, n0, E0)

                Rpa = ptc.parallel_reflectivity(incident_angle, refracted_angle)
                Rpe = ptc.perpendicular_reflectivity(incident_angle, refracted_angle)
                reflectivity = ptc.reflectivity(incident_angle, refracted_angle,
                                               crossing_angle)
                Tpa = ptc.parallel_trasmissivity(incident_angle, refracted_angle)
                Tpe = ptc.perpendicular_trasmissivity(incident_angle, refracted_angle)
                trasmissivity = ptc.trasmissivity(incident_angle, refracted_angle,
                                                  crossing_angle)
                Qk, Qd = ptc.Qkd(incident_angle, refracted_angle, reflectivity, trasmissivity)

                msg = ''
                msg += 'np=' + str(ptc.refractive_index) + ';\n'
                msg += 'nm=' + str(ptc.medium_refractive_index) + ';\n'
                msg += 'Rp=' + str(ptc.radius) + ';\n'
                msg += 'alpha=' + str(ptc.absorption_coefficient) + ';\n'
                msg += 'beampos=List' + str(beam_pos.cylindrical()) + ';\n'
                msg += 'point=List' + str((bps-beam_pos).cartesian()) + ';\n'
                msg += 'theta=' + str(theta) + ';\n'
                msg += 'phi=' + str(phi) + ';\n'
                msg += 'n0=List' + str(list(n0)) + ';\n'
                msg += 'k0=List' + str(list(k0)) + ';\n'
                msg += 'E0=List' + str(list(E0)) + ';\n'
                msg += 'beta=' + str(crossing_angle) + ';\n'
                msg += 'thetai=' + str(incident_angle) + ';\n'
                msg += 'thetar=' + str(refracted_angle)  + ';\n'
                msg += 'd0=List' + str(list(d0)) + ';\n'
                msg += 'Rpa=' + str(Rpa) + ';\n'
                msg += 'Rpe=' + str(Rpe) + ';\n'
                msg += 'R=' + str(reflectivity) + ';\n'
                msg += 'Tpa=' + str(Tpa) + ';\n'
                msg += 'Tpe=' + str(Tpe) + ';\n'
                msg += 'T=' + str(trasmissivity) + ';\n'
                msg += 'Qkd=List' + str(list([Qk, Qd])) + ';\n'
                msg += 'intensity=' + str(poynting) + ';\n'
                msg += 'radPressure=List' + str(list(radiation_pressure_ray)) + ';\n'
                msg = msg.replace('e-', '*10^-')
                pyperclip.copy(msg)
                pyperclip.paste()

            #test_msg()

            if force == 'fx':
                return radiation_pressure_ray[0]*ma.sin(theta)
            elif force == 'fy':
                return radiation_pressure_ray[1]*ma.sin(theta)
            elif force == 'fz':
                return radiation_pressure_ray[2]*ma.sin(theta)
            else:
                return 0

        def quad_integration():
            def theta_integral(phi):
                value, err = quad(lambda theta: radiation_pressure(theta, phi),
                                0, pi, epsabs=0, epsrel=epsrel, limit=limit)
                return value

            value, err = quad(theta_integral, 0, 2*pi, epsabs=0, epsrel=epsrel,
                            limit=limit)

            return value

        return quad_integration()


if __name__ == '__main__':
    print("Please, visit: https://github.com/arantespp/opticalforces")
