import math as ma
from math import pi
import cmath as cm
from numbers import Number
from functools import wraps
import time
from scipy.integrate import simps
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
        time0 = time.time()
        _func = func(*args, **kwargs)
        print('time:', time.time() - time0)
        return _func
    return wrapped

def save_database(regime):
    def _force(func):
        @wraps(func)
        def wrapped(self, beam, ptc, ptc_pos, force, *args, **kwargs):
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
            all_params.update({'x': ptc_pos.x,
                               'y': ptc_pos.y,
                               'z': ptc_pos.z})

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
            __force = func(self, beam, ptc, ptc_pos, force, *args, **kwargs)

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
            return [0, 0, 0]

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

        _electric_field = electric_field/numpy.linalg.norm(electric_field)

        return ma.acos(abs(numpy.dot(_electric_field, plane_normal)))

    @staticmethod
    def parallel_reflection(incident_angle, refracted_angle):
        if incident_angle == refracted_angle:
            return 0
        return (ma.tan(incident_angle - refracted_angle)
                / ma.tan(incident_angle + refracted_angle))

    @staticmethod
    def parallel_trasmission(incident_angle, refracted_angle):
        if incident_angle == refracted_angle:
            return 1
        return (2*ma.sin(refracted_angle)*ma.cos(incident_angle)
                /(ma.sin(incident_angle+refracted_angle)
                  *ma.cos(incident_angle-refracted_angle)))

    @staticmethod
    def perpendicular_reflection(incident_angle, refracted_angle):
        if incident_angle == refracted_angle:
            return 0
        return -(ma.sin(incident_angle - refracted_angle)
                 / ma.sin(incident_angle + refracted_angle))

    @staticmethod
    def perpendicular_trasmission(incident_angle, refracted_angle):
        if incident_angle == refracted_angle:
            return 1
        return (2*ma.sin(refracted_angle)*ma.cos(incident_angle)
                /(ma.sin(incident_angle+refracted_angle)))

    @classmethod
    def reflectance(cls, incident_angle, refracted_angle, crossing_angle):

        # Rs and Rp: Fresnell transmission and reflection coeffici-
        # ents for s (perpendicular) and p (paralell) polarization.
        Rs = (cls.perpendicular_reflection(incident_angle, refracted_angle))**2
        # Rs must not be greater than 1
        if Rs > 1:
            Rs = 1

        Rp = (cls.parallel_reflection(incident_angle, refracted_angle))**2
        # Rp must not be greater than 1
        if Rp > 1:
            Rp = 1

        # Reflection coefficient.
        return Rs*ma.sin(crossing_angle)**2 + Rp*ma.cos(crossing_angle)**2

    @classmethod
    def transmittance(cls, incident_angle, refracted_angle, crossing_angle):
        return (1 - cls.reflectance(incident_angle, refracted_angle,
                                    crossing_angle))

    @classmethod
    def reflected_ray(cls, incident_ray, normal, reflectance):
        if incident_ray == [0, 0, 0]:
            return [0, 0, 0]

        incident_ray_abs = numpy.linalg.norm(incident_ray)
        k0 = incident_ray/incident_ray_abs
        d0 = cls.ortonormal_ray_direction(k0, normal)
        incident_angle = cls.incident_angle(k0, n0)

        rotat = cm.exp(1j*(pi-2*incident_angle))
        ref_ray = [rotat.real*k + rotat.imag*d for k, d in zip(k0, d0)]
        ref_ray_abs = incident_ray_abs*reflectance

        return [ref_ray_abs*value for value in ref_ray]

    '''def internal_ray(self, theta, phi, n=0):
        if self.incident_ray_abs == 0:
            return [0, 0, 0]

        k0 = self.incident_ray
        thetai = self.incident_angle(theta, phi)
        thetar = self.refracted_angle(theta, phi)
        d0 = self.ortonormal_incident_ray(theta, phi)
        R = self.reflectance(theta, phi)
        # Distance travelled by a single ray before and after hit
        # sphere's surface.
        l = 2*self.radius*ma.cos(thetar)

        rotation = cm.exp(-1j*(n*(pi-2*thetar)+(thetai-thetar)))
        a_k0, a_d0 = rotation.real, rotation.imag
        int_ray = [a_k0*k + a_d0*d for k, d in zip(k0, d0)]
        abs_int_ray = (self.incident_ray_abs
                       *ma.exp(-self.absorption_coefficient*l)*(1-R)*R**n)

        return [abs_int_ray*component for component in int_ray]'''

    def outgoing_ray(self, theta, phi, incident_ray, electric_field, ray=0):
        if incident_ray == [0, 0, 0]:
            return [0, 0, 0]

        incident_ray_abs = numpy.linalg.norm(incident_ray)
        n0 = self.normal(theta, phi)
        k0 = incident_ray/incident_ray_abs
        d0 = self.ortonormal_ray_direction(k0, n0)
        incident_angle = self.incident_angle(k0, n0)
        refracted_angle = self.refracted_angle(incident_angle,
                                               self.medium_refractive_index,
                                               self.refractive_index)
        _electric_field = electric_field/numpy.linalg.norm(electric_field)
        crossing_angle = self.crossing_angle(k0, n0, _electric_field)
        reflectance = self.reflectance(incident_angle, refracted_angle,
                                       crossing_angle)
        length = 2*self.radius*ma.cos(refracted_angle)

        rotat = cm.exp(-2j*(incident_angle-refracted_angle))
        rotat *= cm.exp(-1j*ray*(pi-2*refracted_angle))

        out_ray = [rotat.real*k + rotat.imag*d for k, d in zip(k0, d0)]

        out_ray_abs = (1-reflectance)**2*ma.exp(-self.absorption_coefficient
                                                *length)
        out_ray_abs *= (reflectance*ma.exp(-self.absorption_coefficient
                                           *length))**ray

        return [out_ray_abs*value for value in out_ray]

    def Qkd(self, incident_angle, refracted_angle, reflectance):

        length = 2*self.radius*ma.cos(refracted_angle)
        internal_absorption = ma.exp(-self._absorption_coefficient*length)

        den = (1 + 2*reflectance*internal_absorption*ma.cos(2*refracted_angle)
               + reflectance**2*internal_absorption)

        if den != 0:
            Qk = (1 + reflectance*ma.cos(2*incident_angle)
                  - (1-reflectance)**2*internal_absorption
                  * (ma.cos(2*(incident_angle-refracted_angle))
                     + reflectance*internal_absorption
                     * ma.cos(2*incident_angle))/den)

            Qd = (-reflectance*ma.sin(2*incident_angle)
                  - (1-reflectance)**2*internal_absorption
                  * (ma.sin(2*(refracted_angle-incident_angle))
                     - reflectance*internal_absorption
                     * ma.sin(2*incident_angle))/den)
        else:
            Qk = (1 + reflectance*ma.cos(2*incident_angle)
                  - internal_absorption
                  * (ma.cos(2*(refracted_angle-incident_angle))
                     + reflectance*internal_absorption
                     * ma.cos(2*incident_angle)))

            Qd = (-reflectance*ma.sin(2*incident_angle)
                  - internal_absorption
                  * (ma.sin(2*(refracted_angle-incident_angle))
                     - reflectance*internal_absorption
                     * ma.sin(2*incident_angle)))

        return Qk, Qd

    def force_ray(self, theta, phi, incident_ray, poynting, electric_field):
        if incident_ray == [0, 0, 0]:
            return [0, 0, 0]

        incident_ray_abs = numpy.linalg.norm(incident_ray)
        n0 = self.normal(theta, phi)
        k0 = incident_ray/incident_ray_abs
        d0 = self.ortonormal_ray_direction(k0, n0)
        incident_angle = self.incident_angle(k0, n0)

        # Check if is a valid condition
        if incident_angle >= pi/2:
            return [0, 0, 0]

        refracted_angle = self.refracted_angle(incident_angle,
                                               self._medium_refractive_index,
                                               self._refractive_index)
        _electric_field = electric_field/numpy.linalg.norm(electric_field)
        crossing_angle = self.crossing_angle(k0, n0, _electric_field)

        reflectance = self.reflectance(incident_angle, refracted_angle,
                                       crossing_angle)
        Qk, Qd = self.Qkd(incident_angle, refracted_angle, reflectance)

        force = [Qk*k + Qd*d for k, d in zip(k0, d0)]

        force_factor = self.medium_refractive_index*poynting/speed_of_light

        return [force_factor*f for f in force]


class Force(object):
    def __init__(self):
        pass

    @classmethod
    @timing
    #@save_database('geo-opt')
    def geo_opt(cls, beam, ptc, ptc_pos, force, simps_points=101):
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
            ptc_pos (:obj:'Point'): point at which particle is placed.
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

        ptc.electric_field_direction = beam.electric_field_direction

        def integrand(theta, phi):
            """ Return each infinitesimal force contribuition on sphere
            surface as function of theta and phi at particle
            coordinates."""

            # Beam particle surface: beam coordinates point that
            # match the point at theta and phi on particle surface.
            bps = Point([ptc.radius, theta, phi], 'spherical') + ptc_pos

            # Vector parallel to the direction of a single ray.
            k0 = beam.wavenumber_direction(bps)

            # Beam's power at particle surface
            poynting = beam.intensity(bps)

            # Electric field direction
            E0 = beam._electric_field_direction

            # Force of a single ray.
            force_ray = ptc.force_ray(theta, phi, k0, poynting, E0)

            def test_msg():
                incident_ray_abs = 1
                n0 = ptc.normal(theta, phi)
                d0 = ptc.ortonormal_ray_direction(k0, n0)
                incident_angle = ptc.incident_angle(k0, n0)

                refracted_angle = ptc.refracted_angle(incident_angle,
                                                       ptc._medium_refractive_index,
                                                       ptc._refractive_index)
                crossing_angle = ptc.crossing_angle(k0, n0, E0)

                reflectance = ptc.reflectance(incident_angle, refracted_angle,
                                               crossing_angle)
                Qk, Qd = ptc.Qkd(incident_angle, refracted_angle, reflectance)

                msg = ''
                msg += 'np=' + str(ptc.refractive_index) + ';\n'
                msg += 'nm=' + str(ptc.medium_refractive_index) + ';\n'
                msg += 'Rp=' + str(ptc.radius) + ';\n'
                msg += 'partpos=List' + str(ptc_pos.cartesian()) + ';\n'
                msg += 'point=List' + str(bps.cylindrical()) + ';\n'
                msg += 'theta=' + str(theta) + ';\n'
                msg += 'phi=' + str(phi) + ';\n'
                msg += 'n0=List' + str(list(n0)) + ';\n'
                msg += 'k0=List' + str(list(k0)) + ';\n'
                msg += 'E0=List' + str(list(E0)) + ';\n'
                msg += 'beta=' + str(crossing_angle) + ';\n'
                msg += 'thetai=' + str(incident_angle) + ';\n'
                msg += 'thetar=' + str(refracted_angle)  + ';\n'
                msg += 'd0=List' + str(list(d0)) + ';\n'
                msg += 'R=' + str(reflectance) + ';\n'
                msg += 'Qkd=List' + str(list([Qk, Qd])) + ';\n'
                msg += 'intensity=' + str(poynting) + ';\n'
                msg += 'Force=List' + str(list(force_ray)) + ';\n'
                msg = msg.replace('e-', '*10^-')
                #pyperclip.copy(msg)
                #pyperclip.paste()
                print(msg)
                time.sleep(1)

            test_msg()

            if force == 'fx':
                return {'force': force_ray[0]*ma.sin(theta), 'eff_area': ma.sin(theta)}
            elif force == 'fy':
                return {'force': force_ray[1]*ma.sin(theta), 'eff_area': ma.sin(theta)}
            elif force == 'fz':
                return {'force': force_ray[2]*ma.sin(theta), 'eff_area': ma.sin(theta)}
            else:
                return {'force': 0, 'eff_area': 0}

        def integration():
            nptheta = simps_points

            theta_list = numpy.linspace(pi, 0, 1000)

            dforce_theta = []
            deff_area_theta = []

            for theta in theta_list:
                phi_list = numpy.linspace(0, 2*pi, 20)
                                       #ma.floor(nptheta*ma.sin(theta)+1))

                dforce_phi = []
                deff_area_phi = []
                for phi in phi_list:
                    values = integrand(theta, phi)
                    dforce_phi.append(values['force'])
                    deff_area_phi.append(values['eff_area'])

                dforce_theta.append(simps(dforce_phi, phi_list))
                deff_area_theta.append(simps(deff_area_phi, phi_list))

            eff_area = simps(deff_area_theta, theta_list)

            return simps(dforce_theta, theta_list)/eff_area

        return integration()


if __name__ == '__main__':
    print("Please, visit: https://github.com/arantespp/opticalforces")

    from beam import FrozenWave

    fw = FrozenWave()

    def func(z):
        if -0.1*1*1e-3 < z  and z < 0.1*1*1e-3:
            return 1
        else:
            return 0

    fw.vacuum_wavelength = 1064e-9
    fw.medium_refractive_index = 1.33
    fw.N = 1
    fw.L = 1*1e-3
    fw.R = 17.5e-6
    fw.Q = 0.95*fw.wavenumber
    fw.reference_function = func

    Rp = 17.5e-6

    ptc = SphericalParticle(radius=Rp, medium_refractive_index=1.33,
                            refractive_index=1.1*1.33)

    nm = 1.33
    np = 1.33*1.1
    ptc.medium_refractive_index = nm
    ptc.refractive_index = np

    print(Force.geo_opt(fw, ptc, Point([0, 0, 0]), 'fz'))
    '''theta = 2*pi/3
    phi = 0

    n0 = ptc.normal(theta, phi)
    k0 = [-0.6, 0, 0.8]
    E0 = [1, 0, 0]
    thetai = ptc.incident_angle(k0, n0)
    thetar = ptc.refracted_angle(thetai, nm, np)
    d0 = ptc.ortonormal_ray_direction(k0, n0)
    beta = ptc.crossing_angle(k0, n0, E0)
    reflectance = ptc.reflectance(thetai, thetar, beta)
    ref_ray = ptc.reflected_ray(k0, n0, reflectance)
    out_ray = ptc.outgoing_ray(theta, phi, k0, E0, ray=2)
    Qk, Qd = ptc.Qkd(thetai, thetar, reflectance)
    f_ray = ptc.force_ray(theta, phi, k0, 10, E0)

    print('n0: ', n0)
    print('k0: ', k0)
    print('d0: ', d0)
    print('thetai: ', thetai*180/pi)
    print('thetar: ', thetar*180/pi)
    print('beta: ', beta*180/pi)
    print('reflectance: ', reflectance)
    print('reflected ray:', ref_ray)
    print('outgoing ray:', out_ray)
    print('Qk, Qd: ', Qk, Qd)
    print('force: ', f_ray)'''

