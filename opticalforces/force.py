import math as ma
from math import pi
import cmath as cm
from numbers import Number
from scipy.integrate import dblquad
from scipy.integrate import simps
from astropy.table import Table
import numpy as np
import time

from beam import Point
from plots import *

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
        t0 = time.time()
        F = func(*args, **kwargs)
        print('time:', time.time() - t0)
        return F
    return wrapped

def save_database(regime):
    def force(func):
        @wraps(func)
        def wrapped(self, beam, ptc, ptc_pos, *args, **kwargs):
            # Database's name
            database_name = beam.name + '-' + regime + '.fits'
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
            all_params.update({'fx': None,
                               'fy': None,
                               'fz': None})

            # Verify if exist a database and if currently params
            # have already been calculated for this beam. if there,
            # is no dabatase, create one.
            try:
                db = Table.read(database_name)

                # Match function used to filter database.
                def match(table, key_colnames):
                    for key in key_colnames:
                        if (key == 'fx'
                                or key == 'fy'
                                or key == 'fz'):
                            continue

                        if table[key][0] != all_params_WF[key]:
                            return False

                    return True

                # If exist some value on table, search for match
                if len(db) != 0:

                    # sort by column's names except forces
                    key_names = db.colnames
                    key_names.remove('fx')
                    key_names.remove('fy')
                    key_names.remove('fz')
                    db = db.group_by([key for key in key_names])

                    # Verify if currently forces have already been
                    # calculated.
                    db_match = db.groups.filter(match)

                    # If length db_match is greater than zero,
                    # means that at least one point have already
                    # been calculated.
                    if len(db_match) > 0:
                        fx = db_match['fx'][0]
                        fy = db_match['fy'][0]
                        fz = db_match['fz'][0]

                        return [fx, fy, fz]

            except:
                # database create
                db_names = [name for name in all_params]
                db = Table(names=db_names)
                db.write(database_name)

            # Force (:obj:'Point')
            F = func(self, beam, ptc, ptc_pos, *args, **kwargs)

            all_params['fx'] = F[0]
            all_params['fy'] = F[1]
            all_params['fz'] = F[2]

            data_rows = [all_params[key] for key in db.colnames]

            # Add 'all_params' in a last row on database
            db.add_row(data_rows)

            # Save new database
            db.write(database_name, overwrite=True)

            return F

        return wrapped

    return force


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

        # auxiliary
        self._incident_ray = None
        self._incident_ray_abs = None
        self._incident_ray_direction = None
        self._electric_field_direction = None

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

    @property
    def incident_ray(self):
        return self._incident_ray

    @incident_ray.setter
    def incident_ray(self, vector):
        self._incident_ray = vector
        self._incident_ray_abs = np.linalg.norm(vector)
        self._incident_ray_power = np.linalg.norm(vector)
        if self._incident_ray_abs >= 1e-99:
            self._incident_ray_direction = [v/np.linalg.norm(vector)
                                            for v in vector]
        else:
            self._incident_ray_direction = [0, 0, 0]
            self._incident_ray_abs = 0
            self._incident_ray_power = 0

    @property
    def incident_ray_abs(self):
        return self._incident_ray_abs

    @property
    def incident_ray_power(self):
        return self._incident_ray_power

    @property
    def incident_ray_direction(self):
        return self._incident_ray_direction

    @property
    def electric_field_direction(self):
        return self._electric_field_direction

    @electric_field_direction.setter
    def electric_field_direction(self, vector):
        if np.linalg.norm(vector) != 0:
            self._electric_field_direction = [v/np.linalg.norm(vector)
                                              for v in vector]
        else:
            self._electric_field_direction = [0, 0, 0]

    @staticmethod
    def normal(theta, phi):
        """ Vector normal to surface point where a single ray hits. """
        if theta == 0:
            return [0, 0, 1]
        elif theta == pi:
            return [0, 0, -1]
        else:
            return [ma.sin(theta)*ma.cos(phi),
                    ma.sin(theta)*ma.sin(phi),
                    ma.cos(theta)]

    def incident_angle(self, theta, phi):
        normal = self.normal(theta, phi)
        return ma.acos(-np.dot(self.incident_ray_direction, normal))

    def ortonormal_incident_ray(self, theta, phi):
        k0 = self.incident_ray_direction
        normal = self.normal(theta, phi)
        dot = np.dot(k0, normal)
        if dot == 0:
            return [0, 0, 0]

        d0 = [n-k for n, k in zip(normal, [dot*k for k in k0])]

        if np.linalg.norm(d0) == 0:
            return [0, 0, 0]

        return [d/np.linalg.norm(d0) for d in d0]

    def crossing_angle(self, theta, phi):
        """ Crossing angle between the polarization direction of the
        incident beam and the normal vector of the incident plane."""

        k0 = self.incident_ray_direction
        ef0 = self.electric_field_direction
        n0 = self.normal(theta, phi)
        plane_normal = np.cross(k0, n0)

        if np.linalg.norm(plane_normal) == 0:
            return 0

        plane_normal /= np.linalg.norm(plane_normal)
        return ma.acos(abs(np.dot(ef0, plane_normal)))

    def refracted_angle(self, theta, phi):
        incident_angle = self.incident_angle(theta, phi)
        return cm.asin(self.medium_refractive_index
                       *ma.sin(incident_angle)
                       /self.refractive_index).real

    def reflectance(self, theta, phi):
        thetai = self.incident_angle(theta, phi)
        thetar = self.refracted_angle(theta, phi)
        beta = self.crossing_angle(theta, phi)

        # Rs and Rp: Fresnell transmission and reflection coeffici-
        # ents for s (perpendicular) and p (paralell) polarization.
        if thetai + thetar != 0:
            Rs = (ma.tan(thetai-thetar)/ma.tan(thetai+thetar))**2
            # Rs must not be greater than 1
            if Rs > 1:
                Rs = 1

            Rp = (ma.sin(thetai-thetar)/ma.sin(thetai+thetar))**2
            # Rp must not be greater than 1
            if Rp > 1:
                Rp = 1
        else:
            Rs = 1
            Rp = 1

        # Reflection coefficient.
        return Rs*ma.sin(beta)**2 + Rp*ma.cos(beta)**2

    def transmittance(self, theta, phi):
        return 1 - self.reflectance(theta, phi)

    def reflected_ray(self, theta, phi):
        if self.incident_ray_abs == 0:
            return [0, 0, 0]

        k0 = self.incident_ray
        thetai = self.incident_angle(theta, phi)
        d0 = self.ortonormal_incident_ray(theta, phi)

        rotation = cm.exp(1j*(pi-2*thetai))
        a_k0, a_d0 = rotation.real, rotation.imag
        ref_ray = [a_k0*k + a_d0*d for k, d in zip(k0, d0)]
        abs_ref_ray = self.incident_ray_abs*self.reflectance(theta, phi)

        return [abs_ref_ray*component for component in ref_ray]

    def internal_ray(self, theta, phi, n=0):
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

        return [abs_int_ray*component for component in int_ray]

    def refracted_ray(self, theta, phi, n=0):
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

        rotation = cm.exp(-1j*(n*(pi-2*thetar)+2*(thetai-thetar)))
        a_k0, a_d0 = rotation.real, rotation.imag
        refrac_ray = [a_k0*k + a_d0*d for k, d in zip(k0, d0)]
        abs_refrac_ray = (self.incident_ray_abs
                          *ma.exp(-self.absorption_coefficient*l)
                          *(1-R)**2*R**n)

        return [abs_refrac_ray*component for component in refrac_ray]

    def Qkd(self, theta, phi):
        if self.incident_ray_abs == 0:
            return 0, 0

        k0 = self.incident_ray
        thetai = self.incident_angle(theta, phi)
        thetar = self.refracted_angle(theta, phi)
        d0 = self.ortonormal_incident_ray(theta, phi)
        R = self.reflectance(theta, phi)
        # Distance travelled by a single ray before and after hit
        # sphere's surface.
        l = 2*self.radius*ma.cos(thetar)

        den = (1 + R**2 * ma.exp(-2*self.absorption_coefficient*l)
               + 2*R*ma.exp(-self.absorption_coefficient*l)*ma.cos(2*thetar))

        if den != 0:
            Qk = (1 + R*ma.cos(2*thetai)
                  - (1-R)**2*ma.exp(-self.absorption_coefficient*l)
                  * (ma.cos(2*(thetai-thetar))
                     + R*ma.exp(-self.absorption_coefficient*l)
                     * ma.cos(2*thetai))
                  / den)

            Qd = (-R*ma.sin(2*thetai)
                   - (1-R)**2 * ma.exp(-self.absorption_coefficient*l)
                   * (ma.sin(2*(thetar-thetai))
                      - R*ma.exp(-self.absorption_coefficient*l)
                      * ma.sin(2*thetai))
                   / den)

        else:
            Qk = (1 + R*ma.cos(2*thetai)
                  - ma.exp(-self.absorption_coefficient*l)
                  * (ma.cos(2*(thetar-thetai))
                     + R*ma.exp(-self.absorption_coefficient*l)
                     * ma.cos(2 * thetai)))

            Qd = (-R*ma.sin(2*thetai)
                   - ma.exp(-self.absorption_coefficient*l)
                   * (ma.sin(2*(thetar-thetai))
                      - R*ma.exp(-self.absorption_coefficient*l)
                      * ma.sin(2*thetai)))

        return Qk, Qd

    def force_ray(self, theta, phi):
        if self.incident_ray_abs == 0:
            return [0, 0, 0]

        k0 = self.incident_ray_direction
        thetai = self.incident_angle(theta, phi)
        d0 = self.ortonormal_incident_ray(theta, phi)
        Qk, Qd = self.Qkd(theta, phi)

        f = [Qk*k + Qd*d for k, d in zip(k0, d0)]

        return [self.medium_refractive_index*self.incident_ray_power
                *component/speed_of_light for component in f]


class Force(object):
    def __init__(self):
        pass

    @classmethod
    @timing
    @save_database('geo-opt')
    def geo_opt(cls, beam, ptc, ptc_pos):
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
            power = beam.intensity(bps)

            # Incident ray: vector k0 plus its power (intensity)
            ptc.incident_ray = [power*k for k in k0]

            # Check if is a valid condition
            if ptc.incident_angle(theta, phi) >= pi/2:
                return {'fx': 0, 'fy': 0, 'fz': 0, 'dA': 0}

            # Force of a single ray.
            force = ptc.force_ray(theta, phi)

            '''msg = ''
            msg += 'np=' + str(ptc.refractive_index) + ';\n'
            msg += 'nm=' + str(ptc.medium_refractive_index) + ';\n'
            msg += 'Rp=' + str(ptc.radius) + ';\n'
            msg += 'partpos=List' + str(ptc_pos.cartesian()) + ';\n'
            msg += 'point=List' + str(bps.cylindrical()) + ';\n'
            msg += 'theta=' + str(theta) + ';\n'
            msg += 'phi=' + str(phi) + ';\n'
            msg += 'n0=List' + str(ptc.normal(theta, phi)) + ';\n'
            msg += 'k0=List' + str(list(k0)) + ';\n'
            msg += 'E0=List' + str(list(ptc.electric_field_direction)) + ';\n'
            msg += 'beta=' + str(ptc.crossing_angle(theta, phi)) + ';\n'
            msg += 'thetai=' + str(ptc.incident_angle(theta, phi)) + ';\n'
            msg += 'thetar=' + str(ptc.refracted_angle(theta, phi))  + ';\n'
            msg += 'd0=List' + str(list(ptc.ortonormal_incident_ray(theta, phi))) + ';\n'
            msg += 'R=' + str(ptc.reflectance(theta, phi)) + ';\n'
            msg += 'Qkd=List' + str(list(ptc.Qkd(theta, phi))) + ';\n'
            msg += 'intensity=' + str(beam.intensity(bps)) + ';\n'
            msg += 'Force=List' + str(list(force)) + ';\n'
            msg = msg.replace('e-', '*10^-')
            pyperclip.copy(msg)
            pyperclip.paste()
            print(msg)
            time.sleep(1)'''

            return {'fx': force[0]*ma.sin(theta),
                    'fy': force[1]*ma.sin(theta),
                    'fz': force[2]*ma.sin(theta),
                    'dA': ma.sin(theta)}

        def integration():
            nptheta = 301

            theta_list = np.linspace(0, pi, nptheta)

            fx_theta_integral = []
            fy_theta_integral = []
            fz_theta_integral = []
            dA_theta_integral = []
            for theta in theta_list:
                phi_list = np.linspace(0, 2*pi,
                                       ma.floor(nptheta*ma.sin(theta)+1))
                fx_phi_integral = []
                fy_phi_integral = []
                fz_phi_integral = []
                dA_phi_integral = []
                for phi in phi_list:
                    values = integrand(theta, phi)
                    fx_phi_integral.append(values['fx'])
                    fy_phi_integral.append(values['fy'])
                    fz_phi_integral.append(values['fz'])
                    dA_phi_integral.append(values['dA'])
                fx_theta_integral.append(simps(fx_phi_integral, phi_list))
                fy_theta_integral.append(simps(fy_phi_integral, phi_list))
                fz_theta_integral.append(simps(fz_phi_integral, phi_list))
                dA_theta_integral.append(simps(dA_phi_integral, phi_list))

            eff_area = simps(dA_theta_integral, theta_list)
            fx = simps(fx_theta_integral, theta_list)/eff_area
            fy = simps(fy_theta_integral, theta_list)/eff_area
            fz = simps(fz_theta_integral, theta_list)/eff_area

            return [fx, fy, fz]

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
    fw.N = 15
    fw.L = 1*1e-3
    fw.R = 17.5e-6
    fw.Q = 0.95*fw.wavenumber
    fw.reference_function = func

    Rp = 17.5e-6

    ptc = SphericalParticle(radius=Rp, medium_refractive_index=1.33,
                            refractive_index=1.010*1.33)

    print(Force.geo_opt(fw, ptc, Point([0, 0, -0.01])))
