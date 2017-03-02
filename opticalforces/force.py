import math as ma
from math import pi
import cmath as cm
from numbers import Number
from scipy.integrate import dblquad
from astropy.table import Table
import numpy as np

from beam import Point


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

def save_database(regime):
    def force(func):
        @wraps(func)
        def wrapped(self, beam, ptc, ptc_pos, *args, **kwargs):
            # Database's name
            database_name = beam.name + '-' + regime + '.fits'
            all_params = {}

            # Beam's parameters
            for param in beam.params:
                all_params.update({param: getattr(beam, param)})

            # Particle's parameters
            for param in ptc.params:
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

                        return Point(fx, fy, fz)

            except:
                # database create
                db_names = [name for name in all_params]
                db = Table(names=db_names)
                db.write(database_name)

            # Force (:obj:'Point')
            F = func(self, beam, ptc, ptc_pos, *args, **kwargs)

            all_params['fx'] = F.x
            all_params['fy'] = F.y
            all_params['fz'] = F.z

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

    # Vector normal to surface point where a single ray hits.
    @staticmethod
    def n0(theta, phi):
        if theta == 0:
            return [0, 0, 1]
        elif theta == pi:
            return [0, 0, -1]
        else:
            return [ma.sin(theta)*ma.cos(phi),
                    ma.sin(theta)*ma.sin(phi),
                    ma.cos(theta)]

    @classmethod
    def incident_angle(cls, incident_ray, theta, phi):
        k0 = incident_ray
        n0 = cls.n0(theta, phi)
        if np.linalg.norm(k0) != 0:
            k0 /= np.linalg.norm(k0)
        return ma.acos(-np.dot(k0, n0))

    @classmethod
    def d0(cls, incident_ray, theta, phi):
        k0 = incident_ray
        n0 = cls.n0(theta, phi)
        if np.linalg.norm(k0) != 0:
            k0 /= np.linalg.norm(k0)
        dot = np.dot(k0, n0)
        d0 = [n-k for n, k in zip(n0, [dot*k for k in k0])]
        if np.linalg.norm(d0) != 0:
            return d0/np.linalg.norm(d0)
        else:
            return [0, 0, 0]

    def refracted_angle(self, incident_angle):
        return cm.asin(self.medium_refractive_index
                       *ma.sin(incident_angle)
                       /self.refractive_index).real

    def reflectance(self, incident_angle, crossing_angle=pi/4):
        # Crossing angle between the polarization direction of
        # the incident beam and the normal vector of the incident
        # plane.

        thetai = incident_angle
        thetar = self.refracted_angle(thetai)
        beta = crossing_angle

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

    def transmittance(self, incident_angle, crossing_angle=pi/4):
        return 1 - self.reflectance(incident_angle, crossing_angle)

    def reflected_ray(self, incident_ray, theta, phi, 
            crossing_angle=pi/4):
        k0 = incident_ray
        abs_k0 = np.linalg.norm(k0)
        if abs_k0 == 0:
            return [0, 0, 0]
        else:
            k0 /= abs_k0

        thetai = self.incident_angle(k0, theta, phi)

        if abs(thetai) >= pi/2:
            return [0, 0, 0]

        d0 = self.d0(k0, theta, phi)
        rotation = cm.exp(1j*(pi-2*thetai))
        a_k0, a_d0 = rotation.real, rotation.imag
        ref_ray = [a_k0*k + a_d0*d for k, d in zip(k0, d0)]
        abs_ref_ray = abs_k0*self.reflectance(thetai, crossing_angle)

        return [abs_ref_ray*component for component in ref_ray]

    def internal_ray(self, incident_ray, theta, phi, n=0, 
            crossing_angle=pi/4):
        k0 = incident_ray
        abs_k0 = np.linalg.norm(k0)
        if abs_k0 == 0:
            return [0, 0, 0]
        else:
            k0 /= abs_k0

        thetai = self.incident_angle(k0, theta, phi)

        if abs(thetai) >= pi/2:
            return [0, 0, 0]

        thetar = self.refracted_angle(thetai)
        d0 = self.d0(k0, theta, phi)
        R = self.reflectance(thetai, crossing_angle)
        # Distance travelled by a single ray before and after hit
        # sphere's surface.
        l = 2*self.radius*ma.cos(thetar)

        rotation = cm.exp(-1j*(n*(pi-2*thetar)+(thetai-thetar))) 
        a_k0, a_d0 = rotation.real, rotation.imag
        int_ray = [a_k0*k + a_d0*d for k, d in zip(k0, d0)]
        abs_int_ray = (abs_k0*ma.exp(-self.absorption_coefficient*l)
                       * (1-R)*R**n)

        return [abs_int_ray*component for component in int_ray]

    def refracted_ray(self, incident_ray, theta, phi, n=0, 
            crossing_angle=pi/4):
        k0 = incident_ray
        abs_k0 = np.linalg.norm(k0)
        if abs_k0 == 0:
            return [0, 0, 0]
        else:
            k0 /= abs_k0

        thetai = self.incident_angle(k0, theta, phi)

        if abs(thetai) >= pi/2:
            return [0, 0, 0]

        thetar = self.refracted_angle(thetai)
        d0 = self.d0(k0, theta, phi)
        R = self.reflectance(thetai, crossing_angle)
        # Distance travelled by a single ray before and after hit
        # sphere's surface.
        l = 2*self.radius*ma.cos(thetar)

        rotation = cm.exp(-1j*(n*(pi-2*thetar)+2*(thetai-thetar))) 
        a_k0, a_d0 = rotation.real, rotation.imag
        refrac_ray = [a_k0*k + a_d0*d for k, d in zip(k0, d0)]
        abs_refrac_ray = (abs_k0*ma.exp(-self.absorption_coefficient*l)
                          *(1-R)**2*R**n)

        return [abs_refrac_ray*component for component in refrac_ray]

    def Qkd(self, incident_ray, theta, phi, crossing_angle=pi/4):
        k0 = incident_ray
        abs_k0 = np.linalg.norm(k0)
        if abs_k0 == 0:
            return [0, 0, 0]
        else:
            k0 /= abs_k0

        thetai = self.incident_angle(k0, theta, phi)

        if abs(thetai) >= pi/2:
            return 0, 0

        thetar = self.refracted_angle(thetai)
        d0 = self.d0(k0, theta, phi)
        R = self.reflectance(thetai, crossing_angle)
        # Distance travelled by a single ray before and after hit
        # sphere's surface.
        l = 2*self.radius*ma.cos(thetar)

        den = (1 + R**2 * ma.exp(-2*self.absorption_coefficient*l)
               + 2*R*ma.exp(-self.absorption_coefficient*l)
               * ma.cos(2*thetar))

        if den != 0:
            Qk = (1 + R*ma.cos(2*thetai)
                  - (1-R)**2*ma.exp(-self.absorption_coefficient*l)
                  * (ma.cos(2*(thetai-thetar))
                     + R*ma.exp(-self.absorption_coefficient*l)
                     * ma.cos(2*thetai))
                  / den)

            Qd = -(R*ma.sin(2*thetai)
                   - (1-R)**2 * ma.exp(-self.absorption_coefficient*l)
                   * (ma.sin(2*(thetai-thetar))
                      + R*ma.exp(-self.absorption_coefficient*l)
                      * ma.sin(2*thetai))
                   / den)

        else:
            Qk = (1 + R*ma.cos(2*thetai)
                  - ma.exp(-self.absorption_coefficient*l)
                  * (ma.cos(2*(thetar-thetai))
                     + R*ma.exp(-self.absorption_coefficient*l)
                     * ma.cos(2 * thetai)))

            Qd = -(R*ma.sin(2*thetai)
                   - ma.exp(-self.absorption_coefficient*l)
                   * (ma.sin(2*(thetai-thetar))
                      + R*ma.exp(-self.absorption_coefficient*l)
                      * ma.sin(2*thetai)))

        return Qk, Qd

    def force_ray(self, incident_ray, theta, phi, crossing_angle=pi/4):
        k0 = incident_ray
        abs_k0 = np.linalg.norm(k0)
        if abs_k0 == 0:
            return [0, 0, 0]
        else:
            k0 /= abs_k0

        thetai = self.incident_angle(k0, theta, phi)

        if abs(thetai) >= pi/2:
            return [0, 0, 0]

        d0 = self.d0(k0, theta, phi)

        Qk, Qd = self.Qkd(k0, theta, phi, crossing_angle)

        f = [Qk*k + Qd*d for k, d in zip(k0, d0)]

        msg = ''
        msg += 'theta: ' + str(round_sig(theta*180/pi, 3)) + '\n'
        msg += 'phi:   ' + str(round_sig(phi*180/pi, 3)) + '\n'
        msg += 'thetai:' + str(round_sig(thetai*180/pi, 3)) + '\n'
        msg += 'k0:    ' + '[' + str(round_sig(k0[0], 3)) + ', ' + str(round_sig(k0[1], 3)) + ', ' + str(round_sig(k0[2], 3)) + '] \n'
        msg += 'd0:    ' + '[' + str(round_sig(d0[0], 3)) + ', ' + str(round_sig(d0[1], 3)) + ', ' + str(round_sig(d0[2], 3)) + '] \n'
        msg += 'Qk:    ' + str(round_sig(Qk, 3)) + '\n'
        msg += 'Qd:    ' + str(round_sig(Qd, 3)) + '\n'
        msg += 'f:     ' + '[' + str(round_sig(f[0], 3)) + ', ' + str(round_sig(f[1], 3)) + ', ' + str(round_sig(f[2], 3)) + '] \n'


        print(msg)

        return [self.medium_refractive_index*abs_k0**2*component
                / speed_of_light for component in f]


class Force(object):
    def __init__(self):
        pass

    @classmethod
    #@save_database('geo-opt-without-k0')
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

        # Return each infinitesimal sphere surface force contribuition,
        # in other words, the force integral over sphere surface's
        # integrand value as function of theta and phi, which are par-
        # ticle coordinates.
        def integrand(theta, phi):

            # Beam particle surface: beam coordinates point that
            # match the point at theta and phi on particle surface.
            bps = Point(ptc.radius, theta, phi, 'spher') + ptc_pos

            # Vector parallel to the direction of a single ray.
            k0 = beam.k0(bps)

            # Beam's power at particle surface
            power = beam.intensity(bps)

            # Incident ray: vector k0 plus its power (intensity)
            incident_ray = [power*k for k in k0]

            # Force of a single ray.
            force = ptc.force_ray(incident_ray, theta, phi, 
                                  beam.crossing_angle)

            # Return [[Integrand value], [dA]]
            return [f*ma.sin(theta)*ptc.radius**2 for f in force]


        # Effective area's element. Some regions are not illuminated
        # by beam, therefore, this function returns only element of
        # area that is illuminated.
        def effective_dA(theta, phi):
            # Beam particle surface: beam coordinates point that
            # match the point at theta and phi on particle surface.
            bps = Point(ptc.radius, theta, phi, 'spher') + ptc_pos

            # Vector parallel to the direction of a single ray.
            k0 = beam.k0(bps)

            # Vector perpendicular to the direction of a single ray.
            d0 = ptc.d0(k0, theta, phi)

            # Incident angle.
            thetai = ptc.incident_angle(k0, theta, phi)

            # If incidente angle is breater than pi/2, means that a ray
            # at specific point actually does not hit there. So, there
            # is no force contribuition.
            if thetai >= pi/2:
                return 0
            else:
                return ptc.radius**2*ma.sin(theta)

        # integral over semisphere surface
        def force_times_surface(forceDirection):
            if forceDirection == 'fx':
                fd = 0
            elif forceDirection == 'fy':
                fd = 1
            elif forceDirection == 'fz':
                fd = 2
            else:
                raise 'Direction not defined'

            f, err = dblquad(lambda t, p: integrand(t, p)[fd],
                             0,  # phi initial
                             2*pi,  # phi final
                             lambda t: 0,  # theta initial
                             lambda t: pi)  # theta final

            return f

        # Effective surface which is illuminated.
        A, err = dblquad(effective_dA,
                         0,  # phi initial
                         2*pi,  # phi final
                         lambda t: 0,  # theta initial
                         lambda t: pi)  # theta final

        # Forces
        fx = force_times_surface('fx')/A
        fy = force_times_surface('fy')/A
        fz = force_times_surface('fz')/A

        return Point(fx, fy, fz)


if __name__ == '__main__':
    from beam import *
    ptc = SphericalParticle(radius=11*1064e-9, 
                            refractive_index=1, 
                            medium_refractive_index=1, 
                            absorption_coefficient=1e10)

    beam = BesselBeam(medium_refractive_index=1, wavelength=1064e-9, 
                      axicon_angle=0.1)

    print(ptc.force_ray(theta=3*pi/4, phi=0, incident_ray=[0,0,2]))
    #print(Force.geo_opt(beam, ptc, Point(1e-6, 0, 0.1)))
