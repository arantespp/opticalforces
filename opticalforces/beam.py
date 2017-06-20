# -*- coding: utf-8 -*-
"""Beam module belonging to optical forces master degree project

This module provides classes which defines some optical beams, like
Bessel beam or Frozen Waves. Until now, we have these beams implemented:
    - Plane Wave
    - Bessel Beam
    - Gaussian Beam
    - Bessel-Gauss Beam
    - Bessel-Gauss Beam superposition
    - Frozen Waves

Example:
    ...

Todo:
    * Create all docstrings

.. Project:
   https://github.com/arantespp/opticalforces

"""

import math as ma
from math import pi
import cmath as cm
import copy
import scipy.special as ss
from scipy.integrate import quad
import numpy as np

# Speed of light.
SPEED_OF_LIGHT = 299792458
VACUUM_PERMEABILITY = pi*4e-7

def derivative(func, x0):
    # Delta
    h = 1e-9

    # Denominator coefficient
    den = 840*h

    # Locations of Sampled Points
    lsp = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    # Finite Difference Coefficients
    fdc = [3, -32, 168, -672, 0, 672, -168, 32, -3]

    return np.dot(fdc, [func(x0+i*h) for i in lsp])/den


class Beam(object):
    """ This class has all properties and methods that a specific scalar
    beam should have.

    """
    generic_params = ('_vacuum_wavelength',
                      '_vacuum_wavenumber',
                      '_medium_refractive_index',
                      '_wavelength',
                      '_wavenumber',)

    amp_pha_params = ('_amplitude',
                      '_phase',)

    intrinsic_params = ()

    params = amp_pha_params + generic_params + intrinsic_params

    def __init__(self, beams, name='generic-scalar-beam'):
        self.beams = beams
        self.name = name

        if isinstance(beams, list) is True:
            self._vacuum_wavelength = beams[0].vacuum_wavelength
            self._vacuum_wavenumber = beams[0].vacuum_wavenumber
            self._medium_refractive_index = beams[0].medium_refractive_index
            self._wavelength = beams[0].wavelength
            self._wavenumber = beams[0].wavenumber
        else:
            self._vacuum_wavelength = None
            self._vacuum_wavenumber = None
            self._medium_refractive_index = None
            self._wavelength = None
            self._wavenumber = None

        self._amplitude = 1
        self._phase = 0

    def __str__(self):
        out = 'name: ' + self.name + '\n'

        # print amplitude and phase
        for param in self.amp_pha_params:
            out += '    ' + param + ': ' + str(self.__dict__[param])
            out += '\n'

        # print generic params
        for param in self.generic_params:
            out += '    ' + param + ': ' + str(self.__dict__[param])
            out += '\n'

        for param in self.intrinsic_params:
            out += '    ' + param + ': ' + str(self.__dict__[param])
            out += '\n'

        if len(self.beams) > 1:
            # print beams
            for i, beam in enumerate(self.beams):
                out += '\n' + 'beam %d (%d)' %(i+1, i-len(self.beams)//2)
                out += ': %s' %beam.name
                out += '\n'

                for param in beam.amp_pha_params:
                    out += '    ' + param + ': ' + str(beam.__dict__[param])
                    out += '\n'

                for param in beam.intrinsic_params:
                    out += '    ' + param + ': ' + str(beam.__dict__[param])
                    out += '\n'

        return out

    def __add__(self, other):
        # raise error if one generic params if different from another.
        if self.wavelength != other.wavelength:
            raise NameError('Beams with differents wavelength')
        if self.vacuum_wavelength != other.vacuum_wavelength:
            raise NameError('Beams with differents vacuum_wavelength')

        # effetuate the sum because all generic params are equal.
        beams = []

        for beam in self.beams:
            beam._amplitude *= self._amplitude
            beam._phase += self._phase
            beams.append(copy.copy(beam))

        for beam in other.beams:
            beam._amplitude *= other._amplitude
            beam._phase += other._phase
            beams.append(copy.copy(beam))

        return Beam(beams)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    # ----- vacuum -----

    @property
    def vacuum_wavelength(self):
        return self._vacuum_wavelength

    @vacuum_wavelength.setter
    def vacuum_wavelength(self, wl0):
        self._vacuum_wavelength = wl0

        if self.vacuum_wavenumber is None:
            self.vacuum_wavenumber = 2*pi/wl0

        if (self.medium_refractive_index is not None
                and self.wavelength is None):
            self.wavelength = wl0/self.medium_refractive_index

        if (self.medium_refractive_index is None
                and self.wavenumber is not None):
            self.medium_refractive_index = wl0*self.wavelength/(2*pi)

    @property
    def vacuum_wavenumber(self):
        return self._vacuum_wavenumber

    @vacuum_wavenumber.setter
    def vacuum_wavenumber(self, k0):
        self._vacuum_wavenumber = k0

        if self.vacuum_wavelength is None:
            self.vacuum_wavelength = 2*pi/k0

        if (self.medium_refractive_index is not None
                and self.wavenumber is None):
            self.wavenumber = k0*self.medium_refractive_index

        if (self.medium_refractive_index is None
                and self.wavelength is not None):
            self.medium_refractive_index = 2*pi/(k0*self.wavelength)

    # ----- medium -----

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl

        if self.wavenumber is None:
            self.wavenumber = 2*pi/wl

        if (self.medium_refractive_index is not None
                and self.vacuum_wavelength is None):
            self.vacuum_wavelength = wl*self.medium_refractive_index

        if (self.medium_refractive_index is None
                and self.vacuum_wavenumber is not None):
            k0 = self.vacuum_wavenumber
            self.medium_refractive_index = 2*pi/(wl*k0)

    @property
    def wavenumber(self):
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, k):
        self._wavenumber = k

        if self.wavelength is None:
            self.wavelength = 2*pi/k

        if (self.medium_refractive_index is not None
                and self.vacuum_wavenumber is None):
            self.vacuum_wavenumber = k/self.medium_refractive_index

        if (self.medium_refractive_index is None
                and self.vacuum_wavelength is not None):
            self.medium_refractive_index = (k*self.vacuum_wavelength/(2*pi))

    # ----- medium refractive index -----

    @property
    def medium_refractive_index(self):
        return self._medium_refractive_index

    @medium_refractive_index.setter
    def medium_refractive_index(self, nm):
        self._medium_refractive_index = nm

        if (self.vacuum_wavelength is None
                and self.wavelength is not None):
            self.vacuum_wavelength = self.wavelength*nm

        if (self.vacuum_wavelength is not None
                and self.wavelength is None):
            self.wavelength = self.vacuum_wavelength/nm

        if (self.vacuum_wavenumber is None
                and self.wavenumber is not None):
            self.vacuum_wavenumber = self.wavenumber/nm

        if (self.vacuum_wavenumber is not None
                and self.wavenumber is None):
            self.wavenumber = self.vacuum_wavenumber*nm

    def is_all_params_defined(self):
        for param, value in self.__dict__.items():
            if value is None and param[0] == '_':
                return False
        return True

    def psi(self, x1, x2, x3, system='cartesian'):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.psi(x1, x2, x3, system)
                     for beam in self.beams]))

    def intensity(self, x1, x2, x3, system='cartesian'):
        """ Wave's intensity.

        Args:

        Returns:
            Wave's intensity.

        """
        return abs(self.psi(x1, x2, x3, system))**2

    def wavenumber_direction(self, x1, x2, x3, system='cartesian'):
        """ k0 vector's direction.

        k0 vector's direction is defined by gradient of phase function.
        This method makes the phase derivative in x, y and z using Fi-
        nite Difference Coefficients found on
        http://web.media.mit.edu/~crtaylor/calculator.htm site.

        Args:
            point (:obj:'Point'): point at which want to calculate
                wave's k0 vector's direction.

        Returns:
            A list containing the normalized k0 vector - [kx, ky, kz]
        """

        # k0 components
        (x0, y0, z0) = Point(x1, x2, x3).cartesian()
        psi = self.psi(x0, y0, z0, system)
        k0x = (derivative(lambda x: self.psi(x, y0, z0, system), x0)/psi).imag
        k0y = (derivative(lambda y: self.psi(x0, y, z0, system), y0)/psi).imag
        k0z = (derivative(lambda z: self.psi(x0, y0, z, system), z0)/psi).imag

        if (ma.isinf(k0x) is True
                or ma.isinf(k0y) is True
                or ma.isinf(k0z) is True):
            return (0, 0, 0)

        if (ma.isnan(k0x) is True
                or ma.isnan(k0y) is True
                or ma.isnan(k0z) is True):
            return (0, 0, 0)

        # normalize k0 vector
        if k0x != 0 or k0y != 0 or k0z != 0:
            k = [k0x, k0y, k0z]
            return (k0x/np.linalg.norm(k),
                    k0y/np.linalg.norm(k),
                    k0z/np.linalg.norm(k))
        else:
            return (0, 0, 0)


class ScalarPlaneWave(Beam):
    intrinsic_params = ()

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.beams = [self]
        self.name = 'scalar-plane-wave'

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    def psi(self, x1, x2, x3, system='cartesian'):
        """ Wave's equation 'psi'.

        Args:

        Returns:
            Wave's equation complex value of default plane wave decla-
                red on beam class.

        """
        if system == 'cartesian' or system == 'cylindrical':
            z = x3
        else:
            z = Point(x1, x2, x3, system).z

        return (self._amplitude*cm.exp(1j*self._phase)
                *cm.exp(1j*self._wavenumber*z))


class ScalarBesselBeam(Beam):
    intrinsic_params = ('_longitudinal_wavenumber',
                        '_transversal_wavenumber',
                        '_axicon_angle',
                        '_axicon_angle_degree',
                        '_bessel_order',)

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.beams = [self]
        self.name = 'scalar-bessel-beam'

        self._transversal_wavenumber = None
        self._longitudinal_wavenumber = None
        self._axicon_angle = None
        self._axicon_angle_degree = None
        self._bessel_order = 0

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    @property
    def wavenumber(self):
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, k):
        self._wavenumber = k

        if self.wavelength is None:
            self.wavelength = 2*pi/k

        if (self.medium_refractive_index is not None
                and self.vacuum_wavenumber is None):
            self.vacuum_wavenumber = k/self.medium_refractive_index

        if (self.medium_refractive_index is None
                and self.vacuum_wavelength is not None):
            self.medium_refractive_index = (k*self.vacuum_wavelength/(2*pi))

        if self.longitudinal_wavenumber is not None:
            kz = self.longitudinal_wavenumber

            if self.transversal_wavenumber is None:
                self.transversal_wavenumber = ma.sqrt(k**2 - kz**2)

            if self.axicon_angle is None:
                self.axicon_angle = ma.acos(kz/k)

        if self.transversal_wavenumber is not None:
            krho = self.transversal_wavenumber

            if self.longitudinal_wavenumber is None:
                self.longitudinal_wavenumber = ma.sqrt(k**2 - krho**2)

            if self.axicon_angle is None:
                self.axicon_angle = ma.asin(krho/k)

        if self.axicon_angle is not None:
            theta = self.axicon_angle

            if self.longitudinal_wavenumber is None:
                self.longitudinal_wavenumber = k*ma.cos(theta)

            if self.transversal_wavenumber is None:
                self.transversal_wavenumber = k.ma.sin(theta)

    @property
    def longitudinal_wavenumber(self):
        return self._longitudinal_wavenumber

    @longitudinal_wavenumber.setter
    def longitudinal_wavenumber(self, kz):
        self._longitudinal_wavenumber = kz

        if self.transversal_wavenumber is not None:
            krho = self.transversal_wavenumber
            self.wavenumber = ma.sqrt(kz**2 + krho**2)

        if self.axicon_angle is not None:
            theta = self.axicon_angle
            if theta != pi/2:
                self.wavenumber = kz/ma.cos(theta)

        if self.wavenumber is not None:
            self.wavenumber = self.wavenumber


    @property
    def transversal_wavenumber(self):
        return self._transversal_wavenumber

    @transversal_wavenumber.setter
    def transversal_wavenumber(self, krho):
        self._transversal_wavenumber = krho

        if self.longitudinal_wavenumber is not None:
            kz = self.longitudinal_wavenumber
            self.wavenumber = ma.sqrt(kz**2 + krho**2)

        if self.axicon_angle is not None:
            theta = self.axicon_angle
            if theta != 0:
                self.wavenumber = krho/ma.sin(theta)

        if self.wavenumber is not None:
            self.wavenumber = self.wavenumber

    @property
    def axicon_angle(self):
        return self._axicon_angle

    @axicon_angle.setter
    def axicon_angle(self, theta):
        self._axicon_angle = theta
        self._axicon_angle_degree = 180*theta/pi

        if self.longitudinal_wavenumber is not None:
            kz = self.longitudinal_wavenumber
            if theta != pi/2:
                self.wavenumber = kz/ma.cos(theta)

        if self.transversal_wavenumber is not None:
            krho = self.transversal_wavenumber
            if theta != 0:
                self.wavenumber = krho/ma.sin(theta)

        if self.wavenumber is not None:
            self.wavenumber = self.wavenumber

    @property
    def axicon_angle_degree(self):
        return self._axicon_angle_degree

    @axicon_angle_degree.setter
    def axicon_angle_degree(self, value):
        self._axicon_angle_degree = value
        self.axicon_angle = value*pi/180

    @property
    def bessel_order(self):
        return self._bessel_order

    @bessel_order.setter
    def bessel_order(self, value):
        self._bessel_order = value

    def psi(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        return (self._amplitude*cm.exp(1j*self._phase)
                *ss.jv(self._bessel_order,
                       self._transversal_wavenumber*rho)
                * cm.exp(1j*self._longitudinal_wavenumber*z)
                * cm.exp(1j*self._bessel_order*phi))


class ScalarGaussianBeam(Beam):
    intrinsic_params = ('_q',
                        '_gaussian_spot',
                        '_rayleigh_range',)

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.beams = [self]
        self.name = 'scalar-gaussian-beam'

        self._q = None
        self._gaussian_spot = None
        self._rayleigh_range = None

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value
        if value == 0:
            self.gaussian_spot = ma.inf
        elif cm.isinf(value):
            self.gaussian_spot = 0
        else:
            self.gaussian_spot = ma.sqrt(1/value.real)

    @property
    def gaussian_spot(self):
        return self._gaussian_spot

    @gaussian_spot.setter
    def gaussian_spot(self, value):
        self._gaussian_spot = value
        self._rayleigh_range = pi*value**2/self._wavelength

        if self._q is not None:
            return

        if value == 0:
            self._q = ma.inf
        elif ma.isinf(value):
            self._q = 0
        else:
            self._q = 1/value**2

    @property
    def rayleigh_range(self):
        return self._rayleigh_range

    @rayleigh_range.setter
    def rayleigh_range(self, value):
        self._rayleigh_range = value
        self.gaussian_spot = ma.sqrt(value*self._wavelength/pi)

    def waist_radius(self, x1, x2, x3, system='cartesian'):
        point = Point(x1, x2, x3, system)
        if self.rayleigh_range == 0:
            return ma.inf
        return self._gaussian_spot*ma.sqrt(1+(point.z/self.rayleigh_range)**2)

    def fwhm(self, x1, x2, x3, system='cartesian'):
        point = Point(x1, x2, x3, system)
        return self.waist_radius(point)*ma.sqrt(2*ma.log(2))

    def curvature_radius(self, x1, x2, x3, system='cartesian'):
        point = Point(x1, x2, x3, system)
        if point.z == 0:
            return ma.inf
        return point.z*(1+(self.rayleigh_range/point.z)**2)

    def gouy_phase(self, x1, x2, x3, system='cartesian'):
        point = Point(x1, x2, x3, system)
        if self.rayleigh_range == 0:
            return pi
        return ma.atan(point.z/self.rayleigh_range)

    def psi(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()
        k = self._wavenumber
        q = self._q
        return (self._amplitude*cm.exp(1j*self._phase)
                * (1/(1+1j*z*2*q/k))*cm.exp(1j*z*k)
                * cm.exp((-q*rho**2)/(1+1j*z*2*q/k)))


class ScalarBesselGaussBeam(ScalarBesselBeam, ScalarGaussianBeam):
    intrinsic_params = ScalarBesselBeam.intrinsic_params
    intrinsic_params += ScalarGaussianBeam.intrinsic_params
    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        ScalarBesselBeam.__init__(self)
        ScalarGaussianBeam.__init__(self)

        self.beams = [self]
        self.name = 'scalar-bessel-gauss-beam'

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    def psi(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        q = self._q
        k = self._wavenumber
        krho = self._transversal_wavenumber

        if z != 0:
            Q = q - 1j*k/(2*z)
            num = 1j*k/(2*z*Q)
            exp1 = cm.exp(1j*k*(z+rho**2/(2*z)))
            bessel = ss.jv(0, num*krho*rho)
            exp2 = cm.exp(-(krho**2 + k**2*rho**2/z**2)/(4*Q))
            if ma.isinf(bessel.real) is True:
                value = ss.jv(0, krho*rho)*cm.exp(-q*rho**2)
            value = -num*exp1*bessel*exp2
        else:
            value = ss.jv(0, krho*rho)*cm.exp(-q*rho**2)

        return self._amplitude*cm.exp(1j*self._phase)*value


class ScalarBesselGaussBeamSuperposition(ScalarBesselGaussBeam):
    intrinsic_params = ScalarBesselGaussBeam.intrinsic_params

    intrinsic_params += ('_N',
                         '_zmax',
                         '_aperture_radius',
                         '_L',
                         '_qr')

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        ScalarBesselGaussBeam.__init__(self)

        self.beams = [self]
        self.name = 'scalar-bessel-gauss-beam-superposition'

        self._N = None
        self._zmax = None
        self._aperture_radius = None
        self._L = None
        self._qr = None

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

        self.__create_superposition()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value

        if value == 0:
            self.gaussian_spot = ma.inf
        elif cm.isinf(value):
            self.gaussian_spot = 0
        else:
            self.gaussian_spot = ma.sqrt(1/value.real)

        self.__create_superposition()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        self.__create_superposition()

    @property
    def zmax(self):
        return self._zmax

    @zmax.setter
    def zmax(self, value):
        self._zmax = value

        if (self.axicon_angle is None
                and self.aperture_radius is not None):
            self.axicon_angle = ma.atan(self.aperture_radius/self.zmax)

        if (self.axicon_angle is not None
                and self.aperture_radius is None):
            self.aperture_radius = (self.zmax*ma.tan(self.axicon_angle))

        self.__create_superposition()

    @property
    def aperture_radius(self):
        return self._aperture_radius

    @aperture_radius.setter
    def aperture_radius(self, value):
        self._aperture_radius = value
        self.L = 3*value**2

        if self.axicon_angle is None and self.zmax is not None:
            self.axicon_angle = ma.atan(value/self.zmax)

        if self.axicon_angle is not None and self.zmax is None:
            self.zmax = value/ma.tan(self.axicon_angle)

        self.__create_superposition()

    @property
    def axicon_angle(self):
        return self._axicon_angle

    @axicon_angle.setter
    def axicon_angle(self, theta):
        self._axicon_angle = theta
        self._axicon_angle_degree = 180*theta/pi

        if self.longitudinal_wavenumber is not None:
            kz = self.longitudinal_wavenumber
            if theta != pi/2:
                self.wavenumber = kz/ma.cos(theta)

        if self.transversal_wavenumber is not None:
            krho = self.transversal_wavenumber
            if theta != 0:
                self.wavenumber = krho/ma.sin(theta)

        if self.zmax is None and self.aperture_radius is not None:
            self.zmax = self.aperture_radius/ma.tan(theta)

        if self.zmax is not None and self.aperture_radius is None:
            self.aperture_radius = self.zmax*ma.tan(theta)

        if self.wavenumber is not None:
            self.wavenumber = self.wavenumber

        self.__create_superposition()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):
        self._L = L
        self.qr = 6/L
        self.__create_superposition()

    @property
    def qr(self):
        return self._qr

    @qr.setter
    def qr(self, qr):
        self._qr = qr
        self.__create_superposition()

    def __create_superposition(self):
        if Beam.is_all_params_defined(self) is False:
            return

        def amplitude_n(n):
            arg = (self.qr - self.q - 2j*pi*n/self.L)*self.aperture_radius**2
            den = self.L*(self.qr-self.q)/2 - 1j*pi*n
            return cm.sinh(arg)/den

        self.beams = []

        for i in range(2*self.N + 1):
            n_index = i - self.N
            beam = ScalarBesselGaussBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self.wavelength
            beam.medium_refractive_index = self.medium_refractive_index
            beam.transversal_wavenumber = self.transversal_wavenumber
            beam.q = (self.qr - 1j*2*pi*n_index/self.L)
            self.beams.append(beam)

    def psi(self, x1, x2, x3, system='cartesian'):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.psi(x1, x2, x3, system) for beam in self.beams]))


class ScalarFrozenWave(Beam):
    intrinsic_params = ('_Q',
                        '_N',
                        '_L',
                        '_bessel_order',
                        '_reference_function',)

    params = Beam.params + intrinsic_params

    def __init__(self, centered=True, **kwargs):
        Beam.__init__(self, self)
        self.name = 'scalar-frozen-wave'
        self.beams = [self]
        self.centered = centered

        self._Q = None
        self._N = None
        self._L = None
        self._bessel_order = 0
        self._reference_function = None  # string
        self.func = None  # function
        #self.amplitudes = []

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

        self.__create_superposition()

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value
        self.__create_superposition()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value
        self.__create_superposition()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        self.__create_superposition()

    @property
    def bessel_order(self):
        return self._bessel_order

    @bessel_order.setter
    def bessel_order(self, value):
        self._bessel_order = value

    @property
    def reference_function(self):
        return self.func

    @reference_function.setter
    def reference_function(self, func):
        self.func = func
        self._reference_function = 'Already defined (name: %s)' % func.__name__
        self.__create_superposition()

    def __create_superposition(self):
        if (Beam.is_all_params_defined(self) is False or self.func is None):
            return

        def amplitude_n(n):
            func_real = lambda z: (self.func(z)*cm.exp(-2j*pi*z*n/self.L)).real
            func_imag = lambda z: (self.func(z)*cm.exp(-2j*pi*z*n/self.L)).imag
            if self.centered:
                an_real, err = quad(func_real, -self.L/2, self.L/2)
                an_imag, err = quad(func_imag, -self.L/2, self.L/2)
            else:
                an_real, err = quad(func_real, 0, self.L)
                an_imag, err = quad(func_imag, 0, self.L)
            return (an_real + 1j*an_imag)/self.L

        if 2*pi*self.N/self.L > self.wavenumber/2:
            error_msg = 'Combination of N, L and k does not '
            error_msg += 'satisfy Q range condition.'
            raise NameError(error_msg)

        if self.Q + 2*pi*self.N/self.L > self.wavenumber:
            msg = 'Q is too large. '
            msg += 'It was changed from %fk '%(self.Q/self.wavenumber)
            self.Q = self.wavenumber - 2*pi*self.N/self.L
            msg += 'to %fk.' % (self.Q/self.wavenumber)
            print(msg)

        if self.Q - 2*pi*self.N/self.L < 0:
            msg = 'Q is too low. '
            msg += 'It was changed from %fk '%(self.Q/self.wavenumber)
            self.Q = 2*pi*self.N/self.L
            msg += 'to %fk.' % (self.Q/self.wavenumber)
            print(msg)

        self.beams = []
        for i in range(2*self.N + 1):
            n_index = i - self.N
            beam = ScalarBesselBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self.wavelength
            beam.medium_refractive_index = self.medium_refractive_index
            beam.longitudinal_wavenumber = self.Q + 2*pi*n_index/self.L
            beam.bessel_order = self.bessel_order
            self.beams.append(beam)
            #self.amplitudes.append(amplitude_n(n_index))

    #def psi(self, x1, x2, x3, system='cartesian'):
    #    return (self._amplitude*cm.exp(1j*self._phase)
    #            *sum([beam.psi(x1, x2, x3, system)
    #                 for beam in self.beams]))


class VectorialBeam(Beam):
    intrinsic_params = ()

    params = Beam.params + intrinsic_params

    def __init__(self, beams, name='generic-vectorial-beam'):
        Beam.__init__(self, beams, name)

    def __add__(self, other):
        # raise error if one generic params if different from another.
        if self.wavelength != other.wavelength:
            raise NameError('Beams with differents wavelength')
        if self.vacuum_wavelength != other.vacuum_wavelength:
            raise NameError('Beams with differents vacuum_wavelength')

        # effetuate the sum because all generic params are equal.
        beams = []
        for beam in self.beams:
            beam._amplitude *= self._amplitude
            beam._phase += self._phase
            beams.append(copy.copy(beam))

        for beam in other.beams:
            beam._amplitude *= other._amplitude
            beam._phase += other._phase
            beams.append(copy.copy(beam))

        return VectorialBeam(beams)

    def Ex(self, x1, x2, x3, system='cartesian'):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.Ex(x1, x2, x3, system)
                      for beam in self.beams]))

    def Ey(self, x1, x2, x3, system='cartesian'):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.Ey(x1, x2, x3, system)
                      for beam in self.beams]))

    def Ez(self, x1, x2, x3, system='cartesian'):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.Ez(x1, x2, x3, system)
                      for beam in self.beams]))

    def E(self, x1, x2, x3, system='cartesian'):
        return (self.Ex(x1, x2, x3, system),
                self.Ey(x1, x2, x3, system),
                self.Ez(x1, x2, x3, system),)

    def electric_field(self, x1, x2, x3, system='cartesian'):
        return self.E(x1, x2, x3, system)

    def H(self, x1, x2, x3, system='cartesian'):
        Hx = (self._amplitude*cm.exp(1j*self._phase)
              *sum([beam.Hx(x1, x2, x3, system)
                    for beam in self.beams]))

        Hy = (self._amplitude*cm.exp(1j*self._phase)
              *sum([beam.Hy(x1, x2, x3, system)
                    for beam in self.beams]))

        Hz = (self._amplitude*cm.exp(1j*self._phase)
              *sum([beam.Hz(x1, x2, x3, system)
                    for beam in self.beams]))

        return (Hx, Hy, Hz)

    def magnetic_field(self, x1, x2, x3, system='cartesian'):
        return self.H(x1, x2, x3, system)

    def intensity(self, x1, x2, x3, system='cartesian'):
        Ex, Ey, Ez = self.E(x1, x2, x3, system)
        return abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2

    def electric_field_direction(self, x1, x2, x3, system='cartesian'):
        E0 = self.E(x1, x2, x3, system)
        E0_abs = np.linalg.norm(E0)
        return [abs(E)/E0_abs for E in E0]

    def wavenumber_direction(self, x1, x2, x3, system='cartesian'):
        wdir = np.cross(self.E(x1, x2, x3, system), self.H(x1, x2, x3, system))
        wdir_abs = np.linalg.norm(wdir)
        return [abs(k)/wdir_abs for k in wdir]


class VectorialBesselBeam(ScalarBesselBeam, VectorialBeam):
    intrinsic_params = ScalarBesselBeam.intrinsic_params
    intrinsic_params += VectorialBeam.intrinsic_params
    params = VectorialBeam.params + intrinsic_params

    def __init__(self, **kwargs):
        ScalarBesselBeam.__init__(self)
        VectorialBeam.__init__(self, self)

        self.beams = [self]
        self.name = 'vectorial-bessel-beam'

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    #def wavenumber_direction(self, x1, x2, x3, system='cartesian'):
    #    return super(ScalarBesselBeam, self).wavenumber_direction(x1, x2, x3, system)

    def __some_params(self):
        return (self._longitudinal_wavenumber,
                self._transversal_wavenumber,
                self._bessel_order,
                self._axicon_angle,)

    def Ex(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        kz, krho, ni, alpha = self.__some_params()

        return (self._amplitude*cm.exp(1j*self._phase)
                *0.25*(1+ma.cos(alpha))*(-1j)**ni*cm.exp(-1j*kz*z)
                *(+(1+ma.cos(alpha))*ss.jv(ni, krho*rho)
                  +0.5*(1-ma.cos(alpha))*(+cm.exp(+2j*phi)
                                           *ss.jv(ni+2, krho*rho)
                                          +cm.exp(-2j*phi)
                                           *ss.jv(ni-2, krho*rho))))

    def Ey(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        kz, krho, ni, alpha = self.__some_params()

        return (self._amplitude*cm.exp(1j*self._phase)
                *0.25*(1+ma.cos(alpha))*(-1j)**ni*cm.exp(-1j*kz*z)
                *(-0.5j*(1-ma.cos(alpha))*(+cm.exp(+2j*phi)
                                            *ss.jv(ni+2, krho*rho)
                                           -cm.exp(-2j*phi)
                                            *ss.jv(ni-2, krho*rho))))

    def Ez(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        kz, krho, ni, alpha = self.__some_params()

        return (self._amplitude*cm.exp(1j*self._phase)
                *0.25*(1+ma.cos(alpha))*(-1j)**ni*cm.exp(-1j*kz*z)
                *(+1j*ma.sin(alpha)*(+cm.exp(+1j*phi)
                                      *ss.jv(ni+1, krho*rho)
                                     -cm.exp(-1j*phi)
                                      *ss.jv(ni-1, krho*rho))))

    def Hx(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        kz, krho, ni, alpha = self.__some_params()

        const = SPEED_OF_LIGHT*VACUUM_PERMEABILITY

        return (self._amplitude*cm.exp(1j*self._phase)/const
                *0.25*(1+ma.cos(alpha))*(-1j)**ni*cm.exp(-1j*kz*z)
                *(-0.5j*(1-ma.cos(alpha))*(+cm.exp(+2j*phi)
                                            *ss.jv(ni+2, krho*rho)
                                           -cm.exp(-2j*phi)
                                            *ss.jv(ni-2, krho*rho))))

    def Hy(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        kz, krho, ni, alpha = self.__some_params()

        const = SPEED_OF_LIGHT*VACUUM_PERMEABILITY

        return (self._amplitude*cm.exp(1j*self._phase)/const
                *0.25*(1+ma.cos(alpha))*(-1j)**ni*cm.exp(-1j*kz*z)
                *(+(1+ma.cos(alpha))*ss.jv(ni, krho*rho)
                  -0.5*(1-ma.cos(alpha))*(+cm.exp(+2j*phi)
                                           *ss.jv(ni+2, krho*rho)
                                          +cm.exp(-2j*phi)
                                           *ss.jv(ni-2, krho*rho))))

    def Hz(self, x1, x2, x3, system='cartesian'):
        if system == 'cylindrical':
            rho, phi, z = x1, x2, x3
        else:
            rho, phi, z = Point(x1, x2, x3, system).cylindrical()

        kz, krho, ni, alpha = self.__some_params()

        const = SPEED_OF_LIGHT*VACUUM_PERMEABILITY

        return (self._amplitude*cm.exp(1j*self._phase)/const
                *0.25*(1+ma.cos(alpha))*(-1j)**ni*cm.exp(-1j*kz*z)
                *(ma.sin(alpha)*(+cm.exp(+1j*phi)
                                  *ss.jv(ni+1, krho*rho)
                                 +cm.exp(-1j*phi)
                                  *ss.jv(ni-1, krho*rho))))


class VectorialFrozenWave(VectorialBeam):
    intrinsic_params = ('_Q',
                        '_N',
                        '_L',
                        '_bessel_order',
                        '_reference_function',)

    params = VectorialBeam.params + intrinsic_params

    def __init__(self, centered=True, **kwargs):
        VectorialBeam.__init__(self, self)

        self.name = 'vectorial-frozen-wave'
        self.beams = [self]
        self.centered = centered

        self._Q = None
        self._N = None
        self._L = None
        self._bessel_order = 0
        self._reference_function = None  # string
        self.func = None  # function
        #self.amplitudes = []

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

        self.__create_superposition()

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value
        self.__create_superposition()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value
        self.__create_superposition()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        self.__create_superposition()

    @property
    def bessel_order(self):
        return self._bessel_order

    @bessel_order.setter
    def bessel_order(self, value):
        self._bessel_order = value

    @property
    def reference_function(self):
        return self.func

    @reference_function.setter
    def reference_function(self, func):
        self.func = func
        self._reference_function = 'Already defined (name: %s)' % func.__name__
        self.__create_superposition()

    def __create_superposition(self):
        if (Beam.is_all_params_defined(self) is False or self.func is None):
            return

        def amplitude_n(n):
            func_real = lambda z: (self.func(z)*cm.exp(-2j*pi*z*n/self.L)).real
            func_imag = lambda z: (self.func(z)*cm.exp(-2j*pi*z*n/self.L)).imag

            if self.centered:
                an_real, err = quad(func_real, -self.L/2, self.L/2)
                an_imag, err = quad(func_imag, -self.L/2, self.L/2)
            else:
                an_real, err = quad(func_real, 0, self.L)
                an_imag, err = quad(func_imag, 0, self.L)

            return (an_real + 1j*an_imag)/self.L

        if 2*pi*self.N/self.L > self.wavenumber/2:
            error_msg = 'Combination of N, L and k does not '
            error_msg += 'satisfy Q range condition.'
            raise NameError(error_msg)

        if self.Q + 2*pi*self.N/self.L > self.wavenumber:
            msg = 'Q is too large. '
            msg += 'It was changed from %fk '%(self.Q/self.wavenumber)
            self.Q = self.wavenumber - 2*pi*self.N/self.L
            msg += 'to %fk.' % (self.Q/self.wavenumber)
            print(msg)

        if self.Q - 2*pi*self.N/self.L < 0:
            msg = 'Q is too low. '
            msg += 'It was changed from %fk '%(self.Q/self.wavenumber)
            self.Q = 2*pi*self.N/self.L
            msg += 'to %fk.' % (self.Q/self.wavenumber)
            print(msg)

        self.beams = []
        for i in range(2*self.N + 1):
            n_index = i - self.N
            beam = VectorialBesselBeam()
            beam.wavelength = self.wavelength
            beam.medium_refractive_index = self.medium_refractive_index
            beam.longitudinal_wavenumber = self.Q + 2*pi*n_index/self.L
            beam.amplitude = (amplitude_n(n_index)*4
                              /(1+ma.cos(beam._axicon_angle))**2)
            beam.bessel_order = self.bessel_order
            self.beams.append(beam)


class Point(object):
    def __init__(self, x1, x2, x3, system='cartesian'):
        if system == 'cartesian':
            self.__init(x1, x2, x3)
        elif system == 'cylindrical':
            self.__init(x1*ma.cos(x2), x1*ma.sin(x2), x3)
        elif system == 'spherical':
            self.__init(x1*ma.sin(x2)*ma.cos(x3),
                        x1*ma.sin(x2)*ma.sin(x3),
                        x1*ma.cos(x2))
        else:
            raise NameError('System not defined. Choose amoung '
                            + '"cartesian", "cylindrical" or "spherical".')

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Point(x, y, z)

    def __str__(self):
        return ("cartesian = (%s, %s, %s)."
                % (str(self.x), str(self.y), str(self.z)) + '\n'
                + "cylindrical = (%s, %s, %s)."
                % (str(self.rho), str(self.phi), str(self.z)) + '\n'
                + "spherical = (%s, %s, %s)."
                % (str(self.r), str(self.theta), str(self.phi)))

    def __init(self, x, y, z):
        # cartesian
        self.x = x
        self.y = y
        self.z = z

        # cylindrical
        self.rho = ma.sqrt(x**2 + y**2)
        if x != 0:
            self.phi = ma.atan(y/x)
            self.phi += pi if x <= 0 and y >= 0 else 0
            self.phi -= pi if x <= 0 and y < 0 else 0
        else:
            if self.y < 0:
                self.phi = -pi/2
            elif self.y == 0:
                self.phi = 0.0
            else:
                self.phi = pi/2

        # spherical
        self.r = ma.sqrt(x**2 + y**2 + z**2)
        if self.r != 0:
            self.theta = ma.acos(z/self.r)
        else:
            self.theta = 0.0

    def abs(self):
        return self.r

    def normalize(self):
        return [self.x/self.r, self.y/self.r, self.z/self.r]

    def cartesian(self):
        return [self.x, self.y, self.z]

    def cylindrical(self):
        return [self.rho, self.phi, self.z]

    def spherical(self):
        return [self.r, self.theta, self.phi]


if __name__ == "__main__":
    print("Please, visit: https://github.com/arantespp/opticalforces")

    vbb0 = VectorialBesselBeam()
    vbb0.wavelength = 1064e-9
    vbb0.medium_refractive_index = 1.33
    vbb0.amplitude = 5
    vbb0.axicon_angle_degree = 10
    vbb0.bessel_order = 3

    vbb1 = VectorialBesselBeam()
    vbb1.wavelength = 1064e-9
    vbb1.medium_refractive_index = 1.33
    vbb1.amplitude = 5
    vbb1.axicon_angle_degree = 10
    vbb1.bessel_order = 0

    vbb2 = VectorialBesselBeam()
    vbb2.wavelength = 1064e-9
    vbb2.medium_refractive_index = 1.33
    vbb2.amplitude = 1
    vbb2.axicon_angle_degree = 15
    vbb2.bessel_order = 2

    sbb2 = ScalarBesselBeam()
    sbb2.wavelength = 1064e-9
    sbb2.medium_refractive_index = 1.33
    sbb2.amplitude = 1
    sbb2.axicon_angle_degree = 15
    sbb2.bessel_order = 0

    fw1 = VectorialFrozenWave()
    fw1.vacuum_wavelength = 1064e-9
    fw1.medium_refractive_index = 1.33
    fw1.Q = 0.699*fw1.wavenumber
    fw1.N = 15
    #print(fw1.centered)

    #vbb = vbb0 + vbb1 + vbb2

    rho, phi, z = 0.03, 8, 0.01

    #print(vbb2)

    #Ex = vbb2.Ex(rho, phi, z, 'cylindrical')
    #Ey = vbb2.Ey(x, y, z)
    #Ez = vbb2.Ez(x, y, z)

    print(vbb2)
    print('Ex', vbb2.Ex(rho, phi, z, 'cylindrical'))
    print('Ey', vbb2.Ey(rho, phi, z, 'cylindrical'))
    print('Ez', vbb2.Ez(rho, phi, z, 'cylindrical'))
    print('Hx', vbb2.Hx(rho, phi, z, 'cylindrical'))
    print('Hy', vbb2.Hy(rho, phi, z, 'cylindrical'))
    print('Hz', vbb2.Hz(rho, phi, z, 'cylindrical'))


    #print(Ex0, Ex1)
    #print(Ex0 + Ex1 + Ex2)
    #print(vbb.Ex(x, y, z))

    def ref_func(z):
        if abs(z) < 0.35*0.1:
            return ss.jv(0, 500*z)
        else:
            return 0

    fw = ScalarFrozenWave()
    fw.wavelength = 1064e-9
    fw.medium_refractive_index = 1.33
    fw.amplitude = 8
    fw.phase = 3
    fw.Q = 0.99*fw.wavenumber
    fw.N = 3
    fw.L = 0.1
    fw.bessel_order = 0
    #print(fw)
    fw.reference_function = ref_func

    #print(fw.psi(0.002*0,0,0.003*0))

    vfw = VectorialFrozenWave()
    vfw.wavelength = 1064e-9
    vfw.medium_refractive_index = 1.33
    vfw.Q = 0.99*vfw.wavenumber
    vfw.N = 5
    vfw.L = 0.1
    vfw.reference_function = ref_func

    #print(vfw.electric_field_direction(x, y, z))

    '''vb = VectorialBeam()

    b = ScalarPlaneWave()
    b.wavelength = 1064e-9
    b.medium_refractive_index = 1.33
    b.amplitude = 5

    vb.Ex = b.psi


    print(b.psi(1.1, 0, 1.1))
    print(vb.electric_field(1.1, 0, 1.1))'''

    '''b = ScalarPlaneWave()
    b.wavelength = 1064e-9
    b.medium_refractive_index = 1.33
    b.amplitude = 5

    b2 = ScalarPlaneWave()
    b2.wavelength = 1064e-9
    b2.medium_refractive_index = 1.33
    b2.phase = 3

    bb = ScalarBesselBeam()
    bb.wavelength = 1064e-9
    bb.medium_refractive_index = 1.33
    bb.phase = 1
    bb.axicon_angle_degree = 10

    bg = ScalarGaussianBeam()
    bg.wavelength = 1064e-9
    bg.medium_refractive_index = 1.33
    bg.amplitude = 4
    bg.phase = pi/10
    bg.q = 2

    bgb = ScalarBesselGaussBeam()
    bgb.wavelength = 1064e-9
    bgb.medium_refractive_index = 1.33
    bgb.axicon_angle_degree = 10
    bgb.amplitude = 3.1
    bgb.phase = pi/7
    bgb.q = 2

    bgbs = ScalarBesselGaussBeamSuperposition()
    bgbs.wavelength = 1064e-9
    bgbs.medium_refractive_index = 1.33
    bgbs.axicon_angle_degree = 10
    bgbs.amplitude = 3.5
    bgbs.phase = pi/4
    bgbs.q = 0.01
    bgbs.N = 4
    bgbs.aperture_radius = 1e-3

    def ref_func(z):
        if abs(z) < 0.35*0.1:
            return ss.jv(0, 500*z)
        else:
            return 0

    fw = ScalarFrozenWave()
    fw.wavelength = 1064e-9
    fw.medium_refractive_index = 1.33
    fw.amplitude = 3.5
    fw.Q = 0.99*fw.wavenumber
    fw.N = 3
    fw.L = 0.1
    fw.bessel_order = 1
    print(fw)
    fw.reference_function = ref_func

    print(fw.psi(0.002*0,0,0.003*0))

    c = b + b2 + bb + bg + bgb + bgbs + fw
    #c = fw
    c.amplitude = 3
    print(c)
    print(c.psi(0, 0, 0.001))
    print(c.psi(0.01, pi, 0.01, 'cylindrical'))
    print(c.psi(0.01, pi, 0.01, 'spherical'))
    print(c.wavenumber_direction(0.0001, pi, 0.02, 'spherical'))
    print(c.wavenumber_direction(0.0001, pi, 0.02, 'cylindrical'))
    '''
    #a = (ref_func, 2)
    #print(a)
