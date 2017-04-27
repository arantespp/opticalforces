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


class Beam(object):
    """ This class has all properties and methods that a specific beam
    should have.

    """
    generic_params = ('_electric_field_direction',
                      '_vacuum_wavelength',
                      '_vacuum_wavenumber',
                      '_medium_refractive_index',
                      '_wavelength',
                      '_wavenumber',)

    amp_pha_params = ('_amplitude',
                      '_phase',)

    intrinsic_params = ()

    params = amp_pha_params + generic_params + intrinsic_params

    def __init__(self, beams, name='generic-beam'):
        self.beams = beams
        self.name = name

        if isinstance(beams, list) is True:
            self._electric_field_direction = beams[0].electric_field_direction
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

        self._electric_field_direction = [1, 0, 0]
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
        # effetuate the sum because all generic params is equal.
        beams = []
        for beam in self.beams:
            _beam = copy.copy(beam)
            _beam.amplitude *= self.amplitude
            _beam.phase += self.phase
            beams.append(_beam)

        for beam in other.beams:
            _beam = copy.copy(beam)
            _beam.amplitude *= other.amplitude
            _beam.phase += other.phase
            beams.append(_beam)
        #beams = list(self.beams) + list(other.beams)
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

    @property
    def electric_field_direction(self):
        return self._electric_field_direction

    @electric_field_direction.setter
    def electric_field_direction(self, vector):
        self._electric_field_direction = [v/np.linalg.norm(vector)
                                          for v in vector]

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

    def is_all_parameters_defined(self):
        for param, value in self.__dict__.items():
            if value is None and param[0] == '_':
                return False
        return True

    def psi(self, point):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.psi(point) for beam in self.beams]))

    def intensity(self, point):
        """ Wave's intensity.

        Args:
            point (:obj:'Point'): point at which want to calculate
                wave's intensity, that is psi's abs squared.

        Returns:
            Wave's intensity.

        """
        return abs(self.psi(point))**2

    def wavenumber_direction(self, point):
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

        # Delta
        h = 1e-15

        # Denominator coefficient
        den = 840*h

        # Locations of Sampled Points
        lsp = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

        # Finite Difference Coefficients
        fdc = [3, -32, 168, -672, 0, 672, -168, 32, -3]

        # Psi calculated over samples points
        psix = [self.psi(Point([point.x+i*h, point.y, point.z])) for i in lsp]
        psiy = [self.psi(Point([point.x, point.y+i*h, point.z])) for i in lsp]
        psiz = [self.psi(Point([point.x, point.y, point.z+i*h])) for i in lsp]

        # Psi derivative
        psi_x = np.dot(fdc, psix)/den
        psi_y = np.dot(fdc, psiy)/den
        psi_z = np.dot(fdc, psiz)/den

        # k0 components
        psi = self.psi(point)
        k0x = (psi_x/psi).imag
        k0y = (psi_y/psi).imag
        k0z = (psi_z/psi).imag

        if (ma.isinf(k0x) is True
                or ma.isinf(k0y) is True
                or ma.isinf(k0z) is True):
            return [0, 0, 0]

        if (ma.isnan(k0x) is True
                or ma.isnan(k0y) is True
                or ma.isnan(k0z) is True):
            return [0, 0, 0]

        # normalize k0 vector
        if k0x != 0 or k0y != 0 or k0z != 0:
            k = [k0x, k0y, k0z]
            return [k0x/np.linalg.norm(k),
                    k0y/np.linalg.norm(k),
                    k0z/np.linalg.norm(k)]
        else:
            return [0, 0, 1]


class PlaneWave(Beam):
    intrinsic_params = ()

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.beams = [self]

        self.name = 'plane-wave'

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    def psi(self, point):
        """ Wave's equation 'psi'.

        Args:
            point (:obj:'Point'): point at which want to calculate
                'psi' value.

        Returns:
            Wave's equation complex value of default plane wave decla-
                red on beam class.

        """
        return (self._amplitude*cm.exp(1j*self._phase)
                *cm.exp(1j*self._wavenumber*point.z))


class BesselBeam(Beam):
    intrinsic_params = ('_longitudinal_wavenumber',
                        '_transversal_wavenumber',
                        '_axicon_angle',
                        '_axicon_angle_degree',
                        '_bessel_order',)

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.beams = [self]

        self.name = 'bessel-beam'

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

    def psi(self, point):
        return (self._amplitude*cm.exp(1j*self._phase)
                *ss.jv(self._bessel_order,
                       self._transversal_wavenumber*point.rho)
                * cm.exp(1j*self._longitudinal_wavenumber*point.z)
                * cm.exp(1j*self._bessel_order*point.phi))


class GaussianBeam(Beam):
    intrinsic_params = ('_q',
                        '_gaussian_spot',
                        '_rayleigh_range',)

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.beams = [self]

        self.name = 'gaussian-beam'

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

    def waist_radius(self, point):
        if self.rayleigh_range == 0:
            return ma.inf
        return self._gaussian_spot*ma.sqrt(1+(point.z/self.rayleigh_range)**2)

    def fwhm(self, point):
        return self.waist_radius(point)*ma.sqrt(2*ma.log(2))

    def curvature_radius(self, point):
        if point.z == 0:
            return ma.inf
        return point.z*(1+(self.rayleigh_range/point.z)**2)

    def gouy_phase(self, point):
        if self.rayleigh_range == 0:
            return pi
        return ma.atan(point.z/self.rayleigh_range)

    def psi(self, point):
        k = self._wavenumber
        q = self._q
        return (self._amplitude*cm.exp(1j*self._phase)
                * (1/(1+1j*point.z*2*q/k))*cm.exp(1j*point.z*k)
                * cm.exp((-q*point.rho**2)/(1+1j*point.z*2*q/k)))

    def wavenumber_direction(self, point):
        wavelength = self._wavelength
        gaussian_spot = self._gaussian_spot
        wavenumber = self._wavenumber
        curvature_radius = self.curvature_radius(point)

        if ma.isinf(curvature_radius):
            return [0, 0, 1]

        k0x = point.x/curvature_radius
        k0y = point.y/curvature_radius
        k0z = ((1-(2/(wavenumber*gaussian_spot)**2)
                * (1/(1+((point.z*wavelength)/(pi*gaussian_spot**2))**2))
                - (point.rho**2)/(2*curvature_radius**2)
                * (1-(pi*gaussian_spot**2/(wavelength*point.z))**2)))

        k0 = [k0x, k0y, k0z]

        return k0/np.linalg.norm(k0)


class BesselGaussBeam(BesselBeam, GaussianBeam):
    intrinsic_params = BesselBeam.intrinsic_params
    intrinsic_params += GaussianBeam.intrinsic_params

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        BesselBeam.__init__(self)
        GaussianBeam.__init__(self)

        self.beams = [self]

        self.name = 'bessel-gauss-beam'

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    def psi(self, point):
        q = self._q
        k = self._wavenumber
        krho = self._transversal_wavenumber

        if point.z != 0:
            Q = q - 1j*k/(2*point.z)
            num = 1j*k/(2*point.z*Q)
            exp1 = cm.exp(1j*k*(point.z+point.rho**2/(2*point.z)))
            bessel = ss.jv(0, num*krho*point.rho)
            exp2 = cm.exp(-(krho**2 + k**2*point.rho**2/point.z**2)/(4*Q))
            if ma.isinf(bessel.real) is True:
                value = ss.jv(0, krho*point.rho)*cm.exp(-q*point.rho**2)
            value = -num*exp1*bessel*exp2
        else:
            value = ss.jv(0, krho*point.rho)*cm.exp(-q*point.rho**2)

        return self._amplitude*cm.exp(1j*self._phase)*value


class BesselGaussBeamSuperposition(BesselGaussBeam):
    intrinsic_params = BesselGaussBeam.intrinsic_params

    intrinsic_params += ('_N',
                         '_zmax',
                         '_aperture_radius',
                         '_L',
                         '_qr')

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        BesselGaussBeam.__init__(self)

        self.beams = [self]

        self.name = 'bessel-gauss-beam-superposition'

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
        if Beam.is_all_parameters_defined(self) is False:
            return

        def amplitude_n(n):
            arg = (self.qr - self.q - 2j*pi*n/self.L)*self.aperture_radius**2
            den = self.L*(self.qr-self.q)/2 - 1j*pi*n
            return cm.sinh(arg)/den

        self.beams = []

        for i in range(2*self.N + 1):
            n_index = i - self.N
            beam = BesselGaussBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self.wavelength
            beam.medium_refractive_index = self.medium_refractive_index
            beam.transversal_wavenumber = self.transversal_wavenumber
            beam.q = (self.qr - 1j*2*pi*n_index/self.L)
            self.beams.append(beam)

    def psi(self, point):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.psi(point) for beam in self.beams]))


class FrozenWave(Beam):
    intrinsic_params = ('_Q',
                        '_N',
                        '_L',
                        '_func',)

    params = Beam.params + intrinsic_params

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.name = 'frozen-wave'

        self.beams = [self]

        self._Q = None
        self._N = None
        self._L = None
        self._func = None
        self.func = None

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
    def reference_function(self):
        return self.func

    @reference_function.setter
    def reference_function(self, func):
        self.func = func
        self._func = "Already defined"
        self.__create_superposition()

    def __create_superposition(self):
        if (Beam.is_all_parameters_defined(self) is False
                or self.func is None):
            return

        def amplitude_n(n):
            func_real = lambda z: (self.func(z)*cm.exp(-2j*pi*z*n/self.L)).real
            func_imag = lambda z: (self.func(z)*cm.exp(-2j*pi*z*n/self.L)).imag
            an_real, err = quad(func_real, -self.L/2, self.L/2)
            an_imag, err = quad(func_imag, -self.L/2, self.L/2)
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
            beam = BesselBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self.wavelength
            beam.medium_refractive_index = self.medium_refractive_index
            beam.longitudinal_wavenumber = self.Q + 2*pi*n_index/self.L
            self.beams.append(beam)

    def psi(self, point):
        return (self._amplitude*cm.exp(1j*self._phase)
                *sum([beam.psi(point) for beam in self.beams]))


class Point(object):
    def __init__(self, vector, system='cartesian'):
        v1 = vector[0]
        v2 = vector[1]
        v3 = vector[2]
        if system == 'cartesian':
            self.__init(v1, v2, v3)

        elif system == 'cylindrical':
            self.__init(v1*ma.cos(v2), v1*ma.sin(v2), v3)

        elif system == 'spherical':
            self.__init(v1*ma.sin(v2)*ma.cos(v3),
                        v1*ma.sin(v2)*ma.sin(v3),
                        v1*ma.cos(v2))

        else:
            raise NameError('System not defined. Choose amoung '
                            + '"cartesian", "cylindrical" or "spherical".')

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point([x, y, z])

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Point([x, y, z])

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
