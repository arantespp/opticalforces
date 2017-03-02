import math as ma
from math import pi
import cmath as cm
from functools import wraps
import scipy.special as ss
from scipy.integrate import quad
import numpy as np


def add_amplitude(psi):
    @wraps(psi)
    def wrapped(self, pt):
        return psi(self, pt)*self.amplitude
    return wrapped

def add_phase(psi):
    @wraps(psi)
    def wrapped(self, pt):
        return psi(self, pt)*cm.exp(1j*self.phase)
    return wrapped


class Beam(object):
    def __init__(self, beams, name='generic-beam'):
        self.beams = beams
        self.name = name

        self._amplitude = 1
        self._phase = 0
        self._crossing_angle = pi/4

    def __str__(self):
        out = 'name: ' + self.name + '\n'
        for i, beam in enumerate(self.beams):
            out += '\n' + 'beam %d (%d)' %(i+1, i-len(self.beams)//2)
            out += '\n' + str(beam)
        return out

    def __add__(self, other):
        beams = list(self.beams) + list(other.beams)
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
    def crossing_angle(self):
        return self._crossing_angle

    @crossing_angle.setter
    def crossing_angle(self, value):
        self._crossing_angle = value

    def is_all_parameters_defined(self):
        for param, value in self.__dict__.items():
            if value is None and param[0] == '_':
                return False
        return True

    def psi(self, pt):
        return sum([beam.psi(pt) for beam in self.beams])

    def intensity(self, pt):
        """ Wave's intensity.

        Args:
            pt (:obj:'Point'): point at which want to calculate wave's
                intensity, that is psi's abs squared.

        Returns:
            Wave's intensity.

        """
        return abs(self.psi(pt))**2

    def RS(self, pt, r_max, r_min=0, np_rho=501, np_phi=3):
        """ Wave's Raleigh-Sommerfeld integral by 1/3 Simpson's Rule.

        Args:
            pt (:obj:'Point'): point at which want to calculate wave's
                Rayleigh-Sommerfeld complex value.
            r_max: aperture radius that will be used to simulate the
                wave aperture truncate.
            np_rho: rho's variable 1/3 Simpson's Rule discretization.
                a.k.a: Rho's number points.
            np_phi: phi's variable 1/3 Simpson's Rule discretization.
                a.k.a: Phi's number points.

        Returns:
            Raleigh-Sommerfeld complex value at a specific point 'pt'
                given a aperture with radius 'r_max'.

        """

        # RS integral's integrand
        def R(rho, phi):
            return (pt - Point(rho, phi, 0, 'cilin')).r

        def E(rho, phi):
            return cm.exp(1j*self.wavenumber*R(rho, phi))

        def integrand(rho, phi):
            return (rho*self.psi(Point(rho, phi, 0, 'cilin'))
                    *E(rho, phi) / R(rho, phi)**2)

        # S: 1/3 Simp's rule auxiliary matrix.
        auxrhovec = [1 if (rho == 0 or rho == np_rho - 1) \
            else (4 if rho % 2 != 0 else 2) for rho in range(np_rho)]

        auxphivec = [1 if (phi == 0 or phi == np_phi - 1) \
            else (4 if phi % 2 != 0 else 2) for phi in range(np_phi)]

        S = [[i * j for i in auxrhovec] for j in auxphivec]

        # F: 'integrand' mapped at discretized RS points.
        rhovec = np.linspace(r_min, r_max, np_rho)

        phivec = np.linspace(0, 2*ma.pi, np_phi)

        F = [[pt.z*integrand(rho, phi) / (1j*self.wavelength)
              for rho in rhovec]
             for phi in phivec]

        # Hadamard product between 'F' and 'S'
        H = sum(sum(np.multiply(F, S)))

        # Interval's discretization
        hrho = (r_max - r_min) / (np_rho-1)
        hphi = 2*ma.pi / (np_phi-1)

        return hphi*hrho*H/9

    def RSI(self, pt, r_max, r_min=0, np_rho=501, np_phi=3):
        """ Wave's Raleigh-Sommerfeld integral intensity.

        Args:
            pt (:obj:'Point'): point at which want to calculate wave's
                Rayleigh-Sommerfeld complex value.
            r_max: aperture radius that will be used to simulate the
                wave aperture truncate.

        Returns:
            Raleigh-Sommerfeld intensity value at a specific point 'pt'
                given a aperture with radius 'r_max'.

        """

        return abs(self.RS(pt, r_max, r_min, np_rho, np_phi))**2

    def k0(self, pt):
        """ k0 vector's direction.

        k0 vector's direction is defined by gradient of phase function.
        This method makes the phase derivative in x, y and z using Fi-
        nite Difference Coefficients found on
        http://web.media.mit.edu/~crtaylor/calculator.htm site.

        Args:
            pt (:obj:'Point'): point at which want to calculate wave's
                k0 vector's direction.

        Returns:
            A list containing the normalized k0 vector - [kx, ky, longitudinal_wavenumber]
        """

        # Delta
        h = 1e-10

        # Denominator coefficient
        den = 60*h

        # Locations of Sampled Points
        lsp = [-3, -2, -1, 0, 1, 2, 3]

        # Finite Difference Coefficients
        fdc = [-1, 9, -45, 0, 45, -9, 1]

        # Psi calculated over samples points
        psix = [self.psi(Point(pt.x + i*h, pt.y, pt.z)) for i in lsp]
        psiy = [self.psi(Point(pt.x, pt.y + i*h, pt.z)) for i in lsp]
        psiz = [self.psi(Point(pt.x, pt.y, pt.z + i*h)) for i in lsp]

        # Psi derivative
        psi_x = np.dot(fdc, psix)/den
        psi_y = np.dot(fdc, psiy)/den
        psi_z = np.dot(fdc, psiz)/den

        # k0 components
        psi = self.psi(pt)
        k0x = (psi_x/psi).imag
        k0y = (psi_y/psi).imag
        k0z = (psi_z/psi).imag

        if ma.isinf(k0x) is True or ma.isnan(k0x) is True:
            return [0, 0, 0]
        if ma.isinf(k0y) is True or ma.isnan(k0y) is True:
            return [0, 0, 0]
        if ma.isinf(k0z) is True or ma.isnan(k0z) is True:
            return [0, 0, 0]

        # normalize k0 vector
        if k0x != 0 or k0y != 0 or k0z != 0:
            k = [k0x, k0y, k0z]
            return k/np.linalg.norm(k)
        else:
            return [0, 0, 1]


class PlaneWave(Beam):
    params = ('_amplitude',
              '_phase',
              '_crossing_angle',
              '_vacuum_wavelength',
              '_vacuum_wavenumber',
              '_medium_refractive_index',
              '_wavelength',
              '_wavenumber',)

    def __init__(self, **kwargs):
        Beam.__init__(self, self)

        self.beams = [self]

        self.name = 'plane-wave'

        self._vacuum_wavelength = None
        self._vacuum_wavenumber = None
        self._medium_refractive_index = None
        self._wavelength = None
        self._wavenumber = None

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    def __str__(self):
        out = 'name: ' + self.name + '\n'
        for param in self.params:
            out += '    '
            out += param + ': ' + str(self.__dict__[param])
            out += '\n'
        return out

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
            self.medium_refractive_index = 2*pi/(wl*self.vacuum_wavenumber)

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
            self.medium_refractive_index = (k*self.vacuum_wavelength
                                            / (2*pi))

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


    @add_phase
    @add_amplitude
    def psi(self, pt):
        """ Wave's equation 'psi'.

        Args:
            pt (:obj:'Point'): point at which want to calculate 'psi'
                value.

        Returns:
            Wave's equation complex value of default plane wave decla-
                red on beam class.

        """
        return cm.exp(1j*self._wavenumber*pt.z)


class BesselBeam(PlaneWave):
    params = PlaneWave.params
    params += ('_longitudinal_wavenumber',
               '_transversal_wavenumber',
               '_axicon_angle',
               '_bessel_order',)

    def __init__(self, **kwargs):
        PlaneWave.__init__(self)

        self.name = 'bessel-beam'

        self._transversal_wavenumber = None
        self._longitudinal_wavenumber = None
        self._axicon_angle = None
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
            self.medium_refractive_index = (k*self.vacuum_wavelength
                                            / (2*pi))

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
    def bessel_order(self):
        return self._bessel_order

    @bessel_order.setter
    def bessel_order(self, value):
        self._bessel_order = value

    @add_phase
    @add_amplitude
    def psi(self, pt):
        return (ss.jv(self._bessel_order, self._transversal_wavenumber*pt.rho)
                * cm.exp(1j*self._longitudinal_wavenumber*pt.z)
                * cm.exp(1j*self._bessel_order*pt.phi))


class GaussianBeam(PlaneWave):
    params = PlaneWave.params
    params += ('_q_constant',)

    def __init__(self, **kwargs):
        PlaneWave.__init__(self)

        self.name = 'gaussian-beam'

        self._q_constant = 1

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    @property
    def q_constant(self):
        return self._q_constant

    @q_constant.setter
    def q_constant(self, q):
        self._q_constant = q

    @add_phase
    @add_amplitude
    def psi(self, pt):
        k = self.wavenumber
        q = self.q_constant
        return ((1/(1 + 1j*pt.z*2*q/k))
                * cm.exp(1j*pt.z*k)
                * cm.exp((-q*pt.rho**2)/(1+1j*pt.z*2*q/k)))


class BesselGaussBeam(BesselBeam, GaussianBeam):
    params = BesselBeam.params
    params += ('_q_constant',)

    def __init__(self, **kwargs):
        BesselBeam.__init__(self)
        GaussianBeam.__init__(self)

        self.name = 'bessel-gauss-beam'

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    @add_phase
    @add_amplitude
    def psi(self, pt):
        q = self.q_constant
        k = self.wavenumber
        krho = self.transversal_wavenumber

        if pt.z != 0:
            Q = q - 1j*k/(2*pt.z)
            num = 1j*k/(2*pt.z*Q)
            exp1 = cm.exp(1j*k*(pt.z+pt.rho**2/(2*pt.z)))
            bessel = ss.jv(0, num*krho*pt.rho)
            exp2 = cm.exp((krho**2 + k**2*pt.rho**2/pt.z**2)
                          * (-1/(4*Q)))
            if ma.isinf(bessel.real) is True:
                return ss.jv(0, krho*pt.rho)*cm.exp(-q*pt.rho**2)
            return -num*exp1*bessel*exp2
        else:
            return ss.jv(0, krho*pt.rho)*cm.exp(-q*pt.rho**2)


class BesselGaussBeamSuperposition(BesselGaussBeam):
    params = BesselGaussBeam.params
    params += ('_N',
               '_z_max',
               '_aperture_radius',
               '_L_constant',
               '_q_constant_real')

    def __init__(self, **kwargs):
        BesselGaussBeam.__init__(self)

        self.name = 'bessel-gauss-beam-superposition'

        self.beams = []

        self._q_constant = None
        self._N = None
        self._z_max = None
        self._aperture_radius = None
        self._L_constant = None
        self._q_constant_real = None

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

        self.__create_superposition()

    def __str__(self):
        out = ''
        out += str(PlaneWave.__str__(self))
        for i, beam in enumerate(self.beams):
            out += ('\n' + 'beam %d (n: %d)' %(i+1,
                                               i-len(self.beams)//2))
            out += '\n    ' + '_amplitude: ' + str(beam.amplitude)
            out += '\n    ' + '_q_constant: ' + str(beam.q_constant)
            out += '\n'
        return out

    @property
    def q_constant(self):
        return self._q_constant

    @q_constant.setter
    def q_constant(self, q):
        self._q_constant = q
        self.__create_superposition()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        self.__create_superposition()

    @property
    def z_max(self):
        return self._z_max

    @z_max.setter
    def z_max(self, value):
        self._z_max = value
        if (self.axicon_angle is None
                and self._aperture_radius is not None):
            self.axicon_angle = ma.atan(self._aperture_radius
                                        /self._z_max)
        if (self._axicon_angle is not None
                and self._aperture_radius is None):
            self._aperture_radius = (self._z_max
                                     *ma.tan(self._axicon_angle))

        self.__create_superposition()

    @property
    def aperture_radius(self):
        return self._aperture_radius

    @aperture_radius.setter
    def aperture_radius(self, value):
        self._aperture_radius = value
        self.L_constant = 3*value**2

        if self.axicon_angle is None and self.z_max is not None:
            self.axicon_angle = ma.atan(value/self._z_max)

        if self.axicon_angle is not None and self._z_max is None:
            self.z_max = value/ma.tan(self._axicon_angle)

        self.__create_superposition()

    @property
    def axicon_angle(self):
        return self._axicon_angle

    @axicon_angle.setter
    def axicon_angle(self, theta):
        self._axicon_angle = theta

        if self.longitudinal_wavenumber is not None:
            kz = self.longitudinal_wavenumber
            if theta != pi/2:
                self.wavenumber = kz/ma.cos(theta)

        if self.transversal_wavenumber is not None:
            krho = self.transversal_wavenumber
            if theta != 0:
                self.wavenumber = krho/ma.sin(theta)

        if self.z_max is None and self.aperture_radius is not None:
            self.z_max = self.aperture_radius/ma.tan(theta)

        if self.z_max is not None and self.aperture_radius is None:
            self.aperture_radius = self.z_max*ma.tan(theta)

        if self.wavenumber is not None:
            self.wavenumber = self.wavenumber

        self.__create_superposition()

    @property
    def L_constant(self):
        return self._L_constant

    @L_constant.setter
    def L_constant(self, L):
        self._L_constant = L
        self.q_constant_real = 6/L
        self.__create_superposition()

    @property
    def q_constant_real(self):
        return self._q_constant_real

    @q_constant_real.setter
    def q_constant_real(self, qr):
        self._q_constant_real = qr
        self.__create_superposition()

    def __create_superposition(self):
        if Beam.is_all_parameters_defined(self) is False:
            return

        def amplitude_n(n):
            qr = self.q_constant_real
            q = self.q_constant
            L = self.L_constant
            R = self.aperture_radius

            arg = (qr - q - 2j*pi*n/L)*R**2
            den = L*(qr-q)/2 - 1j*pi*n
            return cm.sinh(arg)/den

        self.beams = []

        for i in range(2*self.N + 1):
            n_index = i - self.N
            beam = BesselGaussBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self._wavelength
            beam.medium_refractive_index = self.medium_refractive_index
            beam.transversal_wavenumber = self.transversal_wavenumber
            beam.q_constant = (self.q_constant_real
                               - 1j*2*pi*n_index/self.L_constant)
            self.beams.append(beam)

    @add_phase
    @add_amplitude
    def psi(self, pt):
        return sum([beam.psi(pt) for beam in self.beams])


class FrozenWave(PlaneWave):
    params = PlaneWave.params
    params += ('_Q',
               '_N',
               '_L',)

    def __init__(self, **kwargs):
        PlaneWave.__init__(self)

        self.name = 'frozen-wave'

        self.beams = []

        self._Q = None
        self._N = None
        self._L = None
        self.z_max = None
        self.func = None

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

        self.__create_superposition()

    def __str__(self):
        out = ''
        out += str(PlaneWave.__str__(self))
        for i, beam in enumerate(self.beams):
            out += ('\n' + 'beam %d (n: %d)' %(i+1,
                                               i-len(self.beams)//2))
            out += '\n    ' + 'amplitude: ' 
            out += str(beam.amplitude)
            out += '\n    ' + 'longitudinal_wavenumber: ' 
            out += str(beam.longitudinal_wavenumber)
            out += '\n    ' + 'transversal_wavenumber: ' 
            out += str(beam.transversal_wavenumber)
            out += '\n    ' + 'axicon_angle: ' 
            out += str(beam.axicon_angle) + ' rad / '
            out += str(beam.axicon_angle*180/pi)  + ' degrees'
            out += '\n'
        return out

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
        self.z_max = value
        self.__create_superposition()

    @property
    def reference_function(self):
        return self.func

    @reference_function.setter
    def reference_function(self, func):
        self.func = func
        self.__create_superposition()

    def __create_superposition(self):
        if (Beam.is_all_parameters_defined(self) is False
                or self.func is None):
            return

        def amplitude_n(n):
            func_real = lambda z: (self.func(z)
                                   * cm.exp(-2j*pi*z*n/self.L)).real
            func_imag = lambda z: (self.func(z)
                                   * cm.exp(-2j*pi*z*n/self.L)).imag
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
            beam = BesselBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self.wavelength
            beam.medium_refractive_index = self.medium_refractive_index
            beam.longitudinal_wavenumber = self.Q + 2*pi*n_index/self.L
            self.beams.append(beam)

    @add_phase
    @add_amplitude
    def psi(self, pt):
        return sum([beam.psi(pt) for beam in self.beams])


class Point(object):
    def __init__(self, v1, v2, v3, system='cart'):
        if system == 'cart':
            self.__init(v1, v2, v3)

        elif system == 'cilin':
            self.__init(v1*ma.cos(v2), v1*ma.sin(v2), v3)

        elif system == 'spher':
            self.__init(v1*ma.sin(v2)*ma.cos(v3),
                        v1*ma.sin(v2)*ma.sin(v3),
                        v1*ma.cos(v2))

        else:
            raise NameError('System not defined. Choose amoung '
                            + '"cart", "cilin" or "spher".')

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
        return ("Cart = (%s, %s, %s)."
                % (str(self.x), str(self.y), str(self.z)) + '\n'
                + "Cilin = (%s, %s, %s)."
                % (str(self.rho), str(self.phi), str(self.z)) + '\n'
                + "Spher = (%s, %s, %s)."
                % (str(self.r), str(self.axicon_angle), str(self.phi)))

    def __init(self, x, y, z):
        # cartesian
        self.x = x
        self.y = y
        self.z = z

        # cilindrical
        self.rho = ma.sqrt(x**2 + y**2)
        if x != 0:
            self.phi = ma.atan(y / x)
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
            self.axicon_angle = ma.acos(z / self.r)
        else:
            self.axicon_angle = 0.0

    def abs(self):
        return self.r

    def normalize(self):
        return [self.x/self.r, self.y/self.r, self.z/self.r]


if __name__ == "__main__":
    print("Please, visit: https://github.com/arantespp/opticalforces")
