import math as ma
from math import pi
import cmath as cm
import numpy as np
import scipy.special as ss
from scipy.integrate import quad
from functools import wraps

class Beam(object):
    def __init__(self, beams, name='generic-beam'):
        self.beams = beams
        self.name = name
        self.params = None

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
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    def add_amplitude(psi):
        @wraps(psi)
        def psi_args(self, pt):
            return psi(self, pt)*self._amplitude
        return psi_args

    def add_phase(psi):
        @wraps(psi)
        def psi_args(self, pt):
            return psi(self, pt)*cm.exp(1j*self._phase)
        return psi_args

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
            return cm.exp(1j*self.k*R(rho, phi))

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
            A list containing the normalized k0 vector - [kx, ky, kz]

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
        psix = list(map(
            lambda i: self.psi(Point(pt.x + i*h, pt.y, pt.z)), lsp))

        psiy = list(map(
            lambda i: self.psi(Point(pt.x, pt.y + i*h, pt.z)), lsp))

        psiz = list(map(
            lambda i: self.psi(Point(pt.x, pt.y, pt.z + i*h)), lsp))

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
    def __init__(self, **kwargs):
        self.beams = [self]

        self.name = 'plane-wave'

        self._amplitude = 1
        self._phase = 0
        self._beta = pi/4
        self._wavelength = None
        self._k = None
        self._nm = None

        self.params = ('_amplitude',
                       '_phase',
                       '_beta',
                       '_wavelength',
                       '_k',
                       '_nm',)

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    def __str__(self):
        out = 'name: ' + self.name + '\n'
        for param, value in self.__dict__.items():
            if param[0] == '_':
                out += '    '
                out += param + ': ' + str(value)
                out += '\n'
        return out

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value
        #self.__set_wavelength()
        self.__set_k()
        self.__set_nm()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value
        self.__set_wavelength()
        #self.__set_k()
        self.__set_nm()

    @property
    def nm(self):
        return self._nm

    @nm.setter
    def nm(self, value):
        self._nm = value
        self.__set_wavelength()
        self.__set_k()
        #self.__set_nm()

    def __set_wavelength(self):
        if self._k is not None and self._nm is not None:
            self._wavelength = self._nm*2*pi/self.k

    def __set_k(self):
        if self._wavelength is not None and self._nm is not None:
            self._k = self._nm*2*pi/self.wavelength

    def __set_nm(self):
        if self._wavelength is not None and self._k is not None:
            self._nm = self.wavelength*self._k/(2*pi)

    @Beam.add_phase
    @Beam.add_amplitude
    def psi(self, pt):
        """ Wave's equation 'psi'.

        Args:
            pt (:obj:'Point'): point at which want to calculate 'psi'
                value.

        Returns:
            Wave's equation complex value of default plane wave decla-
                red on beam class.

        """
        return cm.exp(1j*self.k*pt.z)


class BesselBeam(PlaneWave):
    def __init__(self, **kwargs):
        PlaneWave.__init__(self)

        self.name = 'bessel-beam'

        self._krho = None
        self._kz = None
        self._theta = None
        self._order = 0

        self.params = ('_amplitude',
                       '_phase',
                       '_beta',
                       '_wavelength',
                       '_k',
                       '_nm',
                       '_krho',
                       '_kz',
                       '_theta',
                       '_order',)

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    @property
    def krho(self):
        return self._krho

    @krho.setter
    def krho(self, value):
        self._krho = value
        self._kz = ma.sqrt(self._k**2 - value**2)
        if self._kz != 0:
            self.theta = ma.atan(self._krho/self._kz)
        else:
            self.theta = pi/2

    @property
    def kz(self):
        return self._kz

    @kz.setter
    def kz(self, value):
        self._kz = value
        self._krho = ma.sqrt(self._k**2 - value**2)
        if self._kz != 0:
            self.theta = ma.atan(self._krho/self._kz)
        else:
            self.theta = pi/2

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self._krho = self._k*ma.sin(value)
        self._kz = self._k*ma.cos(value)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    @Beam.add_phase
    @Beam.add_amplitude
    def psi(self, pt):
        return (ss.jv(self._order, self._krho*pt.rho)
                * cm.exp(1j*self._kz*pt.z)
                * cm.exp(1j*self._order*pt.phi))


class GaussianBeam(PlaneWave):
    def __init__(self, **kwargs):
        PlaneWave.__init__(self)

        self.name = 'gaussian-beam'

        self._q = 1

        self.params = ('_amplitude',
                       '_phase',
                       '_beta',
                       '_wavelength',
                       '_k',
                       '_nm',
                       '_q',)

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value

    @Beam.add_phase
    @Beam.add_amplitude
    def psi(self, pt):
        return ((1/(1+1j*pt.z*2*self._q/self._k))
                * cm.exp(1j*pt.z*self._k)
                * cm.exp((-self._q*pt.rho**2)
                         / (1 + 1j*pt.z*2*self._q / self._k)))


class BesselGaussBeam(BesselBeam, GaussianBeam):
    def __init__(self, **kwargs):
        BesselBeam.__init__(self)
        GaussianBeam.__init__(self)

        self.name = 'bessel-gauss-beam'

        self.params = ('_amplitude',
                       '_phase',
                       '_beta',
                       '_wavelength',
                       '_k',
                       '_nm',
                       '_krho',
                       '_kz',
                       '_theta',
                       '_order',
                       '_q',)

        for key, value in kwargs.items():
            if hasattr(self, '_' + key):
                setattr(self, key, value)

    @Beam.add_phase
    @Beam.add_amplitude
    def psi(self, pt):
        if pt.z != 0:
            Q = self._q - 1j*self._k/(2*pt.z)
            num = 1j*self._k/(2*pt.z*Q)
            exp1 = cm.exp(1j*self._k*(pt.z+pt.rho**2/(2*pt.z)))
            bessel = ss.jv(0, num*self._krho*pt.rho)
            exp2 = cm.exp((self._krho**2+self._k**2*pt.rho**2/pt.z**2)
                          * (-1/(4*Q)))
            return -num*exp1*bessel*exp2
        else:
            return (ss.jv(0, self._krho*pt.rho)
                    * cm.exp(-self._q*pt.rho**2))


class BesselGaussBeamSuperposition(BesselGaussBeam):

    def __init__(self, **kwargs):
        BesselGaussBeam.__init__(self)

        self.name = 'bessel-gauss-beam-superposition'

        self.beams = []

        self._q = None
        self._N = None
        self._z_max = None
        self._R = None
        self._L = None
        self._qr = None

        self.params = ('_amplitude',
                       '_phase',
                       '_beta',
                       '_wavelength',
                       '_k',
                       '_nm',
                       '_krho',
                       '_kz',
                       '_theta',
                       '_order',
                       '_q',
                       '_N',
                       '_z_max',
                       '_R',
                       '_L',
                       '_qr',)

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
            out += '\n    ' + 'An: ' + str(beam.amplitude)
            out += '\n    ' + 'qn: ' + str(beam.q) + '\n'
        return out

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value
        self.__create_superposition()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value
        self.__create_superposition()

    @property
    def z_max(self):
        return self._z_max

    @z_max.setter
    def z_max(self, value):
        self._z_max = value

        if self._theta is None and self._R is not None:
            self.theta = ma.atan(self._R/self._z_max)

        if self._theta is not None and self._R is None:
            self._R = self._z_max*ma.tan(self._theta)

        self.__create_superposition()

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = value
        self.L = 3*value**2

        if self._theta is None and self._z_max is not None:
            self.theta = ma.atan(self._R/self._z_max)

        if self._theta is not None and self._z_max is None:
            self._z_max = self._R/ma.tan(self._theta)

        self.__create_superposition()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self._krho = self._k*ma.sin(value)
        self._kz = self._k*ma.cos(value)

        if self._z_max is None and self._R is not None:
            self._z_max = self._R/ma.tan(value)

        if self._z_max is not None and self._R is None:
            self._R = self._z_max*ma.tan(value)

        self.__create_superposition()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        self.qr = 6/value
        self.__create_superposition()

    @property
    def qr(self):
        return self._qr

    @qr.setter
    def qr(self, value):
        self._qr = value
        self.__create_superposition()

    def __create_superposition(self):
        if Beam.is_all_parameters_defined(self) is False:
            return

        def amplitude_n(n):
            arg = (self._qr - self._q - 2j*pi*n/self._L)*self._R**2
            den = self._L*(self._qr-self._q)/2 - 1j*pi*n
            return cm.sinh(arg)/den

        self.beams = []

        for i in range(2*self._N + 1):
            n_index = i - self._N
            beam = BesselGaussBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self._wavelength
            beam.nm = self._nm
            beam.krho = self._krho
            beam.q = self._qr - 1j*2*pi*n_index/self._L
            self.beams.append(beam)

    @Beam.add_phase
    @Beam.add_amplitude
    def psi(self, pt):
        return sum([beam.psi(pt) for beam in self.beams])


class FrozenWave(PlaneWave):

    def __init__(self, **kwargs):
        PlaneWave.__init__(self)

        self.name = 'frozen-wave'

        self.beams = []

        self._Q = None
        self._N = None
        self._L = None
        self.func = None

        self.params = ('_amplitude',
                       '_phase',
                       '_beta',
                       '_wavelength',
                       '_k',
                       '_nm',
                       '_theta',
                       '_order',
                       '_Q',
                       '_N',
                       '_L',)

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
            out += '\n    ' + 'An: ' + str(beam.amplitude)
            out += '\n    ' + 'k: ' + str(beam.k)
            out += '\n    ' + 'kzn: ' + str(beam.kz)
            out += '\n    ' + 'khron: ' + str(beam.krho) + '\n'
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
        self.__create_superposition()

    @property
    def ref_func(self):
        return self.func

    @ref_func.setter
    def ref_func(self, func):
        self.func = func
        self.__create_superposition()

    def __create_superposition(self):
        if (Beam.is_all_parameters_defined(self) is False
                or self.func is None):
            return

        def amplitude_n(n):
            func_real = lambda z: (self.func(z)
                                   * cm.exp(-2j*pi*z*n/self._L)).real
            func_imag = lambda z: (self.func(z)
                                   * cm.exp(-2j*pi*z*n/self._L)).imag
            an_real, err = quad(func_real, 0, self._L)
            an_imag, err = quad(func_imag, 0, self._L)
            return (an_real + 1j*an_imag)/self._L

        if 2*pi*self._N/self._L > self._k/2:
            error_msg = 'Combination of N, L and k does not '
            error_msg += 'satisfy Q range condition.'
            raise NameError(error_msg)

        if self._Q + 2*pi*self._N/self._L > self._k:
            msg = 'Q is too large. '
            msg += 'It was changed from %fk ' % (self._Q/self._k)
            self._Q = self._k - 2*pi*self._N/self._L
            msg += 'to %fk.' % (self._Q/self._k)
            print(msg)

        if self._Q - 2*pi*self._N/self._L < 0:
            msg = 'Q is too low. '
            msg += 'It was changed from %fk ' % (self._Q/self._k)
            self._Q = 2*pi*self._N/self._L
            msg += 'to %fk.' % (self._Q/self._k)
            print(msg)

        self.beams = []

        for i in range(2*self._N + 1):
            n_index = i - self._N
            beam = BesselBeam()
            beam.amplitude = amplitude_n(n_index)
            beam.wavelength = self._wavelength
            beam.nm = self._nm
            beam.kz = self._Q + 2*pi*n_index/self._L
            self.beams.append(beam)

    @Beam.add_phase
    @Beam.add_amplitude
    def psi(self, pt):
        return sum([beam.psi(pt) for beam in self.beams])


class Point:
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
                % (str(self.r), str(self.theta), str(self.phi)))

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
            self.theta = ma.acos(z / self.r)
        else:
            self.theta = 0.0

    def abs(self):
        return self.r

    def normalize(self):
        return [self.x/self.r, self.y/self.r, self.z/self.r]

if __name__ == "__main__":
    print("Please, visit: https://github.com/arantespp/opticalforces")

    b = BesselGaussBeamSuperposition(nm=1, 
                                     wavelength=1064e-9, 
                                     q=0,
                                     N=2)

    b.R = 1e-3
    b.beta = 2
    b.krho = 652125