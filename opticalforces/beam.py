import math as ma
from math import pi
import cmath as cm
import numpy as np
import scipy.special as ss
from functools import wraps

class Beam(object):
    def __init__(self):
        self.beams = [self]
        self._name = 'generic-beam'

    def __init__(self, beams, name='generic-beam'):
        self.beams = beams
        self._name = name

    def __str__(self):
        out = 'name: ' + self._name + '\n'
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
        psi_x = np.dot(fdc, psix) / den
        psi_y = np.dot(fdc, psiy) / den
        psi_z = np.dot(fdc, psiz) / den

        # k0 components
        psi = self.psi(pt)
        k0x = (psi_x / psi).imag
        k0y = (psi_y / psi).imag
        k0z = (psi_z / psi).imag

        # normalize k0 vector
        if k0x != 0 or k0y != 0 or k0z != 0:
            k = [k0x, k0y, k0z]
            return k / np.linalg.norm(k)
        else:
            return [0, 0, 1]


class PlaneWave(Beam):
    def __init__(self):
        self.beams = [self]

        self._name = 'plane-wave'
        self._amplitude = 1
        self._phase = 0
        self._wavelength = None
        self._k = None
        self._nm = None

    def __str__(self):
        out = ''
        for p, v in self.__dict__.items():
            if p[0] == '_':
                out += '    '
                out += p + ': ' + str(v)
                out += '\n'
        return out[:]

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
        return self._k

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
    def __init__(self):
        PlaneWave.__init__(self)

        self._name = 'bessel-beam'

        self._krho = None
        self._kz = None
        self._theta = None
        self._order = 0

    @property
    def krho(self):
        return self._krho

    @krho.setter
    def krho(self, value):
        self._krho = value
        self._kz = ma.sqrt(self._k**2 - value**2)
        self._theta = ma.atan(self._krho/self._kz)

    @property
    def kz(self):
        return self._kz

    @krho.setter
    def kz(self, value):
        self._kz = value
        self._krho = ma.sqrt(self._k**2 - value**2)
        self._theta = ma.atan(self._krho/self._kz)

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
    def __init__(self):
        PlaneWave.__init__(self)

        self._name = 'gaussian-beam'

        self._q = 1

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
    def __init__(self):
        BesselBeam.__init__(self)
        GaussianBeam.__init__(self)

        self._name = 'bessel-gauss-beam'

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
    
    def __init__(self):
        BesselGaussBeam.__init__(self)

        self._name = 'bessel-gauss-beam-superposition'

        self.beams = []

        self._q = None
        self._N = None
        self._z_max = None

        self.R_DEFAULT = 1e-3
        self.L_DEFAULT = 3*self.R_DEFAULT**2
        self.QR_DEFAULT = 6/self.L_DEFAULT

        self._R = self.R_DEFAULT
        self._L = self.L_DEFAULT
        self._qr = self.QR_DEFAULT

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

        if Beam.is_all_parameters_defined(self) is True:
            self.__create_superposition()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value

        if Beam.is_all_parameters_defined(self) is True:
            self.__create_superposition()

    @property
    def z_max(self):
        return self._z_max

    @z_max.setter
    def z_max(self, value):
        self._z_max = value

        if self._theta is None and self._R is not None:
            self.theta = ma.atan(self._R/self._z_max)

        if self._theta is not None and self._R == self.R_DEFAULT:
            self._R = self._z_max*ma.tan(self._theta)

        if Beam.is_all_parameters_defined(self) is True:
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

        if Beam.is_all_parameters_defined(self) is True:
            self.__create_superposition()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self._krho = self._k*ma.sin(value)
        self._kz = self._k*ma.cos(value)

        if self._z_max is None:
            self._z_max = self._R/ma.tan(value)

        if self._z_max is not None and self._R == self.R_DEFAULT:
            self._R = self._z_max*ma.tan(value)

        if Beam.is_all_parameters_defined(self) is True:
            self.__create_superposition()

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        self.qr = 6/value

        if Beam.is_all_parameters_defined(self) is True:
            self.__create_superposition()

    @property
    def qr(self):
        return self._qr

    @qr.setter
    def qr(self, value):
        self._qr = value

        if Beam.is_all_parameters_defined(self) is True:
            self.__create_superposition()

    def __create_superposition(self):
        def amplitude_n(n):
            arg = (self._qr - self._q - 2j*pi*n/self._L)*self._R**2
            den = self._L*(self._qr-self._q)/2 - 1j*pi*n
            return cm.sinh(arg)/den

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




'''class Beam:
    """Generic scope for a beam in cilindrical coordinates. By default,
    this class is a plane wave.

    All beam's properties, like 'psi', intensity, RS (Rayleigh-Sommer-
    feld integral), forces etc are declared here.

    """
    def __init__(self):
        self.name = 'plane-wave'

        self.params = {
            # default
            'nm': 1,
            'wavelength': 1064e-9,
            'k': 2*cm.pi/1064e-9
        }

    # if two parameters in nm, wavelength and k is defined and the
    # third one is not, this function define it.
    def _set_wavelength_group(self):

        # Verify if all parameters can be defined given currently para-
        # meters that was already defined at class instance. This loop
        # will stop if there is no change in any parameter, which
        # means, it cannot define a parameter anymore or all parameters
        # are already defined.
        while True:
            params_changed = False

            if (self.params['nm'] is not None
                    and self.params['wavelength'] is not None
                    and self.params['k'] is not None):
                pass

            elif (self.params['nm'] is not None
                  and self.params['wavelength'] is not None):
                self.params['k'] = (self.params['nm'] * 2*pi
                                    / self.params['wavelength'])
                params_changed = True

            elif (self.params['nm'] is not None
                  and self.params['k'] is not None):
                self.params['wavelength'] = (self.params['nm']*2*pi
                                             / self.params['k'])
                params_changed = True

            elif (self.params['wavelength'] is not None
                  and self.params['k'] is not None):
                self.params['nm'] = (self.params['wavelength']
                                     * self.params['k'] / (2*pi))
                params_changed = True

            if params_changed is False:
                break

        # raise a error with all non defined parameters
        msg = ''
        for param in ['nm', 'wavelength', 'k']:
            if self.params[param] is None:
                msg += ' ' + param

        if msg != '':
            raise NameError('Cannot define' + msg + ' parameters')

    def __str__(self):
        out = 'name: ' + str(self.name) + '\n'
        out += 'parameters:\n'
        for param in self.params:
            out += '    ' + param + ': ' + str(self.params[param])
            out += '\n'
        return out

    def psi(self, pt):
        """ Wave's equation 'psi'.

        Args:
            pt (:obj:'Point'): point at which want to calculate 'psi'
                value.

        Returns:
            Wave's equation complex value of default plane wave decla-
                red on beam class.

        """
        return cm.exp(1j*self.params['k']*pt.z)

    def intensity(self, pt):
        """ Wave's intensity.

        Args:
            pt (:obj:'Point'): point at which want to calculate wave's
                intensity, that is psi's abs squared.

        Returns:
            Wave's intensity.

        """
        return abs(self.psi(pt))**2

    def RS(self, pt, r_max, r_min=0, **kwargs):
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
        np_rho = 501
        np_phi = 3

        for key in kwargs:
            if key == 'np_rho':
                np_rho = kwargs[key]
            elif key == 'np_phi':
                np_phi = kwargs[key]
            else:
                pass

        # RS integral's integrand
        def R(rho, phi):
            return (pt - Point(rho, phi, 0, 'cilin')).abs()

        def integrand(rho, phi):
            return (rho*self.psi(Point(rho, phi, 0, 'cilin'))
                    * cm.exp(1j*self.params['k']*R(rho, phi))
                    / R(rho, phi)**2)

        # S: 1/3 Simp's rule auxiliary matrix.
        def matrix_s():
            auxrhovec = [1 if (r == 0 or r == np_rho - 1) \
                else (4 if r % 2 != 0 else 2) for r in range(np_rho)]

            auxphivec = [1 if (p == 0 or p == np_phi - 1) \
                else (4 if p % 2 != 0 else 2) for p in range(np_phi)]

            return [[i * j for i in auxrhovec] for j in auxphivec]

        # F: 'integrand' mapped at discretized RS points.
        def matrix_f():
            rhovec = np.linspace(r_min, r_max, np_rho)

            phivec = np.linspace(0, 2*pi, np_phi)

            return [[(pt.z*integrand(rho, phi)
                      / (1j*self.params['wavelength']))
                     for rho in rhovec] for phi in phivec]

        # Hadamard product between 'F' and 'S'
        H = sum(sum(np.multiply(matrix_f(), matrix_s())))

        # Interval's discretization
        hrho = (r_max - r_min) / (np_rho-1)
        hphi = 2*pi / (np_phi-1)

        return hphi*hrho*H/9

    def RSI(self, pt, r_max):
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

        return abs(self.RS(pt, r_max))**2


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
        den = 60 * h

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
        psi_x = np.dot(fdc, psix) / den
        psi_y = np.dot(fdc, psiy) / den
        psi_z = np.dot(fdc, psiz) / den

        # k0 components
        psi = self.psi(pt)
        k0x = (psi_x / psi).imag
        k0y = (psi_y / psi).imag
        k0z = (psi_z / psi).imag

        # normalize k0 vector
        if k0x != 0 or k0y != 0 or k0z != 0:
            k = [k0x, k0y, k0z]
            return k / np.linalg.norm(k)
        else:
            return [0, 0, 1]

class IGB(Beam):
    """Scope for a Gaussian beam in cilindrical coordinates.

    This class is a subclass of beam class. Only 'psi' is changed here
    in comparation with its superclass.

    """

    def __init__(self, **kwargs):
        self.name = 'igb'

        self.params = {
            'nm': None,
            'wavelength': None,
            'k': None,
            'q': None
        }

        for key in kwargs:
            self.params[key] = kwargs[key]

        super()._set_wavelength_group()

        self.__set_all()

    # set all parameters
    def __set_all(self):
        assert self.params['q'] is not None, \
            ('Gaussian beam parameter q not defined')

    def psi(self, point):
        return ((1/(1+1j*point.z*2*self.params['q']/self.params['k']))
                * cm.exp(1j*point.z*self.params['k'])
                * cm.exp((-self.params['q']*point.rho**2)
                         / (1 + 1j*point.z*2*self.params['q']
                            / self.params['k'])))

class IBB(Beam):
    """Scope for a ideal Bessel beam in cilindrical coordinates.

    This class is a subclass of beam class. Only 'psi' is changed here
    in comparation with its superclass.

    """

    def __init__(self, **kwargs):
        # default
        self.name = 'ibb'

        self.params = {
            'order': None,
            'nm': None,
            'wavelength': None,
            'theta': None,
            'k': None,
            'krho': None,
            'kz': None,
            'spot': None
        }

        for key in kwargs:
            self.params[key] = kwargs[key]

        super()._set_wavelength_group()

        self.__set_all()

    # set all parameters
    def __set_all(self):
        assert self.params['order'] is not None, \
            ('Bessel beam order not defined')

        # Verify if all parameters can be defined given currently para-
        # meters that was already defined at class instance. This loop
        # will stop if there is no change in any parameter, which
        # means, it cannot define a parameter anymore or all parameters
        # are already defined.
        while True:
            params_changed = False

            # group 0
            if (self.params['krho'] is not None
                    and self.params['spot'] is not None):
                pass

            elif self.params['krho'] is not None:
                self.params['spot'] = (ss.jn_zeros(
                    self.params['order'], 1)[0]/self.params['krho'])
                params_changed = True

            elif self.params['spot'] is not None:
                self.params['krho'] = (ss.jn_zeros(
                    self.params['order'], 1)[0]/self.params['spot'])
                params_changed = True

            # group 1
            if (self.params['k'] is not None
                    and self.params['krho'] is not None
                    and self.params['kz'] is not None
                    and self.params['theta'] is not None):
                pass

            elif (self.params['k'] is not None
                  and self.params['krho'] is not None):
                self.params['theta'] = (ma.asin(self.params['krho']
                                                / self.params['k']))
                self.params['kz'] = (ma.sqrt(self.params['k']**2
                                             - self.params['krho']**2))
                params_changed = True

            elif (self.params['k'] is not None
                  and self.params['kz'] is not None):
                self.params['theta'] = (ma.acos(self.params['kz']
                                                / self.params['k']))
                self.params['krho'] = (ma.sqrt(self.params['k']**2
                                               - self.params['kz']**2))
                params_changed = True

            elif (self.params['k'] is not None
                  and self.params['theta'] is not None):
                self.params['krho'] = (self.params['k']
                                       * ma.sin(self.params['theta']))
                self.params['kz'] = (self.params['k']
                                     * ma.cos(self.params['theta']))
                params_changed = True

            elif (self.params['krho'] is not None
                  and self.params['kz'] is not None):
                self.params['k'] = (ma.sqrt(self.params['krho']**2
                                            + self.params['kz']**2))
                self.params['theta'] = (ma.atan(self.params['krho']
                                                / self.params['kz']))
                params_changed = True

            elif (self.params['krho'] is not None
                  and self.params['theta'] is not None):
                self.params['k'] = (self.params['krho']
                                    / ma.cos(self.params['theta']))
                self.params['kz'] = (self.params['krho']
                                     / ma.tan(self.params['theta']))
                params_changed = True

            elif (self.params['kz'] is not None
                  and self.params['theta'] is not None):
                self.params['k'] = (self.params['kz']
                                    / ma.sin(self.params['theta']))
                self.params['krho'] = (self.params['z']
                                       * ma.tan(self.params['theta']))
                params_changed = True

            if params_changed is False:
                break

        msg = ''
        for param in self.params:
            if self.params[param] is None:
                msg += ' ' + param
        # raise a error with all non defined parameters
        if msg != '':
            raise NameError('Cannot define' + msg + ' parameters')

    def psi(self, point):
        return (ss.jv(self.params['order'],
                      self.params['krho']*point.rho)
                * cm.exp(1j*self.params['kz']*point.z)
                * cm.exp(1j * self.params['order'] * point.phi))

class BGBS(Beam):
    """Scope for a Bessel-Gauss beam superposition beam in cilindrical
    coordinates.

    This class is a subclass of beam class. Only 'psi' is changed here
    in comparation with its superclass.

    """

    def __init__(self, igb, ibb, N=0, **kwargs):
        assert igb.params['wavelength'] == ibb.params['wavelength'], \
            ('IBG and IBB wavelengths are differents')

        assert igb.params['nm'] == ibb.params['nm'], \
            ('IBG and IBB nm are differents')

        assert igb.params['k'] == ibb.params['k'], \
            ('IBG and IBB k are differents')

        self.name = "bgbs"

        R_default = 10 ** -3
        Zmax_default = R_default / ma.tan(ibb.params['theta'])
        L_default = 3 * R_default ** 2
        qr_default = 6 / L_default

        self.params = {
            # ideal bessel beam parameters
            'wavelength': ibb.params['wavelength'],
            'nm': ibb.params['nm'],
            'k': ibb.params['k'],
            'krho': ibb.params['krho'],
            'kz': ibb.params['kz'],
            'theta': ibb.params['theta'],
            # ideal gaussian beam parameters
            'q': igb.params['q'],
            # self parameters default
            'N': N,
            'R': R_default, # 1mm default
            'Zmax': Zmax_default,
            'L': L_default,
            'qr': qr_default
        }

        for key in kwargs:
            self.params[key] = kwargs[key]
            if key == 'R':
                self.params['Zmax'] = (self.params['R']
                                       / ma.tan(ibb.params['theta']))
            if key == 'Zmax':
                self.params['R'] = (self.params['Zmax']
                                    * ma.tan(ibb.params['theta']))

        # R different and L equally from default, means that a new R
        # was defined and L was not changed, so we need to set L's new
        # value
        if (self.params['R'] != R_default
                and self.params['L'] == L_default):
            self.params['L'] = 3 * self.params['R'] ** 2

        # R different and qr equally from default, means that a new R
        # was defined and qr was not changed, so we need to set qr's
        # new value
        if (self.params['R'] != R_default
                and self.params['qr'] == qr_default):
            self.params['qr'] = 6 / self.params['L']

        self.vec_an = []
        for i in range(2*self.params['N']+1):
            self.vec_an.append(self.An(i-self.params['N']))

    def An(self, n):
        arg = (self.params['qr'] - self.params['q']
               - 2j*pi*n / self.params['L'])*(self.params['R']**2)
        den = (self.params['L']*(self.params['qr']-self.params['q'])/2
               - 1j*pi*n)
        return cm.sinh(arg)/den

    def Qn(self, n, point):
        return (self.params['qr'] - 2j*pi*n/self.params['L']
                - 1j*self.params['k']/(2*point.z))

    def psi(self, point):
        i = np.arange(-self.params['N'], self.params['N'] + 1)

        def M():
            if point.z != 0:
                return (-1j*self.params['k']/(2*point.z)
                        * cm.exp(1j*self.params['k']
                                 * (point.z+point.rho**2/(2*point.z))))
            elif point.z == 0:
                return ss.jv(0, self.params['krho'] * point.rho)
            else:
                return 0

        def S(n):
            def B():
                return (ss.jv(0, 1j * self.params['k']
                              * self.params['krho'] * point.rho
                              / (2*point.z * self.Qn(n, point))))

            def E():
                arg = (-((self.params['krho'] ** 2)
                         + (self.params['k']*point.rho/point.z)**2)
                       / (4 * self.Qn(n, point)))
                return cm.exp(arg)

            if point.z != 0:
                return (B()*E()*self.vec_an[n+self.params['N']]
                        /self.Qn(n, point))

            elif point.z == 0:
                arg = (-point.rho**2
                       *(self.params['qr']-2j*pi*n/self.params['L']))
                return self.vec_an[n+self.params['N']]*cm.exp(arg)

            else:
                return 0

        return M()*sum(map(S, i))

'''

class Point:
    def __init__(self, v1, v2, v3, *system):
        if 'cart' in system:
            self.__init(v1, v2, v3)

        elif 'cilin' in system:
            self.__init(v1*ma.cos(v2), v1*ma.sin(v2), v3)

        elif 'spher' in system:
            self.__init(v1*ma.sin(v2)*ma.cos(v3),
                        v1*ma.sin(v2)*ma.sin(v3),
                        v1*ma.cos(v2))

        else:
            self.__init(v1, v2, v3)

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
        return [self.x/self.r, self.z/self.r, self.z/self.r]

if __name__ == "__main__":
    #print("Please, visit: https://github.com/arantespp/opticalforces")

    b = PlaneWave()
    b.nm = 1
    b.k = 5
    b.amplitude = 3

    c = BesselBeam()
    c.nm = 4
    c.order = 3
    c.k = 100
    c.phase = 0.99
    c.krho = 23
    c.theta = 0.23207768

    a = BesselGaussBeamSuperposition()
    a.wavelength = 1064e-9
    a.nm = 1
    #a.krho = a.k/ma.sqrt(2)
    a.R = 1e-3
    a.z_max = 10e-3
    a.N = 2
    a.q = 0

    d = b + c + c + b + b
    d += d

    print(a)
    #print(a.psi(Point(0,0,0)))
    #print(c.intensity(Point(1,2,3)))
    #print(d)
