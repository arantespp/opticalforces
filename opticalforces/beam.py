import cmath as cm
import math as ma
import numpy as np
import scipy.special as ss

from astropy.table import Table


# Speed of light.
c = 299792458


class Beam:
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

    # if two parameters in nm, wavelenght and k is defined and the
    # third one is not, this function define it.
    def _set_wavelength_group(self):

        # Verify if all parameters can be defined given currently para-
        # meters that was already defined at class instance. This loop
        # will stop if there is no change in any parameter, which
        # means, it cannot define a parameter anymore or all parameters
        # are already defined.
        while True:
            parametersChanged = False

            if (self.params['nm'] is not None
                    and self.params['wavelength'] is not None
                    and self.params['k'] is not None):
                pass

            elif (self.params['nm'] is not None
                  and self.params['wavelength'] is not None):
                self.params['k'] = (self.params['nm'] * 2*ma.pi
                                    / self.params['wavelength'])
                parametersChanged = True

            elif (self.params['nm'] is not None
                  and self.params['k'] is not None):
                self.params['wavelength'] = (self.params['nm']*2*ma.pi
                                             / self.params['k'])
                parametersChanged = True

            elif (self.params['wavelength'] is not None
                  and self.params['k'] is not None):
                self.params['nm'] = (self.params['wavelength']
                                     * self.params['k'] / (2*ma.pi))
                parametersChanged = True

            if parametersChanged is False:
                break

        # raise a error with all non defined parameters
        msg = ''
        for param in ['nm', 'wavelength', 'k']:
            if self.params[param] is None:
                msg += ' ' + param

        if msg != '':
            raise NameError('Cannot define' + msg + ' parameters')

    def __str__(self):
        return ('name:'
                + str(self.name)
                + '-params:'
                + str(self.params))

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

    def RS(self, pt, Rmax, Rmin=0, npRho=501, npPhi=3):
        """ Wave's Raleigh-Sommerfeld integral by 1/3 Simpson's Rule.

        Args:
            pt (:obj:'Point'): point at which want to calculate wave's
                Rayleigh-Sommerfeld complex value.
            Rmax: aperture radius that will be used to simulate the
                wave aperture truncate.
            npRho: rho's variable 1/3 Simpson's Rule discretization.
                a.k.a: Rho's number points.
            npPhi: phi's variable 1/3 Simpson's Rule discretization.
                a.k.a: Phi's number points.

        Returns:
            Raleigh-Sommerfeld complex value at a specific point 'pt'
                given a aperture with radius 'Rmax'.

        """

        # RS integral's integrand
        def R(rho, phi):
            return (pt - Point(rho, phi, 0, 'cilin')).r

        def E(rho, phi):
            return cm.exp(1j*self.params['k']*R(rho, phi))

        def integrand(rho, phi):
            return (rho*self.psi(Point(rho, phi, 0, 'cilin'))
                    *E(rho, phi) / R(rho, phi)**2)

        # S: 1/3 Simp's rule auxiliary matrix.
        auxrhovec = [1 if (rho == 0 or rho == npRho - 1) \
            else (4 if rho % 2 != 0 else 2) for rho in range(npRho)]

        auxphivec = [1 if (phi == 0 or phi == npPhi - 1) \
            else (4 if phi % 2 != 0 else 2) for phi in range(npPhi)]

        S = [[i * j for i in auxrhovec] for j in auxphivec]

        # F: 'integrand' mapped at discretized RS points.
        rhovec = np.linspace(Rmin, Rmax, npRho)

        phivec = np.linspace(0, 2*ma.pi, npPhi)

        F = [[pt.z*integrand(rho, phi) / (1j*self.params['wavelength'])
              for rho in rhovec] for phi in phivec]

        # Hadamard product between 'F' and 'S'
        H = sum(sum(np.multiply(F, S)))

        # Interval's discretization
        hrho = (Rmax - Rmin) / (npRho-1)
        hphi = 2*ma.pi / (npPhi-1)

        return hphi*hrho*H/9

    def RSI(self, pt, Rmax):
        """ Wave's Raleigh-Sommerfeld integral intensity.

        Args:
            pt (:obj:'Point'): point at which want to calculate wave's
                Rayleigh-Sommerfeld complex value.
            Rmax: aperture radius that will be used to simulate the
                wave aperture truncate.

        Returns:
            Raleigh-Sommerfeld intensity value at a specific point 'pt'
                given a aperture with radius 'Rmax'.

        """

        return abs(self.RS(pt, Rmax))**2

    ''' l '''
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

        group0 = ('krho', 'spot')
        group1 = ('k', 'krho', 'kz', 'theta')

        # Verify if all parameters can be defined given currently para-
        # meters that was already defined at class instance. This loop
        # will stop if there is no change in any parameter, which
        # means, it cannot define a parameter anymore or all parameters
        # are already defined.
        while True:
            parametersChanged = False

            # group 0
            if (self.params['krho'] is not None
                    and self.params['spot'] is not None):
                pass

            elif self.params['krho'] is not None:
                self.params['spot'] = (ss.jn_zeros(
                    self.params['order'], 1)[0]/self.params['krho'])
                parametersChanged = True

            elif self.params['spot'] is not None:
                self.params['krho'] = (ss.jn_zeros(
                    self.params['order'], 1)[0]/self.params['spot'])
                parametersChanged = True

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
                parametersChanged = True

            elif (self.params['k'] is not None
                  and self.params['kz'] is not None):
                self.params['theta'] = (ma.acos(self.params['kz']
                                                / self.params['k']))
                self.params['krho'] = (ma.sqrt(self.params['k']**2
                                               - self.params['kz']**2))
                parametersChanged = True

            elif (self.params['k'] is not None
                  and self.params['theta'] is not None):
                self.params['krho'] = (self.params['k']
                                       * ma.sin(self.params['theta']))
                self.params['kz'] = (self.params['k']
                                     * ma.cos(self.params['theta']))
                parametersChanged = True

            elif (self.params['krho'] is not None
                  and self.params['kz'] is not None):
                self.params['k'] = (ma.sqrt(self.params['krho']**2
                                            + self.params['kz']**2))
                self.params['theta'] = (ma.atan(self.params['krho']
                                                / self.params['kz']))
                parametersChanged = True

            elif (self.params['krho'] is not None
                  and self.params['theta'] is not None):
                self.params['k'] = (self.params['krho']
                                    / ma.cos(self.params['theta']))
                self.params['kz'] = (self.params['krho']
                                     / ma.tan(self.params['theta']))
                parametersChanged = True

            elif (self.params['kz'] is not None
                  and self.params['theta'] is not None):
                self.params['k'] = (self.params['kz']
                                    / ma.sin(self.params['theta']))
                self.params['krho'] = (self.params['z']
                                       * ma.tan(self.params['theta']))
                parametersChanged = True

            if parametersChanged is False:
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

        self.vec_An = []
        for i in range(2*self.params['N']+1):
            self.vec_An.append(self.An(i-self.params['N']))

    def An(self, n):
        arg = (self.params['qr'] - self.params['q']
               - 2j*ma.pi*n / self.params['L'])*(self.params['R']**2)
        den = (self.params['L']*(self.params['qr']-self.params['q'])/2
               - 1j*ma.pi*n)
        return cm.sinh(arg)/den

    def Qn(self, n, point):
        return (self.params['qr'] - 2j*ma.pi*n/self.params['L']
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
                return (B()*E()*self.vec_An[n+self.params['N']]
                        /self.Qn(n, point))

            elif point.z == 0:
                arg = (-point.rho**2
                       *(self.params['qr']-2j*ma.pi*n/self.params['L']))
                return self.self.vec_An[n+self.params['N']]*cm.exp(arg)

            else:
                return 0

        return M()*sum(map(S, i))


class Particle:
    def __init__(self, **kwargs):
        self.params = {}
        for key in kwargs:
            self.params[key] = kwargs[key]

    def __str__(self):
        return 'Particle params: ' + str(self.params)


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
            self.phi += ma.pi if x <= 0 and y >= 0 else 0
            self.phi -= ma.pi if x <= 0 and y < 0 else 0
        else:
            if self.y < 0:
                self.phi = -ma.pi/2
            elif self.y == 0:
                self.phi = 0
            else:
                self.phi = ma.pi/2

        # spherical
        self.r = ma.sqrt(x**2 + y**2 + z**2)
        if self.r != 0:
            self.theta = ma.acos(z / self.r)
        else:
            self.theta = 0


# Round a number to 'sig' significatives figures.
'''def round_sig(num, sig=4):
    if num < 0:
        num = -num
        return -round(num, sig-int(ma.floor(ma.log10(num)))-1)
    elif num > 0:
        return +round(num, sig-int(ma.floor(ma.log10(num)))-1)
    # num == 0
    else:
        return num'''

'''
    Main call, just for tests porpouse
'''
if __name__ == "__main__":
    import time

    wl = 632.8 * 10 ** -9
    k = 9.93 * 10 ** 6
    krho = 4.07 * 10 ** 4
    R = 3.5 * 10 ** -3

    beam = Beam()
    # ideal gaussiam beam
    igb = IGB(nm=1, wavelength=wl, q=0)
    # ideal bessel beam
    ibb = IBB(order = 0, nm=1, wavelength=wl, krho=krho)
    # bessel gauss beam superpositon
    bgbs = BGBS(igb, ibb, N=23, R=R)

    #ptc = Particle(Rp=10e-6, np=1.6, alphap=0.5e6)

    #t0 = time.time()
    print(bgbs.RS(Point(0, 0, 1), 10e-3, 0.5e-3))
    #bbs = BBS(lambda z: z)
    #print(bbs.ref(2))
    #print(time.time()-t0)

    '''vecz = np.linspace(0.95 * bgbs.params['Zmax'], 1 * bgbs.params['Zmax'], 501)

    listPositions = list(map(lambda z: Point(0, 0, z), vecz))

    rsi = [bgbs.psi(particlePosition).real for particlePosition in listPositions]

    plt.figure(1)

    plt.plot(vecz, rsi, '-')
    plt.axvline(x=bgbs.params['Zmax'], color='r', linestyle='--', linewidth=0.75)
    plt.grid()

    plt.show()'''


    '''vecz = np.linspace(0.15 * bgbs.params['Zmax'], 1.25 * bgbs.params['Zmax'], 501)

    listPositions = list(map(lambda z: Point(0, 0, z), vecz))

    rsi = [ibb.RSI(particlePosition, bgbs.params['R']) for particlePosition in listPositions]

    #print(ibb.RS(Point(0,0,1), bgbs.params['R']))

    plt.figure(1)

    plt.subplot(211)
    plt.plot(vecz, [bgbs.I(Point(0, 0, z)) for z in vecz], '-')
    plt.axvline(x=bgbs.params['Zmax'], color='r', linestyle='--', linewidth=0.75)
    plt.grid()

    plt.subplot(212)
    plt.plot(vecz, rsi, '-')
    plt.axvline(x=bgbs.params['Zmax'], color='r', linestyle='--', linewidth=0.75)
    plt.grid()

    plt.show()'''
