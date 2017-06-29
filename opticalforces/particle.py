import math as ma
from math import pi
import cmath as cm
from numbers import Number
from functools import wraps
import time
from scipy.integrate import quad, dblquad
import copy
import pandas as pd
import numpy as np
import os

from beam import Point


# Speed of light.
SPEED_OF_LIGHT = 299792458

k0time = []


def round_sig(num, sig=4):
    if isinstance(num, Number):
        if num < 0:
            num = -num
            return -round(num, sig-int(ma.floor(ma.log10(num)))-1)
        elif num > 0:
            return +round(num, sig-int(ma.floor(ma.log10(num)))-1)
        else:
            return num
    elif isinstance(num, list):
        return [round_sig(element) for element in num]


def check_geo_opt_database(func):
    @wraps(func)
    def wrapped(self, beam, beam_pos, force_dir, force_type='total',
                paramx=None, paramy=None, epsrel=5e-2):

        database_name = '_'.join([self.name, beam.name, 'geo-opt.pkl'])
        database_dir = 'database'
        database_full_path = os.path.join(database_dir, database_name)

        if not os.path.isdir(database_dir):
            os.makedirs(database_dir)

        if len(beam_pos) == 3:
            beam_pos = Point(beam_pos[0], beam_pos[1], beam_pos[2])
        elif len(beam_pos) == 4:
            beam_pos = Point(beam_pos[0], beam_pos[1], beam_pos[2], beam_pos[3])
        else:
            raise ValueError('beam_pos parameters wrong')

        # params without force
        params = {'beam_pos_x': round_sig(beam_pos.x),
                  'beam_pos_y': round_sig(beam_pos.y),
                  'beam_pos_z': round_sig(beam_pos.z),
                  'force_type': force_type,
                  'epsrel': epsrel,}

        for param in self.params:
            params.update({param[1:]: round_sig(self.__dict__[param])})

        def get_force_from_params(params):
            # load a dataframe or create if it doesn't exist
            if os.path.isfile(database_full_path):
                df = pd.read_pickle(database_full_path)
            else:
                full_params = params.copy()
                full_params.update({'fx': np.nan,
                                    'fy': np.nan,
                                    'fz': np.nan,})
                df = pd.DataFrame.from_dict([full_params])

            dff = df[(df[list(params)] == pd.Series(params)).all(axis=1)]

            # save old values
            old_ptc = copy.copy(self)

            # change particle parameters to new values
            for param in self.params:
                setattr(self, param, params[param[1:]])

            # change position parameters to new values
            _beam_pos = Point(params['beam_pos_x'],
                              params['beam_pos_y'],
                              params['beam_pos_z'],)

            if dff.empty:
                force = func(self, beam, _beam_pos, force_dir, force_type,
                             epsrel)
                _params = params.copy()
                _params[force_dir] = force
                df = df.append(_params, ignore_index=True)
                df.to_csv(database_full_path[:-3] + 'csv', index=False)
                df.to_pickle(database_full_path)
            else:
                if np.isnan(dff.loc[dff.index[0], force_dir]):
                    force = func(self, beam, _beam_pos, force_dir, force_type,
                                 epsrel)
                    df = df.set_value(dff.index[0], force_dir, force)
                    df.to_csv(database_full_path[:-3] + 'csv', index=False)
                    df.to_pickle(database_full_path)
                else:
                    force = dff.loc[dff.index[0], force_dir]

            # change particle parameters to old ones
            for param in self.params:
                setattr(self, param, old_ptc.__dict__[param])

            return force

        def get_force_from_range(params, paramx):
            _params = params.copy()
            param = paramx['param']
            del _params[param]

            _df = pd.DataFrame(columns=[param, force_dir])

            if os.path.isfile(database_full_path):
                df = pd.read_pickle(database_full_path)

                # select all rows that have the same values as '_params'
                df = df[(df[list(_params)] == pd.Series(_params)).all(axis=1)]

                # select all rows that have 'force_dir' as a finite number
                df = df[pd.notnull(df[force_dir])]

                # select all rows that have its parameter 'param' in range
                df = df[(df[param] >= paramx['start'])
                        & (df[param]<=paramx['stop'])]

                # fill '_df' columns with 'df' values
                _df[param], _df[force_dir] = df[param], df[force_dir]

            # add 'start' and 'stop' values to '_df'
            if not any(_df[param] == paramx['start']):
                new_row = {param: paramx['start'], force_dir: np.nan}
                _df = _df.append(new_row, ignore_index=True)

            if not any(_df[param] == paramx['stop']):
                new_row = {param: paramx['stop'], force_dir: np.nan}
                _df = _df.append(new_row, ignore_index=True)

            def interval(row):
                if row.name + 1 < _df.shape[0]:
                    return _df[param][row.name+1] - _df[param][row.name]
                return 0

            def next_param(row):
                if row.name + 1 < _df.shape[0]:
                    return (_df[param][row.name+1] + _df[param][row.name])/2
                return 0

            # add rows until params reach a interval minimal
            while True:
                _df = _df.sort_values([param]).reset_index(drop=True)
                _df['interval'] = _df.apply(interval, axis=1)
                _df['next_param'] = _df.apply(next_param, axis=1)

                inter_max = (paramx['stop']-paramx['start'])/(paramx['num']-1)

                if _df['interval'].max() <= inter_max:
                    break

                new_row = {param: _df['next_param'][_df['interval'].idxmax()],
                           force_dir: np.nan,}

                _df = _df.append(new_row, ignore_index=True)

            # calculate force of every 'force_dir' that is 'np.nan'
            if _df[force_dir].isnull().values.any():
                nnan_init = _df[force_dir].isnull().sum()

                _params = params.copy()

                for idx in range(_df.shape[0]):

                    if pd.notnull(_df[force_dir][idx]):
                        continue

                    _params[param] = _df[param][idx]

                    nnan_cur = _df[force_dir].isnull().sum()

                    print('%d/%d'% (nnan_init-nnan_cur+1, nnan_init))
                    print('start: ' + str(time.strftime('%d %b %Y %H:%M:%S',
                                                        time.localtime())))

                    time0 = time.time()

                    _df[force_dir][idx] = get_force_from_params(_params)

                    print('time: ' + str(time.time() - time0) + '\n')

            return _df[param].values.tolist(), _df[force_dir].values.tolist()

        if paramx is None:
            return None, get_force_from_params(params)
        else:
            if not paramx['param'] in params:
                raise ValueError('param in paramx does not exist.')

            return get_force_from_range(params, paramx)

    return wrapped


class SphericalParticle(object):
    params = ('_radius',
              '_refractive_index',
              '_medium_refractive_index',
              '_absorption_coefficient',)

    def __init__(self, **kwargs):
        self.name = 'spherical-particle'

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
            return [ma.sin(theta)*ma.cos(phi),
                    ma.sin(theta)*ma.sin(phi),
                    ma.cos(theta)]

    @staticmethod
    def ortonormal_ray_direction(ray_direction, normal):
        dot = np.dot(ray_direction, normal)
        if dot == 0:
            return normal

        d0 = [n-k for n, k in zip(normal, [dot*k for k in ray_direction])]

        if np.linalg.norm(d0) == 0:
            return [0, 0, 0]

        return [d/np.linalg.norm(d0) for d in d0]

    @staticmethod
    def incident_angle(ray_direction, normal):
        return ma.acos(-np.dot(ray_direction, normal))

    @staticmethod
    def refracted_angle(incident_angle, medium_refractive_index,
                        particle_refractive_index):
        return cm.asin(medium_refractive_index*ma.sin(incident_angle)
                       /particle_refractive_index).real

    @staticmethod
    def crossing_angle(ray_direction, normal, electric_field):
        """ Crossing angle between the polarization direction of the
        incident beam and the normal vector of the incident plane."""

        plane_normal = np.cross(ray_direction, normal)
        if np.linalg.norm(plane_normal) == 0:
            return 0

        div = (np.dot(electric_field, plane_normal)
               /np.linalg.norm(plane_normal))

        if abs(div) >= 1:
            return 0
        else:
            return ma.acos(abs(div))

    @staticmethod
    def parallel_reflectivity(incident_angle, refracted_angle):
        if incident_angle == 0:
            nrel = (self._refractive_index/self._medium_refractive_index)
            return (n-1)/(n+1)

        if refracted_angle == pi/2:
            return 1

        return (ma.tan(incident_angle - refracted_angle)
                / ma.tan(incident_angle + refracted_angle))**2

    @staticmethod
    def perpendicular_reflectivity(incident_angle, refracted_angle):
        if incident_angle == 0:
            nrel = (self._refractive_index/self._medium_refractive_index)
            return -(n-1)/(n+1)

        if refracted_angle == pi/2:
            return 1

        return (ma.sin(incident_angle - refracted_angle)
                / ma.sin(incident_angle + refracted_angle))**2

    @classmethod
    def reflectivity(cls, incident_angle, refracted_angle, crossing_angle):
        Rpa = cls.parallel_reflectivity(incident_angle, refracted_angle)
        Rpe = cls.perpendicular_reflectivity(incident_angle, refracted_angle)
        return Rpa*ma.sin(crossing_angle)**2 + Rpe*ma.cos(crossing_angle)**2

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
        return Tpa*ma.sin(crossing_angle)**2 + Tpe*ma.cos(crossing_angle)**2

    def Qt(self, incident_angle, refracted_angle, reflectivity, trasmissivity,
           force_type):

        length = 2*self._radius*ma.cos(refracted_angle)
        internal_attenuation = ma.exp(-self._absorption_coefficient*length)

        if force_type == 'total':
            return (1 + reflectivity*cm.exp(-2j*incident_angle)
                    - trasmissivity**2*internal_attenuation
                    * cm.exp(-2j*(incident_angle-refracted_angle))
                    / (1+reflectivity*internal_attenuation
                       * cm.exp(+2j*refracted_angle)))
        elif force_type == 'incident':
            return 1
        elif force_type == 'reflection':
            return reflectivity*cm.exp(-2j*incident_angle)
        elif force_type == 'transmission':
            return (- trasmissivity**2*internal_attenuation
                    * cm.exp(-2j*(incident_angle-refracted_angle))
                    / (1+reflectivity*internal_attenuation
                       * cm.exp(+2j*refracted_angle)))

    @check_geo_opt_database
    def geo_opt_force(self, beam, beam_pos, force_dir, force_type, epsrel):
        """ Force that beam causes in a spherical particle in a deter-
        mined position in geometrical optics regime (particle radius is
        greater than 10 times beam's wavelenght).

        We used this paper to do the code:
        Zhang, Yanfeng, et al. "Influence of absorption on optical
        trapping force of spherical particles in a focused Gaussian
        beam." Journal of Optics A: Pure and Applied Optics 10.8
        (2008): 085001.
        """

        def dforce(theta, phi):
            # Beam particle surface: beam coordinates point that
            # match the point at theta and phi on particle surface.

            #t0 = time.time()

            bps = Point(self.radius, theta, phi, 'spherical') - beam_pos

            _rho, _phi, _z = bps.cylindrical()

            # Vector parallel to the direction of a single ray.
            k0 = beam.wavenumber_direction(_rho, _phi, _z, 'cylindrical')

            n0 = self.normal(theta, phi)

            thetai = self.incident_angle(k0, n0)

            # Check if this sphere point is being illuminated
            if thetai >= pi/2:
                return 0

            thetar = self.refracted_angle(thetai,
                                          self._medium_refractive_index,
                                          self._refractive_index)

            d0 = self.ortonormal_ray_direction(k0, n0)

            E0 = beam.electric_field_direction(_rho, _phi, _z, 'cylindrical')

            beta = self.crossing_angle(k0, n0, E0)

            reflectivity = self.reflectivity(thetai, thetar, beta)

            trasmissivity = self.trasmissivity(thetai, thetar, beta)

            Qt = self.Qt(thetai, thetar, reflectivity, trasmissivity, force_type)

            intensity = beam.intensity(_rho, _phi, _z, 'cylindrical')

            dpower = intensity*ma.cos(thetai)

            _dforce = [(Qt.real*k + Qt.imag*d)*self._medium_refractive_index
                       * dpower/SPEED_OF_LIGHT for k, d in zip(k0, d0)]

            #k0time.append(time.time()-t0)
            #print('k0: ', np.mean(k0time))

            if force_dir == 'fx':
                return _dforce[0]
            elif force_dir == 'fy':
                return _dforce[1]
            elif force_dir == 'fz':
                return _dforce[2]
            else:
                return 0

        def quad_integration():
            def theta_integral(phi):
                val, err = quad(lambda theta: dforce(theta, phi)*ma.sin(theta),
                                0, pi, epsabs=0, epsrel=epsrel)
                return val

            val, err = quad(theta_integral, 0, 2*pi, epsabs=0, epsrel=epsrel)

            return val

        def dblquad_integration():
            val, err = dblquad(lambda th, ph: dforce(th, ph)*ma.sin(th),
                               0, 2*pi, lambda ph: 0, lambda ph: pi,
                               epsabs=0, epsrel=epsrel)

            return val

        return dblquad_integration()


if __name__ == '__main__':
    print("Please, visit: https://github.com/arantespp/opticalforces")

