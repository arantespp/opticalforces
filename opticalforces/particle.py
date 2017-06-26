import math as ma
from math import pi
import cmath as cm
from numbers import Number
from functools import wraps
import time
from scipy.integrate import quad
import csv
import pandas as pd
import numpy as np
import os

import random

from beam import Point

# Speed of light.
SPEED_OF_LIGHT = 299792458

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

def timing(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        print('start:', time.strftime("%d %b %Y %H:%M:%S", time.localtime()))
        time0 = time.time()
        _func = func(*args, **kwargs)
        print('time:', time.time() - time0)
        return _func
    return wrapped

def check_geo_opt_database(func):
    @wraps(func)
    def wrapped(self, beam, beam_pos, force_dir, force_type='total',
                paramx=None, paramy=None):

        database_name = '_'.join([self.name, beam.name, 'geo-opt.pkl'])
        database_dir = 'database'
        database_full_path = os.path.join(database_dir, database_name)

        if not os.path.isdir(database_dir):
            os.makedirs(database_dir)

        # params without force
        params = {'beam_pos_x': round_sig(beam_pos.x),
                  'beam_pos_y': round_sig(beam_pos.y),
                  'beam_pos_z': round_sig(beam_pos.z),
                  'force_type': force_type,}

        for param in self.params:
            params.update({param[1:]: round_sig(self.__dict__[param])})

        @timing
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

            if dff.empty:
                force = func(self, beam, beam_pos, force_dir, force_type)
                _params = params.copy()
                _params[force_dir] = force
                df = df.append(_params, ignore_index=True)
                df.to_csv(database_full_path[:-3] + 'csv', index=False)
                df.to_pickle(database_full_path)
                return force
            else:
                if np.isnan(dff.loc[dff.index[0], force_dir]):
                    force = func(self, beam, beam_pos, force_dir, force_type)
                    df = df.set_value(dff.index[0], force_dir, force)
                    df.to_csv(database_full_path[:-3] + 'csv', index=False)
                    df.to_pickle(database_full_path)
                    return force
                else:
                    return dff.loc[dff.index[0], force_dir]

        def get_force_from_range(params, paramx):
            _params = params.copy()
            param = paramx['param']
            del _params[param]

            _df = pd.DataFrame(columns=[param, force_dir])

            if not os.path.isfile(database_full_path):
                return _df

            df = pd.read_pickle(database_full_path)

            # select all rows that have the same values as '_params'
            df = df[(df[list(_params)] == pd.Series(_params)).all(axis=1)]

            # select all rows that have 'force_dir' as a finite number
            df = df[pd.notnull(df[force_dir])]

            # select all rows that have its parameter 'param' in range
            df = df[(df[param]>=paramx['start']) & (df[param]<=paramx['stop'])]

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
            while _df[force_dir].isnull().values.any():
                _params = params.copy()
                for idx in range(_df.shape[0]):
                    _params[param] = _df[param][idx]
                    _df[force_dir][idx] = get_force_from_params(_params)

            return _df[param].values.tolist(), _df[force_dir].values.tolist()

        if paramx is None:
            return get_force_from_params(params)
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
            return [ma.sin(theta)*ma.cos(phi), ma.sin(theta)*ma.sin(phi),
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

    #@timing
    @check_geo_opt_database
    def geo_opt_force(self, beam, beam_pos, force_dir, force_type):
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
            bps = Point(self.radius, theta, phi, 'spherical') - beam_pos

            rho, phi, z = bps.cylindrical()

            # Vector parallel to the direction of a single ray.
            k0 = beam.wavenumber_direction(rho, phi, z, 'cylindrical')

            n0 = self.normal(theta, phi)

            thetai = self.incident_angle(k0, n0)

            # Check if this sphere point is being illuminated
            if thetai >= pi/2:
                return 0

            thetar = self.refracted_angle(thetai,
                                          self._medium_refractive_index,
                                          self._refractive_index)

            d0 = self.ortonormal_ray_direction(k0, n0)

            E0 = beam.electric_field_direction(rho, phi, z, 'cylindrical')

            beta = self.crossing_angle(k0, n0, E0)

            reflectivity = self.reflectivity(thetai, thetar, beta)

            trasmissivity = self.trasmissivity(thetai, thetar, beta)

            Qt = self.Qt(thetai, thetar, reflectivity, trasmissivity, force_type)

            intensity = beam.intensity(rho, phi, z, 'cylindrical')

            dpower = intensity*ma.cos(thetai)

            _dforce = [(Qt.real*k + Qt.imag*d)*self._medium_refractive_index
                       * dpower/SPEED_OF_LIGHT for k, d in zip(k0, d0)]

            if force_dir == 'fx':
                return _dforce[0]
            elif force_dir == 'fy':
                return _dforce[1]
            elif force_dir == 'fz':
                return _dforce[2]
            else:
                return 0

        def quad_integration():
            epsrel = 1e-2
            epsabs = 1e-18
            limit = 999

            def theta_integral(phi):
                val, err = quad(lambda theta: dforce(theta, phi)*ma.sin(theta),
                                0, pi, epsabs=epsabs, epsrel=epsrel,
                                limit=limit)
                return val

            val, err = quad(theta_integral, 0, 2*pi, epsabs=epsabs, limit=limit,
                            epsrel=epsrel)

            return val

        #return quad_integration()
        return random.random()


if __name__ == '__main__':
    print("Please, visit: https://github.com/arantespp/opticalforces")

    from beam import VectorialFrozenWave, Point
    import pandas as pd

    def ref_func(z):
        if abs(z) < 0.35*0.1:
            return 1
        else:
            return 0

    vfw = VectorialFrozenWave()
    vfw.wavelength = 1064e-9
    vfw.medium_refractive_index = 1.33
    vfw.Q = 0.99*vfw.wavenumber
    vfw.N = 5
    vfw.L = 0.1
    vfw.reference_function = ref_func

    # ----- particle definition
    ptc = SphericalParticle()
    ptc.radius = 10e-6
    ptc.medium_refractive_index = 1.33
    ptc.refractive_index = 1.2667890

    paramx = {'param': 'radius',
              'start': 1e-6,
              'stop': 200e-6,
              'num': 20,}

    print(ptc.geo_opt_force(beam=vfw, beam_pos=Point(3,22.1,0.01),
                            force_dir='fy', paramx=paramx))

    #print('\n $$$$$$$$$$$$$$$$$$$$')

    #print(ptc.geo_opt_force(beam=vfw, beam_pos=Point(1,0.1,0.01),
    #                        force_dir='fz', paramx=paramx))

    #print('\n $$$$$$$$$$$$$$$$$$$$')

    #print(ptc.geo_opt_force(beam=vfw, beam_pos=Point(2,0.1,0.01),
    #                        force_dir='fz', paramx=paramx))

    df = pd.DataFrame(columns=('a', 'b', 'c'))
    df = df.append({'a':1,'b':2, 'c':np.nan}, ignore_index=True)
    df = df.append({'a':1,'b':2, 'c':np.nan}, ignore_index=True)
    df = df.append({'a':3,'b':3, 'c':np.nan}, ignore_index=True)
    df = df.append({'a':2,'b':5, 'c':np.nan}, ignore_index=True)

    filt = {'a':2, 'b':5}

    dff = df[(df[list(filt)]==pd.Series(filt)).all(axis=1)]

    #print(dff)

    df.set_value(dff.index[0], 'c', 4)

    #print(df)
