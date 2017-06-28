Should add this part of code on geo_opt_force() method.

def test_msg():
    msg = ''
    msg += 'np=' + str(self.refractive_index) + ';\n'
    msg += 'nm=' + str(self.medium_refractive_index) + ';\n'
    msg += 'Rp=' + str(self.radius) + ';\n'
    msg += 'alpha=' + str(self.absorption_coefficient) + ';\n'
    msg += 'beampos=List' + str(beam_pos.cartesian()) + ';\n'
    msg += 'point=List' + str(bps.cartesian()) + ';\n'
    msg += 'theta=' + str(theta) + ';\n'
    msg += 'phi=' + str(phi) + ';\n'
    msg += 'n0=List' + str(list(n0)) + ';\n'
    msg += 'k0=List' + str(list(k0)) + ';\n'
    msg += 'E0=List' + str(list(E0)) + ';\n'
    msg += 'beta=' + str(beta) + ';\n'
    msg += 'thetai=' + str(thetai) + ';\n'
    msg += 'thetar=' + str(thetar)  + ';\n'
    msg += 'd0=List' + str(list(d0)) + ';\n'
    msg += 'Rpa=' + str(self.parallel_reflectivity(thetai, thetar)) + ';\n'
    msg += 'Rpe=' + str(self.perpendicular_reflectivity(thetai, thetar)) + ';\n'
    msg += 'R=' + str(reflectivity) + ';\n'
    msg += 'T=' + str(trasmissivity) + ';\n'
    msg += 'Qtreal=' + str(Qt.real) + ';\n'
    msg += 'Qtimag=' + str(Qt.imag) + ';\n'
    msg += 'intensity=' + str(intensity) + ';\n'
    msg += 'dForce=List' + str(list(_dforce)) + ';\n'
    msg = msg.replace('e-', '*10^-')
    pyperclip.copy(msg)
    pyperclip.paste()
    print(msg)
    time.sleep(1)

test_msg()

And this beam should be created

from beam import ScalarGaussianBeam, Point
from particle import SphericalParticle

sgb = ScalarGaussianBeam()
sgb.name = 'Zhang-beam'
sgb.vacuum_wavelength = 0.488e-6
sgb.medium_refractive_index = 1.33
sgb.gaussian_spot = 0.4e-6
sgb.electric_field_direction = lambda x1, x2, x3, s: [0, 1, 0]

ptc = SphericalParticle()
ptc.medium_refractive_index = 1.33
ptc.refractive_index = 1.6
ptc.absorption_coefficient = 0.00
ptc.radius = 6e-6

beam_pos = Point(0, 0, -4*ptc.radius)

paramx = {'param': 'beam_pos_x',
          'start': -3.5*ptc.radius,
          'stop': 3.3*ptc.radius,
          'num': 36+1,}

X, F = ptc.geo_opt_force(sgb, beam_pos, force_dir='fx', force_type='total')
