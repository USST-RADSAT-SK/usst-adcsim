import numpy as np
from adcsim.CubeSat_model import CubeSat, Face2D, Face3D, Polygons3D
from adcsim.hysteresis_rod import HysteresisRod
from typing import List

"""
Compilation of default Cubesat models. 

Create your own CubeSat model: 
  In the __init__ method of your new class which inherits from the CubeSat class:
   1) Create a Face2D object for every unique panel.
   2) Position the panels in 3D space by making Face3D objects and translating/rotating them.
   3) If there are repeated collections of faces (ex: a cubesat side with 4 solar panels), make a list of the Face3D 
      objects and use it to initialize multiple Polygons3D objects, each of which can be translated and rotated as a 
      whole.
   4) Collect all faces and pass them to the __init__ method of the parent class.
 
The examples below are good starting points.

Use the plot method to visualize your model.
"""


class CubeSatEx1(CubeSat):
    def __init__(self):

        large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T)
        small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T)
        solar_panel = Face2D(np.array(
            [[-0.04, -0.02], [0.04, -0.02], [0.04, 0.01], [0.03, 0.02], [-0.03, 0.02], [-0.04, 0.01],
             [-0.04, -0.02]]).T)

        faces = []
        faces += [Face3D(large_face - solar_panel, '+y+z', np.array([0.05, 0., 0.]), name='+x face', color='g')]
        faces += [Face3D(solar_panel, '+y+z', np.array([0.05, 0., 0.]), name='+x solar panel', color='k')]
        faces += [Face3D(large_face - solar_panel, '-y+z', np.array([-0.05, 0., 0.]), name='-x face', color='g')]
        faces += [Face3D(solar_panel, '-y+z', np.array([-0.05, 0., 0.]), name='-x solar panel', color='k')]

        faces += [Face3D(large_face - solar_panel, '-x+z', np.array([0., 0.05, 0.]), name='+y face', color='g')]
        faces += [Face3D(solar_panel, '-x+z', np.array([0., 0.05, 0.]), name='+y solar panel', color='k')]
        faces += [Face3D(large_face - solar_panel, '+x+z', np.array([0., -0.05, 0.]), name='-y face', color='g')]
        faces += [Face3D(solar_panel, '+x+z', np.array([0., -0.05, 0.]), name='-y solar panel', color='k')]

        faces += [Face3D(small_face - solar_panel, '+x+y', np.array([0., 0., 0.1]), name='+z face', color='g')]
        faces += [Face3D(solar_panel, '+x+y', np.array([0., 0., 0.1]), name='+z solar panel', color='k')]
        faces += [Face3D(small_face - solar_panel, '-x+y', np.array([0., 0., -0.1]), name='-z face', color='g')]
        faces += [Face3D(solar_panel, '-x+y', np.array([0., 0., -0.1]), name='-z solar panel', color='k')]

        super().__init__(faces, np.zeros(3), inertia=np.diag([2e-3, 2e-3, 8e-3]))


class CubeSatSolarPressureEx1(CubeSat):
    def __init__(self, center_of_mass: np.ndarray = np.zeros(3), inertia: np.ndarray = np.diag([2e-3, 2e-3, 8e-3]),
                 residual_magnetic_moment: np.ndarray = np.zeros(3), magnetic_moment: np.ndarray = np.zeros(3),
                 hyst_rods=None):
        large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T,
                            diff_ref_coeff=1.0, spec_ref_coeff=0.0)
        small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T,
                            diff_ref_coeff=1.0, spec_ref_coeff=0.0)
        solar_panel = Face2D(np.array([[-0.04, -0.02], [0.04, -0.02], [0.04, 0.01], [0.03, 0.02], [-0.03, 0.02],
                                       [-0.04, 0.01], [-0.04, -0.02]]).T, diff_ref_coeff=0.0, spec_ref_coeff=0.6,
                             solar_power_efficiency=0.3)
        solarp_px = solar_panel + np.array([0., -0.07])

        faces = []
        faces += [Face3D(large_face - solarp_px, '+y+z', np.array([0.05, 0., 0.]), name='+x face', color='g')]
        faces += [Face3D(solarp_px, '+y+z', np.array([0.05, 0., 0.]), name='+x solar panel', color='k')]
        faces += [Face3D(large_face - solarp_px, '-y+z', np.array([-0.05, 0., 0.]), name='-x face', color='g')]
        faces += [Face3D(solarp_px, '-y+z', np.array([-0.05, 0., 0.]), name='-x solar panel', color='k')]

        faces += [Face3D(large_face - solarp_px, '-x+z', np.array([0., 0.05, 0.]), name='+y face', color='g')]
        faces += [Face3D(solarp_px, '-x+z', np.array([0., 0.05, 0.]), name='+y solar panel', color='k')]
        faces += [Face3D(large_face - solarp_px, '+x+z', np.array([0., -0.05, 0.]), name='-y face', color='g')]
        faces += [Face3D(solarp_px, '+x+z', np.array([0., -0.05, 0.]), name='-y solar panel', color='k')]

        faces += [Face3D(small_face, '+x+y', np.array([0., 0., 0.1]), name='+z face', color='g')]
        faces += [Face3D(small_face, '-x+y', np.array([0., 0., -0.1]), name='-z face', color='g')]

        super().__init__(faces, center_of_mass=center_of_mass, inertia=inertia,
                         residual_magnetic_moment=residual_magnetic_moment, magnetic_moment=magnetic_moment,
                         hyst_rods=hyst_rods)


class CubeSatModel(CubeSat):
    def __init__(self, center_of_mass: np.ndarray=np.zeros(3), inertia: np.ndarray=np.diag([2e-3, 2e-3, 8e-3]), residual_magnetic_moment: np.ndarray = np.zeros(3), magnetic_moment: np.ndarray = np.zeros(3), hyst_rods: List[HysteresisRod] = None):
        solar_panel_up = Face2D(np.array([[-0.04, -0.022], [0.04, -0.022], [0.04, 0.008], [0.026, 0.022], [-0.026, 0.022], [-0.04, 0.008], [-0.04, -0.022]]).T, diff_ref_coeff=0.0, spec_ref_coeff=0.6, solar_power_efficiency=0.3)
        solar_panel_dn = Face2D(np.array([[-0.04, 0.022], [0.04, 0.022], [0.04, -0.008], [0.026, -0.022], [-0.026, -0.022], [-0.04, -0.008], [-0.04, 0.022]]).T[:, ::-1], diff_ref_coeff=0.0, spec_ref_coeff=0.6, solar_power_efficiency=0.3)

        body_small_panel = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T, diff_ref_coeff=1.0, spec_ref_coeff=0.0)
        body_small_panel -= solar_panel_up + np.array([0., 0.026])
        body_small_panel -= solar_panel_dn - np.array([0., 0.026])
        body_small = [
            Face3D(body_small_panel, color=[0.5, 0.5, 0.5]),
            Face3D(solar_panel_up, translation=np.array([0., 0.026, 0.])),
            Face3D(solar_panel_dn, translation=np.array([0., -0.026, 0.]))
        ]

        body_large_panel = Face2D(np.array([[-0.05, -0.108], [0.05, -0.108], [0.05, 0.108], [-0.05, 0.108], [-0.05, -0.108]]).T, diff_ref_coeff=1.0, spec_ref_coeff=0.0)
        body_large_panel -= solar_panel_up + np.array([0., 0.080])
        body_large_panel -= solar_panel_dn + np.array([0., 0.028])
        body_large_panel -= solar_panel_up - np.array([0., 0.028])
        body_large_panel -= solar_panel_dn - np.array([0., 0.080])
        body_large = [
            Face3D(body_large_panel, color=[0.5, 0.5, 0.5]),
            Face3D(solar_panel_up, translation=np.array([0., 0.080, 0.])),
            Face3D(solar_panel_dn, translation=np.array([0., 0.028, 0.])),
            Face3D(solar_panel_up, translation=np.array([0., -0.028, 0.])),
            Face3D(solar_panel_dn, translation=np.array([0., -0.080, 0.]))
        ]

        body_large_panel_px = Face2D(np.array([[-0.05, -0.108], [0.05, -0.108], [0.05, 0.108], [-0.05, 0.108], [-0.05, -0.108]]).T, diff_ref_coeff=1.0, spec_ref_coeff=0.0)
        body_large_panel_px -= solar_panel_up + np.array([0., 0.080])
        body_large_panel_px -= solar_panel_dn + np.array([0., 0.028])
        body_large_px = [
            Face3D(body_large_panel_px, color=[0.5, 0.5, 0.5]),
            Face3D(solar_panel_up, translation=np.array([0., 0.080, 0.])),
            Face3D(solar_panel_dn, translation=np.array([0., 0.028, 0.]))
        ]

        body_panel_px = Polygons3D(body_large_px)
        body_panel_px.rotate(axis='+y+z')
        body_panel_px.translate(np.array([0.05, 0., 0.]))

        body_panel_py = Polygons3D(body_large)
        body_panel_py.rotate(axis='-x+z')
        body_panel_py.translate(np.array([0., 0.05, 0.]))

        body_panel_mx = Polygons3D(body_large)
        body_panel_mx.rotate(axis='-y+z')
        body_panel_mx.translate(np.array([-0.05, 0., 0.]))

        body_panel_my = Polygons3D(body_large)
        body_panel_my.rotate(axis='+x+z')
        body_panel_my.translate(np.array([0., -0.05, 0.]))

        body_panel_pz = Polygons3D(body_small)
        body_panel_pz.translate(np.array([0., 0., 0.108]))

        body_panel_mz = Polygons3D(body_small)
        body_panel_mz.rotate(axis='-x+y')
        body_panel_mz.translate(np.array([0., 0., -0.108]))

        faces = body_panel_px.faces + body_panel_py.faces + body_panel_pz.faces
        faces += body_panel_mx.faces + body_panel_my.faces + body_panel_mz.faces

        antenna_base_large_face = Face2D(np.array([[-3., 0.], [3., 0.], [3., 35.7], [-3., 35.7], [-3., 0.]]).T/1e3)
        antenna_base_small_face = Face2D(np.array([[-2., 0.], [2., 0.], [2., 35.7], [-2., 35.7], [-2., 0.]]).T/1e3)
        long_antenna_face       = Face2D(np.array([[-1.5, 0.], [1.5, 0.], [1.5, 472.3], [-1.5, 472.3], [-1.5, 0.]]).T/1e3)
        short_antenna_face      = Face2D(np.array([[-1.5, 0.], [1.5, 0.], [1.5, 152.3], [-1.5, 152.3], [-1.5, 0.]]).T/1e3)

        antenna_base_panels = [Face3D(antenna_base_large_face, '+x+y', np.array([0., 0., 2.]).T/1e3),
                               Face3D(antenna_base_large_face, '-x+y', np.array([0., 0., -2.]).T/1e3),
                               Face3D(antenna_base_small_face, '-z+y', np.array([3., 0., 0.]).T/1e3),
                               Face3D(antenna_base_small_face, '+z+y', np.array([-3., 0., 0.]).T/1e3)]

        long_antenna_panels = [Face3D(long_antenna_face,  '-z+y', np.array([1., 35.7, 0.]).T/1e3),
                               Face3D(long_antenna_face,  '+z+y', np.array([1., 35.7, 0.]).T/1e3)]
        short_antenna_panels = [Face3D(short_antenna_face, '-z+y', np.array([1., 35.7, 0.]).T/1e3),
                                Face3D(short_antenna_face, '+z+y', np.array([1., 35.7, 0.]).T/1e3)]

        long_antenna_py = Polygons3D(antenna_base_panels + long_antenna_panels)
        long_antenna_py.translate(np.array([22., 50., 105.])/1e3)
        long_antenna_my = Polygons3D(antenna_base_panels + long_antenna_panels)
        long_antenna_my.rotate(axis='+z', angle=180.)
        long_antenna_my.translate(np.array([-22., -50., 105.])/1e3)

        short_antenna_px = Polygons3D(antenna_base_panels + short_antenna_panels)
        short_antenna_px.rotate(axis='+z', angle=-90.)
        short_antenna_px.translate(np.array([50., 22., 105.])/1e3)
        short_antenna_mx = Polygons3D(antenna_base_panels + short_antenna_panels)
        short_antenna_mx.rotate(axis='+z', angle=90.)
        short_antenna_mx.translate(np.array([-50., -22., 105.])/1e3)

        faces += short_antenna_px.faces + short_antenna_mx.faces + long_antenna_py.faces + long_antenna_my.faces
        super().__init__(faces, center_of_mass=center_of_mass, inertia=inertia, residual_magnetic_moment=residual_magnetic_moment, magnetic_moment=magnetic_moment, hyst_rods=hyst_rods)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cubesat = CubeSatModel()
    # cubesat = CubeSatEx1()
    cubesat.plot()
    plt.show()
