import numpy as np
from adcsim.CubeSat_model import CubeSat, Face2D, Face3D, Polygons3D


class CubeSatEx1(CubeSat):
    def __init__(self):
        large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T)
        small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T)
        solar_panel = Face2D(np.array(
            [[-0.04, -0.02], [0.04, -0.02], [0.04, 0.01], [0.03, 0.02], [-0.03, 0.02], [-0.04, 0.01],
             [-0.04, -0.02]]).T)

        face_px = Face3D(large_face - solar_panel, '+y+z', np.array([0.05, 0., 0.]), name='+x face', color='g')
        solar_px = Face3D(solar_panel, '+y+z', np.array([0.05, 0., 0.]), name='+x solar panel', color='k')
        face_mx = Face3D(large_face - solar_panel, '-y+z', np.array([-0.05, 0., 0.]), name='-x face', color='g')
        solar_mx = Face3D(solar_panel, '-y+z', np.array([-0.05, 0., 0.]), name='-x solar panel', color='k')

        face_py = Face3D(large_face - solar_panel, '-x+z', np.array([0., 0.05, 0.]), name='+y face', color='g')
        solar_py = Face3D(solar_panel, '-x+z', np.array([0., 0.05, 0.]), name='+y solar panel', color='k')
        face_my = Face3D(large_face - solar_panel, '+x+z', np.array([0., -0.05, 0.]), name='-y face', color='g')
        solar_my = Face3D(solar_panel, '+x+z', np.array([0., -0.05, 0.]), name='-y solar panel', color='k')

        face_pz = Face3D(small_face - solar_panel, '+x+y', np.array([0., 0., 0.1]), name='+z face', color='g')
        solar_pz = Face3D(solar_panel, '+x+y', np.array([0., 0., 0.1]), name='+z solar panel', color='k')
        face_mz = Face3D(small_face - solar_panel, '-x+y', np.array([0., 0., -0.1]), name='-z face', color='g')
        solar_mz = Face3D(solar_panel, '-x+y', np.array([0., 0., -0.1]), name='-z solar panel', color='k')

        faces = [value for value in locals().values() if isinstance(value, Face3D)]
        super().__init__(faces, center_of_mass=np.zeros(3), inertia=np.diag([2e-3, 2e-3, 8e-3]))


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
        face_px = Face3D(large_face - solarp_px, '+y+z', np.array([0.05, 0., 0.]), name='+x face', color='g')
        solar_px = Face3D(solarp_px, '+y+z', np.array([0.05, 0., 0.]), name='+x solar panel', color='k')
        face_mx = Face3D(large_face - solarp_px, '-y+z', np.array([-0.05, 0., 0.]), name='-x face', color='g')
        solar_mx = Face3D(solarp_px, '-y+z', np.array([-0.05, 0., 0.]), name='-x solar panel', color='k')

        face_py = Face3D(large_face - solarp_px, '-x+z', np.array([0., 0.05, 0.]), name='+y face', color='g')
        solar_py = Face3D(solarp_px, '-x+z', np.array([0., 0.05, 0.]), name='+y solar panel', color='k')
        face_my = Face3D(large_face - solarp_px, '+x+z', np.array([0., -0.05, 0.]), name='-y face', color='g')
        solar_my = Face3D(solarp_px, '+x+z', np.array([0., -0.05, 0.]), name='-y solar panel', color='k')

        face_pz = Face3D(small_face, '+x+y', np.array([0., 0., 0.1]), name='+z face', color='g')
        face_mz = Face3D(small_face, '-x+y', np.array([0., 0., -0.1]), name='-z face', color='g')

        faces = [value for value in locals().values() if isinstance(value, Face3D)]
        super().__init__(faces, center_of_mass=center_of_mass, inertia=inertia,
                         residual_magnetic_moment=residual_magnetic_moment, magnetic_moment=magnetic_moment,
                         hyst_rods=hyst_rods)


class CubeSatAerodynamicEx1(CubeSat):
    def __init__(self):
        body_large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T,
                                 diff_ref_coeff=1.0, spec_ref_coeff=0.0)
        body_small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T,
                                 diff_ref_coeff=1.0, spec_ref_coeff=0.0)

        solar_panel = Face2D(np.array([[-0.04, -0.02], [0.04, -0.02], [0.04, 0.0064], [0.0264, 0.02], [-0.0264, 0.02], [-0.04, 0.0064], [-0.04, -0.02]]).T,
                             diff_ref_coeff=0.0, spec_ref_coeff=0.6)

        solar_positions = {
            '+y+z': (np.array([0.05, 0., 0.]), (np.array([0., 0.0709]), np.array([0., 0.0289]), np.array([0., -0.0289]), np.array([0., -0.0709]))),
            '-y+z': (np.array([-0.05, 0., 0.]), (np.array([0., 0.0709]), np.array([0., 0.0289]), np.array([0., -0.0289]), np.array([0., -0.0709]))),
            '-x+z': (np.array([0., 0.05, 0.]), (np.array([0., 0.0709]), np.array([0., 0.0289]), np.array([0., -0.0289]), np.array([0., -0.0709]))),
            '+x+z': (np.array([0., -0.05, 0.]), (np.array([0., 0.0709]), np.array([0., 0.0289]), np.array([0., -0.0289]), np.array([0., -0.0709]))),
            '+x+y': (np.array([0., 0., 0.1]), (np.array([0., 0.021]), np.array([0., -0.021]))),
            '-x+y': (np.array([0., 0., -0.1]), (np.array([0., 0.021]), np.array([0., -0.021])))
        }
        # solar_panels = Face3D(solar_panel, p[0], p[1]) for p in solar_positions]
        solar_panels = []
        for plane, (normal, inplane) in solar_positions.items():
            solar_panels += [Face3D(solar_panel + position, plane, normal) for position in inplane]

        body_panels = []
        i = 0
        for plane, (normal, inplane) in solar_positions.items():
            face = body_small_face.copy() if 'x+y' in plane else body_large_face.copy()
            for position in inplane:
                face -= solar_panel + position
            body_panels += [Face3D(face, plane, normal, color=f'C{i}')]
            i += 1

        antenna_base_large_face = Face2D(np.array([[-3., 0.], [3., 0.], [3., 35.7], [-3., 35.7], [-3., 0.]]).T/1e3)
        antenna_base_small_face = Face2D(np.array([[-2., 0.], [2., 0.], [2., 35.7], [-2., 35.7], [-2., 0.]]).T/1e3)
        long_antenna_face       = Face2D(np.array([[-1.5, 0.], [1.5, 0.], [1.5, 478.3], [-1.5, 478.3], [-1.5, 0.]]).T/1e3)
        short_antenna_face      = Face2D(np.array([[-1.5, 0.], [1.5, 0.], [1.5, 136.3], [-1.5, 136.3], [-1.5, 0.]]).T/1e3)

        antenna_base_panels = [Face3D(antenna_base_large_face, '+x+y', np.array([0., 0., 2.]).T/1e3),
                               Face3D(antenna_base_large_face, '-x+y', np.array([0., 0., -2.]).T/1e3),
                               Face3D(antenna_base_small_face, '-z+y', np.array([3., 0., 0.]).T/1e3),
                               Face3D(antenna_base_small_face, '+z+y', np.array([-3., 0., 0.]).T/1e3)]

        long_antenna_panels = [Face3D(long_antenna_face,  '-z+y', np.array([1., 35.7, 0.]).T/1e3),
                               Face3D(long_antenna_face,  '+z+y', np.array([1., 35.7, 0.]).T/1e3)]
        short_antenna_panels = [Face3D(short_antenna_face, '-z+y', np.array([1., 35.7, 0.]).T/1e3),
                                Face3D(short_antenna_face, '+z+y', np.array([1., 35.7, 0.]).T/1e3)]

        long_antenna_py = Polygons3D(antenna_base_panels + long_antenna_panels)
        long_antenna_py.translate(np.array([22., 50., 97.])/1e3)
        long_antenna_my = Polygons3D(antenna_base_panels + long_antenna_panels)
        long_antenna_my.rotate(axis='+z', angle=180.)
        long_antenna_my.translate(np.array([-22., -50., 97.])/1e3)

        short_antenna_px = Polygons3D(antenna_base_panels + short_antenna_panels)
        short_antenna_px.rotate(axis='+z', angle=-90.)
        short_antenna_px.translate(np.array([50., 22., 97.])/1e3)
        short_antenna_mx = Polygons3D(antenna_base_panels + short_antenna_panels)
        short_antenna_mx.rotate(axis='+z', angle=90.)
        short_antenna_mx.translate(np.array([-50., -22., 97.])/1e3)

        faces = body_panels + solar_panels + short_antenna_px.faces + short_antenna_mx.faces + long_antenna_py.faces + long_antenna_my.faces
        super().__init__(faces, center_of_mass=np.zeros(3), inertia=np.diag([2e-3, 2e-3, 8e-3]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cubesat = CubeSatAerodynamicEx1()
    cubesat.plot()
    plt.show()
