import numpy as np
from CubeSat_model import CubeSat, Face2D, Face3D, Polygons3D


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
    def __init__(self, center_of_mass: np.ndarray = np.zeros(3), inertia: np.ndarray = np.diag([2e-3, 2e-3, 8e-3])):
        large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T,
                            reflection_coeff=0.6)
        small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T,
                            reflection_coeff=0.6)
        solar_panel = Face2D(np.array([[-0.04, -0.02], [0.04, -0.02], [0.04, 0.01], [0.03, 0.02], [-0.03, 0.02],
                                       [-0.04, 0.01], [-0.04, -0.02]]).T, reflection_coeff=1.0)

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
        super().__init__(faces, center_of_mass=center_of_mass, inertia=inertia)


class CubeSatAerodynamicEx1(CubeSat):
    def __init__(self):
        body_large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T)
        body_small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T)

        body_px = Face3D(body_large_face, '+y+z', np.array([0.05, 0., 0.]), name='body+x', color='C0')
        body_mx = Face3D(body_large_face, '-y+z', np.array([-0.05, 0., 0.]), name='body-x', color='C1')
        body_py = Face3D(body_large_face, '-x+z', np.array([0., 0.05, 0.]), name='body+y', color='C2')
        body_my = Face3D(body_large_face, '+x+z', np.array([0., -0.05, 0.]), name='body-y', color='C3')
        body_pz = Face3D(body_small_face, '+x+y', np.array([0., 0., 0.1]), name='body+z', color='C4')
        body_mz = Face3D(body_small_face, '-x+y', np.array([0., 0., -0.1]), name='body-z', color='C5')

        antenna_base_large_face = Face2D(np.array([[-3., 0.], [3., 0.], [3., 35.7], [-3., 35.7], [-3., 0.]]).T/1e3)
        antenna_base_small_face = Face2D(np.array([[-2., 0.], [2., 0.], [2., 35.7], [-2., 35.7], [-2., 0.]]).T/1e3)
        long_antenna_face       = Face2D(np.array([[-1.5, 0.], [1.5, 0.], [1.5, 478.3], [-1.5, 478.3], [-1.5, 0.]]).T/1e3)
        short_antenna_face      = Face2D(np.array([[-1.5, 0.], [1.5, 0.], [1.5, 136.3], [-1.5, 136.3], [-1.5, 0.]]).T/1e3)

        antenna_base_up    = Face3D(antenna_base_large_face, '+x+y', np.array([0., 0., 2.]).T/1e3)
        antenna_base_down  = Face3D(antenna_base_large_face, '-x+y', np.array([0., 0., -2.]).T/1e3)
        antenna_base_left  = Face3D(antenna_base_small_face, '-z+y', np.array([3., 0., 0.]).T/1e3)
        antenna_base_right = Face3D(antenna_base_small_face, '+z+y', np.array([-3., 0., 0.]).T/1e3)

        long_antenna_left   = Face3D(long_antenna_face,  '-z+y', np.array([1., 35.7, 0.]).T/1e3)
        long_antenna_right  = Face3D(long_antenna_face,  '+z+y', np.array([1., 35.7, 0.]).T/1e3)
        short_antenna_left  = Face3D(short_antenna_face, '-z+y', np.array([1., 35.7, 0.]).T/1e3)
        short_antenna_right = Face3D(short_antenna_face, '+z+y', np.array([1., 35.7, 0.]).T/1e3)

        long_antenna_py = Polygons3D([antenna_base_up, antenna_base_down, antenna_base_left, antenna_base_right, long_antenna_left, long_antenna_right])
        long_antenna_py.translate(np.array([22., 50., 97.])/1e3)
        long_antenna_my = Polygons3D([antenna_base_up, antenna_base_down, antenna_base_left, antenna_base_right, long_antenna_left, long_antenna_right])
        long_antenna_my.rotate(axis='+z', angle=180.)
        long_antenna_my.translate(np.array([-22., -50., 97.])/1e3)

        short_antenna_px = Polygons3D([antenna_base_up, antenna_base_down, antenna_base_left, antenna_base_right, short_antenna_left, short_antenna_right])
        short_antenna_px.rotate(axis='+z', angle=-90.)
        short_antenna_px.translate(np.array([50., 22., 97.])/1e3)
        short_antenna_mx = Polygons3D([antenna_base_up, antenna_base_down, antenna_base_left, antenna_base_right, short_antenna_left, short_antenna_right])
        short_antenna_mx.rotate(axis='+z', angle=90.)
        short_antenna_mx.translate(np.array([-50., -22., 97.])/1e3)

        faces = [body_px, body_mx, body_py, body_my, body_pz, body_mz]
        faces += short_antenna_px.faces + short_antenna_mx.faces
        faces += long_antenna_py.faces + long_antenna_my.faces
        super().__init__(faces, center_of_mass=np.zeros(3), inertia=np.diag([2e-3, 2e-3, 8e-3]))
