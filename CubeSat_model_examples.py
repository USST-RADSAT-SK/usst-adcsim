import numpy as np
from CubeSat_model import CubeSat, Face2D, Face3D


class CubeSatEx1(CubeSat):
    def __init__(self):
        large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T)
        small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T)
        solar_panel = Face2D(np.array(
            [[-0.04, -0.02], [0.04, -0.02], [0.04, 0.01], [0.03, 0.02], [-0.03, 0.02], [-0.04, 0.01],
             [-0.04, -0.02]]).T)

        face_px = Face3D(large_face - solar_panel, '+y+z', np.array([0.05, 0., 0.]), name='+x face', color='C0')
        solar_px = Face3D(solar_panel, '+y+z', np.array([0.05, 0., 0.]), name='+x solar panel', color='k')
        face_mx = Face3D(large_face - solar_panel, '-y+z', np.array([-0.05, 0., 0.]), name='-x face', color='C1')
        solar_mx = Face3D(solar_panel, '-y+z', np.array([-0.05, 0., 0.]), name='-x solar panel', color='k')

        face_py = Face3D(large_face - solar_panel, '-x+z', np.array([0., 0.05, 0.]), name='+y face', color='C2')
        solar_py = Face3D(solar_panel, '-x+z', np.array([0., 0.05, 0.]), name='+y solar panel', color='k')
        face_my = Face3D(large_face - solar_panel, '+x+z', np.array([0., -0.05, 0.]), name='-y face', color='C3')
        solar_my = Face3D(solar_panel, '+x+z', np.array([0., -0.05, 0.]), name='-y solar panel', color='k')

        face_pz = Face3D(small_face - solar_panel, '+x+y', np.array([0., 0., 0.1]), name='+z face', color='C4')
        solar_pz = Face3D(solar_panel, '+x+y', np.array([0., 0., 0.1]), name='+z solar panel', color='k')
        face_mz = Face3D(small_face - solar_panel, '-x+y', np.array([0., 0., -0.1]), name='-z face', color='C5')
        solar_mz = Face3D(solar_panel, '-x+y', np.array([0., 0., -0.1]), name='-z solar panel', color='k')

        faces = [value for value in locals().values() if isinstance(value, Face3D)]
        super().__init__(faces, center_of_mass=np.zeros(3), inertia=np.diag([2e-3, 2e-3, 8e-3]))


class CubeSatSolarPressureEx1(CubeSat):
    def __init__(self):
        large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T,
                            reflection_coeff=0.6)
        small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T,
                            reflection_coeff=0.6)
        solar_panel = Face2D(np.array([[-0.04, -0.02], [0.04, -0.02], [0.04, 0.01], [0.03, 0.02], [-0.03, 0.02],
                                       [-0.04, 0.01], [-0.04, -0.02]]).T, reflection_coeff=1.0)

        solarp_px = solar_panel + np.array([0., -0.07])
        face_px = Face3D(large_face - solarp_px, '+y+z', np.array([0.05, 0., 0.]), name='+x face', color='C0')
        solar_px = Face3D(solarp_px, '+y+z', np.array([0.05, 0., 0.]), name='+x solar panel', color='k')
        face_mx = Face3D(large_face - solarp_px, '-y+z', np.array([-0.05, 0., 0.]), name='-x face', color='C1')
        solar_mx = Face3D(solarp_px, '-y+z', np.array([-0.05, 0., 0.]), name='-x solar panel', color='k')

        face_py = Face3D(large_face - solarp_px, '-x+z', np.array([0., 0.05, 0.]), name='+y face', color='C2')
        solar_py = Face3D(solarp_px, '-x+z', np.array([0., 0.05, 0.]), name='+y solar panel', color='k')
        face_my = Face3D(large_face - solarp_px, '+x+z', np.array([0., -0.05, 0.]), name='-y face', color='C3')
        solar_my = Face3D(solarp_px, '+x+z', np.array([0., -0.05, 0.]), name='-y solar panel', color='k')

        face_pz = Face3D(small_face, '+x+y', np.array([0., 0., 0.1]), name='+z face', color='C4')
        face_mz = Face3D(small_face, '-x+y', np.array([0., 0., -0.1]), name='-z face', color='C5')

        faces = [value for value in locals().values() if isinstance(value, Face3D)]
        super().__init__(faces, center_of_mass=np.zeros(3), inertia=np.diag([2e-3, 2e-3, 8e-3]))
