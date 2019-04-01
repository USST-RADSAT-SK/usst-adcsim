import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Union, List


class Face2D:
    def __init__(self, vertices: np.ndarray, sigma_n: float=0.8, sigma_t: float=0.8):
        """
        Parameters
        ----------
        vertices : np.ndarray
            2D vertices, shape (2, n) - Should be specified in the counterclockwise direction, and the first and last
            point should be the same. Holes can be specified by appending the vertices of the hole listed in the
            clockwise direction. The first and last vertices of the outer polygon, as well as the first and last
            vertices of each hole, should also be the same.
        sigma_n : float
            Normal momentum exchange coefficient, used to compute the aerodynamic force on a surface.
        sigma_t : float
            Tangential momentum exchange coefficient, used to compute the aerodynamic force on a surface.
        """

        assert(len(vertices.shape) == 2)
        assert(vertices.shape[0] == 2)

        self._vertices = vertices
        self._area = self._polygon_area(vertices)
        self._centroid = self._polygon_centroid(vertices)

        self._sigma_n = sigma_n
        self._sigma_t = sigma_t

    @property
    def vertices(self):
        return self._vertices

    @property
    def num_vertices(self):
        return self._vertices.shape[1]

    @property
    def area(self):
        return self._area

    @property
    def centroid(self):
        return self._centroid

    @property
    def sigma_n(self):
        return self._sigma_n

    @property
    def sigma_t(self):
        return self._sigma_t

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return Face2D(self.vertices + other.reshape(2, 1), sigma_n=self.sigma_n, sigma_t=self.sigma_t)
        elif isinstance(other, Face2D):
            return Face2D(np.concatenate((self.vertices, other.vertices, self.vertices[:, :1]), axis=1), sigma_n=self.sigma_n, sigma_t=self.sigma_t)
        else:
            raise TypeError(f'Cannot add object of type {type(other)} to Face2D object')

    def __iadd__(self, other):
        if isinstance(other, np.ndarray):
            self._vertices += other.reshape(2, 1)
            return self
        elif isinstance(other, Face2D):
            self._vertices = np.concatenate((self.vertices, other.vertices, self.vertices[:, :1]), axis=1)
            return self
        else:
            raise TypeError(f'Cannot add object of type {type(other)} to Face2D object')

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return Face2D(self.vertices - other.reshape(2, 1), sigma_n=self.sigma_n, sigma_t=self.sigma_t)
        elif isinstance(other, Face2D):
            return Face2D(np.concatenate((self.vertices, other.vertices[:, ::-1], self.vertices[:, :1]), axis=1), sigma_n=self.sigma_n, sigma_t=self.sigma_t)
        else:
            raise TypeError(f'Cannot subtract object of type {type(other)} from Face2D object')

    def __isub__(self, other):
        if isinstance(other, np.ndarray):
            self._vertices -= other.reshape(2, 1)
            return self
        elif isinstance(other, Face2D):
            self._vertices = np.concatenate((self.vertices, other.vertices[:, ::-1], self.vertices[:, :1]), axis=1)
            return self
        else:
            raise TypeError(f'Cannot add object of type {type(other)} to Face2D object')

    @staticmethod
    def _polygon_area(v: np.ndarray):
        """
        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        """
        n = v.shape[1]
        a = 0.0
        for i in range(n - 1):
            a += v[0, i] * v[1, i + 1] - v[0, i + 1] * v[1, i]
        return 0.5 * a

    @staticmethod
    def _polygon_centroid(v: np.ndarray):
        """
        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        """
        n = v.shape[1]
        cx = 0.0
        cy = 0.0
        for i in range(n - 1):
            cx += (v[0, i] + v[0, i + 1]) * (v[0, i] * v[1, i + 1] - v[0, i + 1] * v[1, i])
            cy += (v[1, i] + v[1, i + 1]) * (v[0, i] * v[1, i + 1] - v[0, i + 1] * v[1, i])
        return np.array([cx, cy]) / (6 * Face2D._polygon_area(v))


class Face3D:
    def __init__(self, face: Face2D, orientation: Union[str, np.ndarray]='+x+y', translation: np.ndarray=np.zeros(3), name='', color='k'):
        self._orientation = np.eye(3)
        self._translation = np.zeros((3, 1))
        self.face = face
        self.orientation = orientation
        self.translation = translation

        self._name = name
        self._color = color

    @property
    def vertices(self):
        return self._vertices

    @property
    def area(self):
        return self._area

    @property
    def centroid(self):
        return self._centroid

    @property
    def normal(self):
        return self._normal

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value: Union[str, np.ndarray]):
        if isinstance(value, str):
            e1 = self._unit_vector_from_string(value[:2])
            e2 = self._unit_vector_from_string(value[2:])
            e3 = np.cross(e1, e2)
            self._orientation = np.column_stack((e1, e2, e3))
        else:
            self._orientation = value
        self._set_face_positions()

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value: np.ndarray):
        self._translation = value.reshape(3, 1)
        self._set_face_positions()

    @property
    def face(self):
        return self._face

    @face.setter
    def face(self, value: Face2D):
        self._face = value
        self._set_face_positions()

    @property
    def name(self):
        return self._name

    @property
    def color(self):
        return self._color

    def _unit_vector_from_string(self, string):
        if string[1] == 'x':
            v = np.array([1., 0., 0.])
        elif string[1] == 'y':
            v = np.array([0., 1., 0.])
        elif string[1] == 'z':
            v = np.array([0., 0., 1.])
        else:
            return None

        if string[0] == '+':
            return v
        elif string[0] == '-':
            return -v
        else:
            return None

    def _set_face_positions(self):
        # convert 2D centroid/vertices to 3D in the xy plane
        self._vertices = np.zeros((3, self.face.num_vertices))
        self._vertices[:2, :] = self.face.vertices
        self._area = self.face.area

        self._centroid = np.array([*self.face.centroid, 0.0])
        self._normal = np.array([0., 0., 1.])

        # rotate and translate centroid/normal/vertices
        self._vertices = self.orientation @ self._vertices
        self._centroid = self.orientation @ self._centroid
        self._normal = self.orientation @ self._normal

        self._vertices += self.translation
        self._centroid += self.translation.squeeze()


class Polygons3D:
    def __init__(self, faces: List[Face3D]):
        self._faces = faces

    def plot(self, ax: plt.Axes):
        vertices = [f.vertices.T for f in self._faces]
        colors = [f.color for f in self._faces]
        poly = Poly3DCollection(vertices, facecolors=colors)
        ax.add_collection(poly)

    @property
    def faces(self):
        return self._faces


class CubeSat(Polygons3D):
    def __init__(self):
        large_face = Face2D(np.array([[-0.05, -0.1], [0.05, -0.1], [0.05, 0.1], [-0.05, 0.1], [-0.05, -0.1]]).T)
        small_face = Face2D(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05]]).T)
        solar_panel = Face2D(np.array([[-0.04, -0.02], [0.04, -0.02], [0.04, 0.01], [0.03, 0.02], [-0.03, 0.02], [-0.04, 0.01], [-0.04, -0.02]]).T)

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
        super().__init__(faces)


if __name__ == '__main__':

    cubesat = CubeSat()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)
    cubesat.plot(ax)
    plt.show()
    exit()

    big_square = Face2D(np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.], [-2., -2.]]).T)
    little_square = Face2D(np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [-1., -1.]]).T)
    square_with_hole = big_square - (little_square + np.array([0.5, 0.5]))

    print(square_with_hole.area)
    print(square_with_hole.centroid)

    face_3d = Face3D(square_with_hole, orientation='+z-y', translation=np.array([0., 1., 2.]), color='g')
    print(face_3d.area)
    print(face_3d.centroid)
    print(face_3d.normal)

    poly_3d = Polygons3D([face_3d])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    poly_3d.plot(ax)
    plt.show()





