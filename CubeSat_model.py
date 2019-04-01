import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Union, List


class Face2D:
    def __init__(self, vertices: np.ndarray, *hole_vertices: np.ndarray):
        """
        Parameters
        ----------
        vertices : np.ndarray
            2D vertices, shape (2, n)
        *hole_vertices : np.ndarray
            2D vertices defining holes in the polygon, shape (2, n)
        """

        assert(len(vertices.shape) == 2)
        assert(vertices.shape[0] == 2)
        assert(np.all(vertices[:, 0] == vertices[:, -1]))

        self._vertices = vertices
        self._area = self._polygon_area(vertices)
        self._centroid = self._polygon_centroid(vertices)

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
    def __init__(self, face: Face2D, orientation: Union[str, np.ndarray], translation: np.ndarray):
        self._orientation = np.eye(3)
        self._translation = np.zeros((3, 1))
        self.face = face
        self.orientation = orientation
        self.translation = translation

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
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value: Union[str, np.ndarray]):
        if isinstance(value, str):
            e1 = self._unit_vector_from_string(value[:2])
            e2 = self._unit_vector_from_string(value[2:])
            e3 = np.cross(e1, e2)
            self._orientation =  np.stack((e1, e2, e3))
        else:
            self._orientation = value
        self._position_face()

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value: np.ndarray):
        self._translation = value.reshape(3, 1)
        self._position_face()

    @property
    def face(self):
        return self._face

    @face.setter
    def face(self, value: Face2D):
        self._face = value
        self._position_face()

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

    def _position_face(self):
        # convert 2D centroid/vertices to 3D in the xy plane
        self._vertices = np.zeros((3, self.face.num_vertices))
        self._vertices[:2, :] = self.face.vertices
        self._area = self.face.area

        self._centroid = np.array([*self.face.centroid, 0.0])

        # rotate and translate centroid/vertices
        self._vertices = self.orientation @ self._vertices
        self._vertices += self.translation

class CubeSat:
    def __init__(self, faces: List[Face3D]):
        self._faces = faces

    def plot(self, ax: plt.Axes):
        vertices = [f.vertices.T for f in self._faces]
        poly = Poly3DCollection(vertices)
        ax.add_collection(poly)

if __name__ == '__main__':
    v = np.array([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0], [1.5, 1.5], [1.5, 2.5], [2.5, 2.5], [2.5, 1.5], [1.5, 1.5], [0, 0]], dtype=float).T
    f2d = Face2D(v)
    f3d = Face3D(f2d, '+x+y', np.array([0., 0., 0.]))
    cubesat = CubeSat([f3d])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cubesat.plot(ax)
    plt.show()





