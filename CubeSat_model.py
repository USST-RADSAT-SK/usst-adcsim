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
        for v in hole_vertices:
            assert(len(v.shape) == 2)
            assert(v.shape[0] == 2)

        self._vertices = vertices
        self._hole_vertices = hole_vertices

        self._area = self._polygon_area(vertices)
        area_x_centroid = self._area * self._polygon_centroid(vertices)
        for v in hole_vertices:
            a = self._polygon_area(v)
            c = self._polygon_centroid(v)
            area_x_centroid -= a * c
            self._area -= a
        self._centroid = area_x_centroid / self._area

    @property
    def vertices(self):
        return self._vertices

    @property
    def num_vertices(self):
        return self._vertices.shape[1]

    @property
    def hole_vertices(self):
        return self._hole_vertices

    @property
    def num_hole_vertices(self):
        return [v.shape[1] for v in self._hole_vertices]

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
        a = v[0, n - 1] * v[1, 0] - v[0, 0] * v[1, n - 1]
        for i in range(n - 1):
            a += v[0, i] * v[1, i + 1] - v[0, i + 1] * v[1, i]
        return a

    @staticmethod
    def _polygon_centroid(v: np.ndarray):
        """
        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        """
        n = v.shape[1]
        cx = (v[0, n - 1] + v[0, 0]) * (v[0, n - 1] * v[1, 0] - v[0, 0] * v[1, n - 1])
        cy = (v[1, n - 1] + v[1, 0]) * (v[0, n - 1] * v[1, 0] - v[0, 0] * v[1, n - 1])
        for i in range(n - 1):
            cx += (v[0, i] + v[0, i + 1]) * (v[0, i] * v[1, i + 1] - v[0, i + 1] * v[1, i])
            cy += (v[1, i] + v[1, i + 1]) * (v[0, i] * v[1, i + 1] - v[0, i + 1] * v[1, i])
        return np.array([cx, cy]) / Face2D._polygon_area(v)


class Face3D:
    def __init__(self, face: Face2D, orientation, translation):
        # copy vertices and centroid into 3D vectors in the xy plane
        self._vertices = np.zeros((3, face.num_vertices))
        self._vertices[:2, :] = face.vertices

        self._hole_vertices = [np.zeros((3, n)) for n in face.num_hole_vertices]
        for i in range(len(self._hole_vertices)):
            self._hole_vertices[i][:2, :] = face.hole_vertices[i]

        self._centroid = np.array([*face.centroid, 0.0])

        # rotate vertices and centroid into the correct plane

        # translate vertices and centroid


