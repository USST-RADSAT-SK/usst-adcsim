import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Union, List

# To calculate areas of polygons, the polygon is broken up into triangles and the areas are summed

# To calculate centroids of polygons, the geometric decomposition equation is used, and the equation of a centroid
# of a triangle. For geometric decomposition see wiki page: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon


class Features:
    def __init__(self, vertices: np.ndarray, reflectance_factor: int, color: str='r'):
        self.v = vertices  # in centimeters
        self.area = self._total_area()
        self.centroid = self._centroid()
        self.q = reflectance_factor
        self.color = color

        # convert to meters
        self.area = self.area / 10
        self.centroid = self.centroid / 10

        # the methods used here to calculate area and centroid only works for common shapes

    @staticmethod
    def _triangle_area(v):
        return abs(v[0, 0]*(v[1, 1] - v[2, 1]) + v[1, 0]*(v[2, 1] - v[0, 1]) + v[2, 0]*(v[0, 1] - v[1, 1]))/2

    def _total_area(self):
        total = 0
        for i in range(1, len(self.v) - 1):
            total += self._triangle_area(np.array([self.v[0], self.v[i], self.v[i+1]]))
        return total

    @staticmethod
    def _triangle_centroid(v):
        return np.array([(v[0, 0] + v[1, 0] + v[2, 0])/3, (v[0, 1] + v[1, 1] + v[2, 1])/3])

    def _centroid(self):
        total_x = 0
        total_y = 0
        for i in range(1, len(self.v) - 1):
            thing = np.array([self.v[0], self.v[i], self.v[i+1]])
            centroid = self._triangle_centroid(thing)
            t_area = self._triangle_area(thing)
            total_x += centroid[0] * t_area
            total_y += centroid[1] * t_area

        return np.array([total_x/self.area, total_y/self.area])


class BackgroundFeature:
    def __init__(self, area: Union[int, float], reflectance_factor: int, features: Union[List[Features], Features] = None):
        self.q = reflectance_factor

        # to calculate the area of the background feature: (l x w) - area of all features
        self.area = area - np.array([i.area for i in features]).sum()

        # to calculate the centroid of the background feature: treat all features as having negative areas and
        # then use the centroid equation
        centroid_x = 0
        centroid_y = 0  # centroids start at zero because the background area has centroid (0, 0)
        for i in range(len(features)):
            cent = features[i].centroid
            ar = features[i].area
            centroid_x -= cent[0] * ar
            centroid_y -= cent[1] * ar

        self.centroid = np.array([centroid_x/self.area, centroid_y/self.area])

        # convert to meters
        self.area = self.area / 10
        self.centroid = self.centroid / 10


class Faces:
    def __init__(self, name: str, side1: Union[int, float], side2: Union[int, float], reflectance_factor: int,
                 features: Union[List[Features], Features] = None, origin=(0, 0)):
        self.name = name
        self.side1 = side1
        self.side2 = side2
        self.features = features
        if isinstance(features, Features):  # make list if it is not
            self.features = [self.features]
        elif features is None:
            self.features = []

        self.features.append(BackgroundFeature(side1 * side2, reflectance_factor, self.features))

        if self.name == 'z+':
            self.sign1 = -1
            self.sign2 = 1
        elif self.name == 'z-':
            self.sign1 = 1
            self.sign2 = 1
        elif self.name == 'y+':
            self.sign1 = 1
            self.sign2 = 1
        elif self.name == 'y-':
            self.sign1 = -1
            self.sign2 = 1
        elif self.name == 'x+':
            self.sign1 = -1
            self.sign2 = 1
        elif self.name == 'x-':
            self.sign1 = 1
            self.sign2 = 1

        # TODO: add logic that makes sure that the features are contained within the rectangle

        # TODO: take origin into account
        # ^ This could come into play if we choose to define "faces"/surfaces that are not one of the six surfaces
        # of the Cube. I.e. an antenna.


class CubeSat:
    def __init__(self, faces: List[Faces]):
        self.faces = faces

        z = [[-0.5, -0.5, -1],
                          [0.5, -0.5, -1],
                          [0.5, 0.5, -1],
                          [-0.5, 0.5, -1],
                          [-0.5, -0.5, 1],
                          [0.5, -0.5, 1],
                          [0.5, 0.5, 1],
                          [-0.5, 0.5, 1]]

        verts = [
            [z[1], z[2], z[6], z[5]],
            [z[4], z[7], z[3], z[0]],
            [z[2], z[3], z[7], z[6]],
            [z[0], z[1], z[5], z[4]],
            [z[4], z[5], z[6], z[7]],
            [z[0], z[1], z[2], z[3]]
        ]

        self.facecolors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        for face in self.faces:
            for feature in face.features[:-1]:
                z = []
                if face.name == 'z+':
                    for vert in feature.v:
                        z.append([vert[0], vert[1], 1.05])
                if face.name == 'z-':
                    for vert in feature.v:
                        z.append([-vert[0], vert[1], -1.05])
                if face.name == 'y+':
                    for vert in feature.v:
                        z.append([-vert[0], 0.55, vert[1]])
                if face.name == 'y-':
                    for vert in feature.v:
                        z.append([vert[0], -0.55, vert[1]])
                if face.name == 'x+':
                    for vert in feature.v:
                        z.append([0.55, vert[0], vert[1]])
                if face.name == 'x-':
                    for vert in feature.v:
                        z.append([-0.55, -vert[0], vert[1]])
                verts.append(z)
                self.facecolors.append(feature.color)

        self.verts = verts

    def visualize(self):
        # plot sides
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        ax.add_collection3d(Poly3DCollection(self.verts, facecolors=self.facecolors, linewidths=1,
                                             edgecolors='r', alpha=.25))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


def create_solar_panel(start):
    s = start
    return np.array([[s[0], s[1]], [s[0] + 0.8, s[1]], [s[0] + 0.8, s[1] + 0.3], [s[0] + 0.6, s[1] + 0.4],
                     [s[0] + 0.2, s[1] + 0.4], [s[0], s[1] + 0.3]])


if __name__ == "__main__":
    # f1 = Features(np.array([[-0.3, -0.3], [-0.3, -0.1], [-0.4, -0.1], [-0.4, -0.3]]), 0.8)
    solar_zplus = Features(create_solar_panel([-0.4, -0.4]), 1, color='k')
    solar_zminus = Features(create_solar_panel([-0.4, -0.4]), 1, color='k')
    solar_yplus = Features(create_solar_panel([-0.4, -0.9]), 1, color='k')
    solar_yminus = Features(create_solar_panel([-0.4, -0.9]), 1, color='k')
    solar_xplus = Features(create_solar_panel([-0.4, -0.9]), 1, color='k')
    solar_xminus = Features(create_solar_panel([-0.4, -0.9]), 1, color='k')

    zplus = Faces('z+', 1, 1, 0.6, features=solar_zplus)
    zminus = Faces('z-', 1, 1, 0.6, features=solar_zminus)
    xplus = Faces('x+', 1, 1, 0.6, features=solar_xplus)
    xminus = Faces('x-', 1, 1, 0.6, features=solar_xminus)
    yplus = Faces('y+', 1, 1, 0.6, features=solar_yplus)
    yminus = Faces('y-', 1, 1, 0.6, features=solar_yminus)

    cubesat = CubeSat([xplus, xminus, yplus, yminus, zplus, zminus])
    cubesat.visualize()
    plt.show()
