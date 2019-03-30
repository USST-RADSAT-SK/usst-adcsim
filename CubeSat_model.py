import numpy as np

# To calculate areas of polygons, the polygon is broken up into triangles and the areas are summed

# To calculate centroids of polygons, the geometric decomposition equation is used, and the equation of a centroid
# of a triangle. For geometric decomposition see wiki page: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon


class Features:
    def __init__(self, vertices, reflectance_factor):
        self.v = vertices
        self.area = self._total_area()
        self.centroid = self._centroid()
        self.q = reflectance_factor

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
    def __init__(self, area, features, reflectance_factor):
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


class Faces:
    def __init__(self, side1, side2, features, reflectance_factor, origin=(0, 0)):
        self.features = features
        if isinstance(features, Features):  # make list if it is not
            self.features = [self.features]

        self.features.append(BackgroundFeature(side1 * side2, self.features, reflectance_factor))

        # TODO: add logic that makes sure that the features are contained within the rectangle

        # TODO: take origin into account
        # ^ This could come into play if we choose to define "faces"/surfaces that are not one of the six surfaces
        # of the Cube. I.e. an antenna.
