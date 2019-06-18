"""
This file contains all the framework for our CubeSat model. The CubeSat model is used to define a geometry.
This geometry is important for solar pressure torque and aerodynamic torque calculations. As well as solar
power generation calculations.

The model can also be input to our animation code so that it is view in the animation matplotlib window

A quick summary of the classes used and their purposes is presented below:
 - Face2D: This class specifies a two dimensional shape and some assorted surface properties.
 - Face3D: This class is a wrapper around a single Face2D object which positions the 2D face in 3D space.
 - Polygon3D: This class represents a 3D polygon by holding onto a list of Face3D objects. It allows you to
     translate and rotate the polygon as a whole.
 - Cubesat3D: This class inherits from Polygon3D but adds useful information such as center of mass, moment of
     inertia and magnetic properties.
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Union, List
from adcsim.hysteresis_rod import HysteresisRod


class Face2D:
    """
    Defines a 2D shape as well as various surface properties.

    Rules for specifying vertices:
     - a shape is specified by vertices listed in the counterclockwise direction with the first/last vertex repeated
     - a hole is specified by vertices listed in the clockwise direction with the first/last vertex repeated
     - each distinct shape or hole specified by the same list should be separated by the first vertex of the entire
     list, which should also be repeated at the end of the list
     - ex: two distinct triangles ABC and abc would be represented by ABCAabcaA
     - ex: triangle ABC with hole abc would be represented by ABCAacbaA
    These rules allow the algorithms for calculating the area and the centroid to work with multiple shapes/holes.

    It may be easier to specify each shape/hole separately and to combine them using addition and subtraction operators
    which will combine them as specified above. Subtraction reverses the order of the vertices, turning shapes into
    holes and vice versa.

    Numpy arrays of length 2 can also be added to and subtracted from Face2D objects - this will simply translate all
    vertices.

    The surface properties are called by functions that calculate aerodynamic disturbance torques, solar pressure
    disturbance torques, or solar power.
    """
    def __init__(self, vertices: np.ndarray, sigma_n: float=0.8, sigma_t: float=0.8, spec_ref_coeff: float=0.6,
                 diff_ref_coeff: float=0.0, accommodation_coeff: float=0.9, solar_power_efficiency: float=None):
        """
        Parameters
        ----------
        vertices : np.ndarray
            List of 2D vertices, shape (2, n) - see class docstring for ordering.
        sigma_n : float
            Normal momentum exchange coefficient, used to compute the aerodynamic force on a surface.
        sigma_t : float
            Tangential momentum exchange coefficient, used to compute the aerodynamic force on a surface.
        spec_ref_coeff : float
            Specular reflectance coefficient
        diff_ref_coeff : float
            Diffuse reflectance coefficient
        accommodation_coeff : float
            Accommodation coefficient
        solar_power_efficiency : float
            Solar power efficiency
        """

        assert(len(vertices.shape) == 2)
        assert(vertices.shape[0] == 2)

        self._vertices = vertices
        self._area = self._polygon_area(vertices)
        self._centroid = self._polygon_centroid(vertices)

        self._sigma_n = sigma_n
        self._sigma_t = sigma_t
        self._spec_ref_coeff = spec_ref_coeff
        self._diff_ref_coeff = diff_ref_coeff
        self._accommodation_coeff = accommodation_coeff
        self._solar_power_efficiency = solar_power_efficiency
        if self._solar_power_efficiency is not None:
            self.is_solar_panel = True
        else:
            self.is_solar_panel = False

        self._kwargs = dict(
            sigma_n=sigma_n,
            sigma_t=sigma_t,
            spec_ref_coeff=spec_ref_coeff,
            diff_ref_coeff=diff_ref_coeff,
            accommodation_coeff=accommodation_coeff,
            solar_power_efficiency=solar_power_efficiency
        )

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

    @property
    def accommodation_coeff(self):
        return self._accommodation_coeff

    @property
    def spec_ref_coeff(self):
        return self._spec_ref_coeff

    @property
    def diff_ref_coeff(self):
        return self._diff_ref_coeff

    @property
    def solar_power_efficiency(self):
        return self._solar_power_efficiency

    def copy(self):
        return Face2D(self.vertices.copy(), **self._kwargs)

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return Face2D(self.vertices + other.reshape(2, 1), **self._kwargs)
        elif isinstance(other, Face2D):
            return Face2D(np.concatenate((self.vertices, other.vertices, self.vertices[:, :1]), axis=1),
                          **self._kwargs)
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
            return Face2D(self.vertices - other.reshape(2, 1), **self._kwargs)
        elif isinstance(other, Face2D):
            return Face2D(np.concatenate((self.vertices, other.vertices[:, ::-1], self.vertices[:, :1]), axis=1),
                          **self._kwargs)
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
        https://en.wikipedia.org/wiki/Polygon#Area
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
    """
    This class is a wrapper around a Face2D object that positions the 2D shape in 3D space.

    The raw Face2D object is assumed to start in the xy plane. It is then rotated by the rotation matrix passed to the
    orientation parameter and translated by the vector passed to the translation parameter.

    Certain rotations can be specified by string shortcuts instead of rotation matrices:
     - axis directions are specified by a sign ('+' or '-') and an axis ('x' or 'y' or 'z')
     - the first two characters represent the new direction for the old positive x axis
     - the second two characters represent the new direction for the old positive y axis
     - ex: '+x+y' would cause no rotation, '+y-x' would cause 90 degree rotation about the positive z axis

    Changing the orientation property or the translation property will override its previous value and apply the new
    rotation or translation to the original position of the 2D face in the xy plane. When either of these
    properties is updated, the (possibly new) rotation matrix is applied to the original face first, and the
    (possibly new) translation vector is applied second.

    To rotate and translate the face relative to its current orientation and translation, the rotate method and the
    translate method should be called.
    """
    _unit_vectors = {
        '+x': np.array([1., 0., 0.]),
        '-x': np.array([-1., 0., 0.]),
        '+y': np.array([0., 1., 0.]),
        '-y': np.array([0., -1., 0.]),
        '+z': np.array([0., 0., 1.]),
        '-z': np.array([0., 0., -1.]),
    }
    def __init__(self, face: Face2D, orientation: Union[str, np.ndarray]='+x+y', translation: np.ndarray=np.zeros(3), name='', color='k'):
        self._orientation = np.eye(3)
        self._translation = np.zeros((3, 1))
        self.face = face
        self.is_solar_panel = self.face.is_solar_panel
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

    @property
    def sigma_n(self):
        return self.face.sigma_n

    @property
    def sigma_t(self):
        return self.face.sigma_t

    @property
    def accommodation_coeff(self):
        return self.face.accommodation_coeff

    @property
    def spec_ref_coeff(self):
        return self.face.spec_ref_coeff

    @property
    def diff_ref_coeff(self):
        return self.face.diff_ref_coeff

    @property
    def solar_power_efficiency(self):
        return self.face.solar_power_efficiency

    def rotate(self, dcm: np.ndarray=None, axis: Union[str, np.ndarray]=None, angle: float=None):
        if dcm is not None:
            rotation_matrix = dcm.T
        elif axis is not None and angle is not None:
            if isinstance(axis, str):
                axis = self._unit_vector_from_string(axis)
            axis *= 1.0 / np.linalg.norm(axis)
            x, y, z = axis[0], axis[1], axis[2]
            c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            rotation_matrix = np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                                        [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                                        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])
        elif axis is not None and angle is None:
            e1 = self._unit_vector_from_string(axis[:2])
            e2 = self._unit_vector_from_string(axis[2:])
            e3 = np.cross(e1, e2)
            rotation_matrix = np.column_stack((e1, e2, e3))
        else:
            print('Face3D.rotate: EITHER THE DCM OR TWO CARTESIAN AXES OR AN AXIS AND AND ANGLE MUST BE SPECIFIED')
            return
        self.translation = rotation_matrix @ self.translation
        self.orientation = rotation_matrix @ self.orientation

    def translate(self, vector: np.ndarray):
        self.translation += vector.reshape(3, 1)

    def copy(self):
        return Face3D(face=self.face.copy(), orientation=self.orientation.copy(), translation=self.translation.copy(),
                      name=self.name, color=self.color)

    def _unit_vector_from_string(self, string):
        if string in self._unit_vectors:
            return self._unit_vectors[string]
        else:
            print('Face3D._unit_vector_from_string: INVALID STRING')

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
    """
    A 3D polygon, represented internally as a list of Face3D objects.

    The rotate and translate methods allow the whole polygon to be rotated and translated as a whole.

    A list of 3D faces can be specified on initialization, or an empty list can be passed at initialization and faces
    can be added throuth the faces property.
    """

    def __init__(self, faces: List[Face3D]):
        self._faces = [face.copy() for face in faces]
        self.solar_panel_faces = [face.copy() for face in faces if face.is_solar_panel]

    def plot(self):
        max = -np.inf
        min = np.inf
        for face in self.faces:
            max = np.max([face.vertices.max(), max])
            min = np.min([face.vertices.min(), min])
        pad = 0.25 * (max - min)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(min - pad, max + pad)
        ax.set_ylim(min - pad, max + pad)
        ax.set_zlim(min - pad, max + pad)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        vertices = [f.vertices.T for f in self._faces]
        colors = [f.color for f in self._faces]
        poly = Poly3DCollection(vertices, facecolors=colors)
        ax.add_collection(poly)

    @property
    def faces(self):
        return self._faces

    def rotate(self, dcm: np.ndarray=None, axis: Union[str, np.ndarray]=None, angle: float=None):
        for face in self.faces:
            face.rotate(dcm=dcm, axis=axis, angle=angle)

    def translate(self, vector: np.ndarray):
        for face in self.faces:
            face.translate(vector)

class CubeSat(Polygons3D):
    """
    Inherits from Polygons3D but adds the following properties:
     - center of mass           (numpy array of length 3)       [m]
     - moment of inertia        (numpy array of shape (3, 3))   [kg m^2]
     - magnetic moment          (numpy array of length 3)       [T]
     - residual magnetic moment (numpy array of length 3)       [T]
     - hysteresis rods          (list of HysteresisRod objects)
    """
    def __init__(self, faces: List[Face3D], center_of_mass: np.ndarray, inertia: np.ndarray,
                 residual_magnetic_moment: np.ndarray = np.zeros(3), magnetic_moment: np.ndarray = np.zeros(3),
                 hyst_rods: List[HysteresisRod] = None):
        self._com = center_of_mass
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)
        self._residual_magnetic_moment = residual_magnetic_moment
        self._magnetic_moment = magnetic_moment
        self._total_magnetic_moment = residual_magnetic_moment + magnetic_moment
        self._hyst_rods = [] if hyst_rods is None else hyst_rods
        super().__init__(faces)

    @property
    def center_of_mass(self):
        return self._com

    @property
    def inertia(self):
        return self._inertia

    @property
    def inertia_inv(self):
        return self._inertia_inv

    @property
    def hyst_rods(self):
        return self._hyst_rods

    @property
    def residual_magnetic_moment(self):
        return self._residual_magnetic_moment

    @property
    def magnetic_moment(self):
        return self._magnetic_moment

    @property
    def total_magnetic_moment(self):
        return self._total_magnetic_moment

