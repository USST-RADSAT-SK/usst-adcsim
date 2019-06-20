"""
This file contains all the framework for the matplotlib animations that can be done for a given attitude simulation.
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from typing import List, Union
from adcsim.CubeSat_model import CubeSat


class DrawingVectors:
    def __init__(self, data: np.ndarray, draw_type: str, color: Union[List[str], str] = 'r',
                 label: Union[List[str], str] = 'NA', length: Union[List[int], float, int] = 6):

        # valid types: single, double, axes
        assert (draw_type == 'single' or draw_type == 'double' or draw_type == 'axes'), "invalid draw_type"

        if len(data) > 3:
            self.changing = True
        else:
            self.changing = False

        if self.changing:
            check_data = data[0]
        else:
            check_data = data

        if draw_type == 'single':
            if not isinstance(color, str) or not isinstance(label, str) or not isinstance(length, (float, int)):
                raise Exception("for draw_type == 'single' color, label, and length should be single values")

        if draw_type == 'double':
            if (not isinstance(color, str) and isinstance(color, list) and not len(color) == 2) or \
                    (not isinstance(label, str) and isinstance(label, list) and not len(label) == 2) or \
                    (not isinstance(length, str) and isinstance(length, list) and not len(length) == 2):
                raise Exception("for draw_type == 'double' color, label, and length should be single values or "
                                "length 2 lists")

        if draw_type == 'axes':
            assert (check_data.shape == (3, 3)), "can only have draw_type == 'axes' for a 3x3 DCM matrix"

            if (not isinstance(color, str) and isinstance(color, list) and not len(color) == 3) or \
                    (not isinstance(label, str) and isinstance(label, list) and not len(label) == 3) or \
                    (not isinstance(length, str) and isinstance(length, list) and not len(length) == 3):
                raise Exception("for draw_type == 'axes' color, label, and length should be single values or "
                                "length 3 lists")

        self.data = data
        self.draw_type = draw_type
        self.color = color
        self.label = label
        self.length = length

        if isinstance(color, str):
            self.color = [self.color]

        if isinstance(label, str):
            self.label = [self.label]

        if isinstance(length, (float, int)):
            self.length = [self.length]


class AdditionalPlots:
    def __init__(self, xdata: np.ndarray, ydata: np.ndarray, labels: Union[List[str], str] = ('X', 'Y', 'Z'),
                 title='NA', xlabel='time', ylabel='NA', groundtrack=False):
        self.xdata = xdata
        self.ydata = ydata
        self.labels = labels
        self.groundtrack = groundtrack

        if groundtrack:
            self.title = ''
            self.xlabel = ''
            self.ylabel = ''
            self.xmin = -180
            self.xmax = 180
            self.ymin = -85
            self.ymax = 85
            self.projection = ccrs.PlateCarree()
        else:
            self.title = title
            self.xlabel = xlabel
            self.ylabel = ylabel
            self.xmin = xdata.min()
            self.xmax = xdata.max()
            self.ymin = ydata.min()
            self.ymax = ydata.max()
            self.projection = None


class AnimateAttitude:
    def __init__(self, dcm, draw_vector: Union[List[DrawingVectors], DrawingVectors] = None,
                 additional_plots: Union[List[AdditionalPlots], AdditionalPlots] = None,
                 x_mag=1, y_mag=1, z_mag=2, cubesat_model: CubeSat = None):
        self.dcm = np.transpose(dcm.copy(), (0, 2, 1))
        # ^ must get [NB] rather than [BN], because the vertices are in the B frame.
        self.draw_vec = draw_vector
        if self.draw_vec is None:
            self.draw_vec = []
        self.single_draw_vec = isinstance(self.draw_vec, DrawingVectors)
        if self.single_draw_vec:
            self.draw_vec = [self.draw_vec]
        self.additional_plots = additional_plots
        self.single_additional_plot = isinstance(self.additional_plots, AdditionalPlots)
        if self.single_additional_plot:
            self.additional_plots = [self.additional_plots]
        # self.V = np.array([[-x_mag, -y_mag, -z_mag],
        #                   [x_mag, -y_mag, -z_mag],
        #                   [x_mag, y_mag, -z_mag],
        #                   [-x_mag, y_mag, -z_mag],
        #                   [-x_mag, -y_mag, z_mag],
        #                   [x_mag, -y_mag, z_mag],
        #                   [x_mag, y_mag, z_mag],
        #                   [-x_mag, y_mag, z_mag]])
        self.cubesat_model = cubesat_model
        self.facecolors = []
        for face in cubesat_model.faces:
            self.facecolors.append(face.color)

        # duplicate data so that animations can run more smoothly (because you dont have to do checks in the code all
        # the time)
        for vect in self.draw_vec:
            if not vect.changing:
                vect.data = np.array([vect.data] * len(dcm))

            if vect.draw_type == 'double' and len(vect.color) == 1:
                vect.color = [vect.color[0], vect.color[0]]
            if vect.draw_type == 'axes' and len(vect.color) == 1:
                vect.color = [vect.color[0], vect.color[0], vect.color[0]]

            if vect.draw_type == 'double' and len(vect.label) == 1:
                vect.label = [vect.label[0], vect.label[0]]
            if vect.draw_type == 'axes' and len(vect.label) == 1:
                vect.label = [vect.label[0], vect.label[0], vect.label[0]]

            if vect.draw_type == 'double' and len(vect.length) == 1:
                vect.length = [vect.length[0], vect.length[0]]
            if vect.draw_type == 'axes' and len(vect.length) == 1:
                vect.length = [vect.length[0], vect.length[0], vect.length[0]]

    def _animate(self, ax, i):
        ax.set_xlim(-0.2, 0.2)  # hardcoded for now
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 0.2)

        # z = (self.dcm[i] @ self.V.T).T
        #
        # # plot vertices
        # ax.scatter3D(z[:, 0], z[:, 1], z[:, 2])
        #
        # # list of sides' polygons of figure
        # verts = [[z[0], z[1], z[2], z[3]],
        #          [z[4], z[5], z[6], z[7]],
        #          [z[0], z[1], z[5], z[4]],
        #          [z[2], z[3], z[7], z[6]],
        #          [z[1], z[2], z[6], z[5]],
        #          [z[4], z[7], z[3], z[0]]]
        verts = []
        for face in self.cubesat_model.faces:
            verts.append((self.dcm[i] @ np.array(face.vertices)).T.tolist())

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts, facecolors=self.facecolors, linewidths=1, edgecolors='r', alpha=.25))
        # plot arrows
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if self.draw_vec is not None:
            for vect in self.draw_vec:
                self._animate_draw_vec(ax, i, vect)
        ax.legend()

    @staticmethod
    def _animate_draw_vec(ax, i, vect):
        if vect.changing:
            ve = vect.data[i]
        else:
            ve = vect.data
        if vect.draw_type == 'single':
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=vect.length[0], color=vect.color[0], label=vect.label[0])
        if vect.draw_type == 'double':
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=vect.length[0], color=vect.color[0], label=vect.label[0])
            ax.quiver(0, 0, 0, -ve[0], -ve[1], -ve[2], length=vect.length[1], color=vect.color[1],
                      label='- ' + vect.label[1])
        if vect.draw_type == 'axes':
            for j in range(3):
                ax.quiver(0, 0, 0, ve[j, 0], ve[j, 1], ve[j, 2], length=vect.length[j], color=vect.color[j],
                          label=vect.label[j])

    @staticmethod
    def _plot(ax, i, ap):
        ax.set_xlim(ap.xmin, ap.xmax)
        ax.set_ylim(ap.ymin, ap.ymax)
        ax.plot(ap.xdata[:i+1], ap.ydata[:i+1])
        ax.set_title(ap.title)
        ax.set_ylabel(ap.ylabel)
        ax.set_xlabel(ap.xlabel)
        plt.gca().legend(ap.labels)

    @staticmethod
    def _plot_ground_track(ax, i, ap):
        ax.coastlines()
        ax.set_xlim(ap.xmin, ap.xmax)
        ax.set_ylim(ap.ymin, ap.ymax)
        ax.plot(ap.xdata[:i+1], ap.ydata[:i+1], '.')

    def animate(self):
        fig = plt.figure()
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            self._animate(ax, i)
            plt.pause(0.01)

    def animate_and_plot(self):
        if self.additional_plots is None:
            raise Exception("There are no additional plots")
        n = len(self.additional_plots)

        fig = plt.figure(figsize=(15, 5))
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            self._animate(ax, i)
            for j, ap in enumerate(self.additional_plots):
                ax = fig.add_subplot(n, 2, 2*(j+1), projection=ap.projection)
                if ap.groundtrack:
                    self._plot_ground_track(ax, i, ap)
                else:
                    self._plot(ax, i, ap)
            plt.pause(0.01)

    def single_instant(self, index):
        # this method will plot a single instance in time of the CubeSat's attitude. This can be useful for debugging.
        # the time is given by the index the integration (i.e. the index of all of the arrays involved)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self._animate(ax, index)


class AnimateAttitudeInside:
    def __init__(self, cubesat_model: CubeSat):
        self.cubesat_model = cubesat_model
        self.facecolors = []
        for face in cubesat_model.faces:
            self.facecolors.append(face.color)

    def _animate(self, ax, dcm, draw_vector):
        ax.set_xlim(-0.2, 0.2)  # hardcoded for now
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 0.2)

        verts = []
        for face in self.cubesat_model.faces:
            verts.append((dcm @ np.array(face.vertices)).T.tolist())

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts, facecolors=self.facecolors, linewidths=1, edgecolors='r', alpha=.25))
        # plot arrows
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if draw_vector is not None:
            for vect in draw_vector:
                self._animate_draw_vec(ax, vect)
        ax.legend()

    def animate(self, fig, dcm, draw_vector: Union[List[DrawingVectors], DrawingVectors] = None):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        self._animate(ax, dcm, draw_vector)
        plt.pause(0.00000000001)

    def animate_and_plot(self, fig, dcm, draw_vector: Union[List[DrawingVectors], DrawingVectors] = None,
                         additional_plots: Union[List[AdditionalPlots], AdditionalPlots] = None):
        n = len(additional_plots)
        plt.clf()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        self._animate(ax, dcm, draw_vector)
        for j, ap in enumerate(additional_plots):
            ax = fig.add_subplot(n, 2, 2*(j+1), projection=ap.projection)
            if ap.groundtrack:
                self._plot_ground_track(ax, ap)
            else:
                self._plot(ax, ap)
        plt.pause(0.00000000001)

    @staticmethod
    def _animate_draw_vec(ax, vect):
        ve = vect.data
        if vect.draw_type == 'single':
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=vect.length[0], color=vect.color[0], label=vect.label[0])
        if vect.draw_type == 'double':
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=vect.length[0], color=vect.color[0], label=vect.label[0])
            ax.quiver(0, 0, 0, -ve[0], -ve[1], -ve[2], length=vect.length[1], color=vect.color[1],
                      label='- ' + vect.label[1])
        if vect.draw_type == 'axes':
            for j in range(3):
                ax.quiver(0, 0, 0, ve[j, 0], ve[j, 1], ve[j, 2], length=vect.length[j], color=vect.color[j],
                          label=vect.label[j])

    @staticmethod
    def _plot(ax, ap):
        ax.set_xlim(ap.xdata.min(), ap.xdata.max())
        ax.set_ylim(ap.ydata.min(), ap.ydata.max())
        ax.plot(ap.xdata, ap.ydata)
        ax.set_title(ap.title)
        ax.set_ylabel(ap.ylabel)
        ax.set_xlabel(ap.xlabel)
        plt.gca().legend(ap.labels)

    @staticmethod
    def _plot_ground_track(ax, ap):
        ax.coastlines()
        ax.set_xlim(ap.xmin, ap.xmax)
        ax.set_ylim(ap.ymin, ap.ymax)
        ax.plot(ap.xdata, ap.ydata, '.')
