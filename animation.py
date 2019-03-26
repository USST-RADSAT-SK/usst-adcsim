import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class AnimateAttitude:
    def __init__(self, dcm, dcm_reference=None, draw_vector=None, x_mag=1, y_mag=1, z_mag=2):
        self.dcm = np.transpose(dcm.copy(), (0, 2, 1))
        # ^ must get [NB] rather than [BN], because the vertices are in the B frame.
        self.dcm_reference = dcm_reference
        self.draw_vec = draw_vector
        self.V = np.array([[-x_mag, -y_mag, -z_mag],
                          [x_mag, -y_mag, -z_mag],
                          [x_mag, y_mag, -z_mag],
                          [-x_mag, y_mag, -z_mag],
                          [-x_mag, -y_mag, z_mag],
                          [x_mag, -y_mag, z_mag],
                          [x_mag, y_mag, z_mag],
                          [-x_mag, y_mag, z_mag]])

        if self.draw_vec is not None and len(self.draw_vec) > 3:
            self.draw_vec_changing = True
        else:
            self.draw_vec_changing = False

        if self.dcm_reference is not None and len(self.dcm_reference) > 3:
            self.dcm_reference_changing = True
        else:
            self.dcm_reference_changing = False

    def _get_draw_vec(self, i):
        if not self.draw_vec_changing:
            return self.draw_vec
        return self.draw_vec[i]

    def _get_dcm_reference(self, i):
        if not self.dcm_reference_changing:
            return self.dcm_reference
        return self.dcm_reference[i]

    def _animate(self, ax, i):
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        z = (self.dcm[i] @ self.V.T).T

        # plot vertices
        ax.scatter3D(z[:, 0], z[:, 1], z[:, 2])

        # list of sides' polygons of figure
        verts = [[z[0], z[1], z[2], z[3]],
                 [z[4], z[5], z[6], z[7]],
                 [z[0], z[1], z[5], z[4]],
                 [z[2], z[3], z[7], z[6]],
                 [z[1], z[2], z[6], z[5]],
                 [z[4], z[7], z[3], z[0]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        ax.quiver(0, 0, 0, self.dcm[i][0, 0], self.dcm[i][1, 0], self.dcm[i][2, 0], length=4)
        ax.quiver(0, 0, 0, self.dcm[i][0, 1], self.dcm[i][1, 1], self.dcm[i][2, 1], length=4)
        ax.quiver(0, 0, 0, self.dcm[i][0, 2], self.dcm[i][1, 2], self.dcm[i][2, 2], length=4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def _animate_reference(self, ax, i):
        de = self._get_dcm_reference(i)
        ax.quiver(0, 0, 0, de[0, 0], de[0, 1], de[0, 2], length=4, color='r')
        ax.quiver(0, 0, 0, de[1, 0], de[1, 1], de[1, 2], length=4, color='r')
        ax.quiver(0, 0, 0, de[2, 0], de[2, 1], de[2, 2], length=4, color='r')

    def _animate_draw_vec(self, ax, i, draw_vec_type):
        ve = self._get_draw_vec(i)
        ve = 2 * ve / np.linalg.norm(ve)
        ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=6, color='r')
        if draw_vec_type == 'double':
            ax.quiver(0, 0, 0, -ve[0], -ve[1], -ve[2], length=6, color='r')

    def animate(self, draw_vec_type='single'):
        fig = plt.figure()
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            self._animate(ax, i)
            if self.dcm_reference is not None:
                self._animate_reference(ax, i)
            if self.draw_vec is not None:
                self._animate_draw_vec(ax, i, draw_vec_type)

            plt.pause(0.01)

    @staticmethod
    def _plot(ax, i, xdata, ydata, title, ylabel, xlabel):
        ax.set_xlim(0, xdata.max())
        ax.set_ylim(ydata.min(), ydata.max())
        ax.plot(xdata[:i+1], ydata[:i+1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    @staticmethod
    def _plot_ground_track(ax, i, lats, longs):
        ax.coastlines()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-85, 85)
        ax.plot(longs[:i+1], lats[:i+1], '.')

    def animate_and_plot(self, xdata, ydata, title='NA', ylabel='NA', xlabel='time', draw_vec_type='single'):
        fig = plt.figure(figsize=(15, 5))
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            self._animate(ax, i)
            if self.dcm_reference is not None:
                self._animate_reference(ax, i)
            if self.draw_vec is not None:
                self._animate_draw_vec(ax, i, draw_vec_type)
            ax = fig.add_subplot(1, 2, 2)
            self._plot(ax, i, xdata, ydata, title, ylabel, xlabel)
            plt.pause(0.01)

    def animate_and_2_plots(self, xdata1, ydata1, xdata2, ydata2, title1='NA', ylabel1='NA', xlabel1='time',
                            title2='NA', ylabel2='NA', xlabel2='time', draw_vec_type='single'):
        fig = plt.figure(figsize=(15, 5))
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            self._animate(ax, i)
            if self.dcm_reference is not None:
                self._animate_reference(ax, i)
            if self.draw_vec is not None:
                self._animate_draw_vec(ax, i, draw_vec_type)
            ax = fig.add_subplot(2, 2, 2)
            self._plot(ax, i, xdata1, ydata1, title1, ylabel1, xlabel1)
            ax = fig.add_subplot(2, 2, 4)
            self._plot(ax, i, xdata2, ydata2, title2, ylabel2, xlabel2)
            plt.pause(0.01)

    def animate_and_n_plots(self, xdata_arr, ydata_arr, title_arr='NA', ylabel_arr='NA', xlabel_arr='time',
                            draw_vec_type='single'):
        fig = plt.figure(figsize=(18, 8))
        n = len(ydata_arr)
        if title_arr == 'NA':
            title_arr = ['NA'] * n
        if ylabel_arr == 'NA':
            ylabel_arr = ['NA'] * n
        if xlabel_arr == 'time':
            xlabel_arr = ['time'] * n
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(1, n, 1, projection='3d')
            self._animate(ax, i)
            if self.dcm_reference is not None:
                self._animate_reference(ax, i)
            if self.draw_vec is not None:
                self._animate_draw_vec(ax, i, draw_vec_type)
            for j in range(n):
                ax = fig.add_subplot(n, 2, 2*(j+1))
                if len(xdata_arr) != n:
                    self._plot(ax, i, xdata_arr, ydata_arr[j], title_arr[j], ylabel_arr[j], xlabel_arr[j])
                else:
                    self._plot(ax, i, xdata_arr[j], ydata_arr[j], title_arr[j], ylabel_arr[j], xlabel_arr[j])
            plt.pause(0.01)

    def animate_and_ground_track(self, lats, lngs, draw_vec_type='single'):
        fig = plt.figure(figsize=(15, 5))
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            self._animate(ax, i)
            if self.dcm_reference is not None:
                self._animate_reference(ax, i)
            if self.draw_vec is not None:
                self._animate_draw_vec(ax, i, draw_vec_type)
            ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
            self._plot_ground_track(ax, i, lats, lngs)
            plt.pause(0.01)

    def animate_and_ground_track_and_plot(self, lats, lngs, xdata, ydata, title='NA', ylabel='NA', xlabel='time',
                                          draw_vec_type='single'):
        fig = plt.figure(figsize=(15, 5))
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            self._animate(ax, i)
            if self.dcm_reference is not None:
                self._animate_reference(ax, i)
            if self.draw_vec is not None:
                self._animate_draw_vec(ax, i, draw_vec_type)
            ax = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
            self._plot_ground_track(ax, i, lats, lngs)
            ax = fig.add_subplot(2, 2, 4)
            self._plot(ax, i, xdata, ydata, title, ylabel, xlabel)
            plt.pause(0.01)

    def animate_and_ground_track_and_n_plots(self, lats, lngs, xdata_arr, ydata_arr, title_arr='NA', ylabel_arr='NA',
                                             xlabel_arr='time', draw_vec_type='single'):
        fig = plt.figure(figsize=(18, 8))
        n = len(ydata_arr)
        if title_arr == 'NA':
            title_arr = ['NA'] * n
        if ylabel_arr == 'NA':
            ylabel_arr = ['NA'] * n
        if xlabel_arr == 'time':
            xlabel_arr = ['time'] * n
        for i in range(len(self.dcm)):
            plt.clf()
            ax = fig.add_subplot(1, n+1, 1, projection='3d')
            self._animate(ax, i)
            if self.dcm_reference is not None:
                self._animate_reference(ax, i)
            if self.draw_vec is not None:
                self._animate_draw_vec(ax, i, draw_vec_type)
            ax = fig.add_subplot(n+1, 2, 2, projection=ccrs.PlateCarree())
            self._plot_ground_track(ax, i, lats, lngs)
            for j in range(n):
                ax = fig.add_subplot(n+1, 2, 2*(j+2))
                if len(xdata_arr) != n:
                    self._plot(ax, i, xdata_arr, ydata_arr[j], title_arr[j], ylabel_arr[j], xlabel_arr[j])
                else:
                    self._plot(ax, i, xdata_arr[j], ydata_arr[j], title_arr[j], ylabel_arr[j], xlabel_arr[j])
            plt.pause(0.01)
