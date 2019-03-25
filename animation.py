import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
z_mag = 2
x_mag = 1
y_mag = 1


class AnimateAttitude:
    def __init__(self, dcm, dcm_reference=None, draw_vector=None, xmag=1, ymag=1, zmag=2):
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

        Z = (self.dcm[i] @ self.V.T).T

        # plot vertices
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

        # list of sides' polygons of figure
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                 [Z[4], Z[5], Z[6], Z[7]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[2], Z[6], Z[5]],
                 [Z[4], Z[7], Z[3], Z[0]]]

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




def animate_attitude(dcm, dcm_reference=None, vec=None):
    dcm1 = dcm.copy()
    V = np.array([[-x_mag, -y_mag, -z_mag],
                  [x_mag, -y_mag, -z_mag],
                  [x_mag, y_mag, -z_mag],
                  [-x_mag, y_mag, -z_mag],
                  [-x_mag, -y_mag, z_mag],
                  [x_mag, -y_mag, z_mag],
                  [x_mag, y_mag, z_mag],
                  [-x_mag, y_mag, z_mag]])

    if vec is not None and len(vec) > 3:
        changing = True
    else:
        changing = False

    def get_vec(b, i):
        if not changing:
            return b
        return b[i]

    if dcm_reference is not None and len(dcm_reference) > 3:
        changing_ref = True
    else:
        changing_ref = False

    def get_dcm_ref(b, i):
        if not changing_ref:
            return b
        return b[i]

    fig = plt.figure()

    for i in range(len(dcm1)):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        dcm1[i] = dcm1[i].T  # must get [NB] rather than [BN], because the vertices are in the B frame.

        Z = (dcm1[i] @ V.T).T

        # plot vertices
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

        # list of sides' polygons of figure
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                 [Z[4], Z[5], Z[6], Z[7]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[2], Z[6], Z[5]],
                 [Z[4], Z[7], Z[3], Z[0]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts,
                                             facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.quiver(0, 0, 0, dcm1[i][0, 0], dcm1[i][1, 0], dcm1[i][2, 0], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 1], dcm1[i][1, 1], dcm1[i][2, 1], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 2], dcm1[i][1, 2], dcm1[i][2, 2], length=4)

        if dcm_reference is not None:
            de = get_dcm_ref(dcm_reference, i)
            ax.quiver(0, 0, 0, de[0, 0], de[0, 1], de[0, 2], length=4, color='r')
            ax.quiver(0, 0, 0, de[1, 0], de[1, 1], de[1, 2], length=4, color='r')
            ax.quiver(0, 0, 0, de[2, 0], de[2, 1], de[2, 2], length=4, color='r')

        if vec is not None:
            ve = get_vec(vec, i)
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=6, color='r')
            ax.quiver(0, 0, 0, -ve[0], -ve[1], -ve[2], length=6, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.pause(0.01)


def animate_wheel_angular_velocity(time, wheel_angular_vel):
    f = plt.figure()

    for i in range(len(wheel_angular_vel)):
        plt.clf()
        a = f.add_subplot(111)
        a.set_xlim(0, time.max())
        a.set_ylim(wheel_angular_vel.min()*9.5493, wheel_angular_vel.max()*9.5493)
        a.plot(time[:i+1], wheel_angular_vel[:i+1]*9.5493)
        a.set_title('reaction wheel angular velocities')
        a.set_ylabel('rpm')
        plt.pause(0.01)


def animate_attitude_and_plot(time, plot_thing, dcm, dcm_reference=None, vec=None, title='NA', ylabel='NA'):
    dcm1 = dcm.copy()

    V = np.array([[-x_mag, -y_mag, -z_mag],
                  [x_mag, -y_mag, -z_mag],
                  [x_mag, y_mag, -z_mag],
                  [-x_mag, y_mag, -z_mag],
                  [-x_mag, -y_mag, z_mag],
                  [x_mag, -y_mag, z_mag],
                  [x_mag, y_mag, z_mag],
                  [-x_mag, y_mag, z_mag]])

    if vec is not None and len(vec) > 3:
        changing = True
    else:
        changing = False

    def get_vec(b, i):
        if not changing:
            return b
        return b[i]


    fig = plt.figure(figsize=(15, 5))
    for i in range(len(dcm1)):
        plt.clf()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        dcm1[i] = dcm1[i].T  # must get [NB] rather than [BN], because the vertices are in the B frame.

        Z = (dcm1[i] @ V.T).T

        # plot vertices
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

        # list of sides' polygons of figure
        verts = [[Z[0],Z[1],Z[2],Z[3]],
         [Z[4],Z[5],Z[6],Z[7]],
         [Z[0],Z[1],Z[5],Z[4]],
         [Z[2],Z[3],Z[7],Z[6]],
         [Z[1],Z[2],Z[6],Z[5]],
         [Z[4],Z[7],Z[3],Z[0]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.quiver(0, 0, 0, dcm1[i][0, 0], dcm1[i][1, 0], dcm1[i][2, 0], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 1], dcm1[i][1, 1], dcm1[i][2, 1], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 2], dcm1[i][1, 2], dcm1[i][2, 2], length=4)
        if dcm_reference is not None:
            ax.quiver(0, 0, 0, dcm_reference[0, 0], dcm_reference[0, 1], dcm_reference[0, 2], length=4, color='r')
            ax.quiver(0, 0, 0, dcm_reference[1, 0], dcm_reference[1, 1], dcm_reference[1, 2], length=4, color='r')
            ax.quiver(0, 0, 0, dcm_reference[2, 0], dcm_reference[2, 1], dcm_reference[2, 2], length=4, color='r')

        if vec is not None:
            ve = get_vec(vec, i)
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=6, color='r')
            ax.quiver(0, 0, 0, -ve[0], -ve[1], -ve[2], length=6, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax = fig.add_subplot(1, 2, 2)
        ax.set_xlim(0, time.max())
        ax.set_ylim(plot_thing.min(), plot_thing.max())
        ax.plot(time[:i+1], plot_thing[:i+1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('time')

        plt.pause(0.01)


def animate_attitude_and_2_plots(time, plot_thing1, plot_thing2, dcm, dcm_reference=None, vec=None, title1='NA',
                                 ylabel1='NA', title2='NA', ylabel2='NA'):
    dcm1 = dcm.copy()

    V = np.array([[-x_mag, -y_mag, -z_mag],
                  [x_mag, -y_mag, -z_mag],
                  [x_mag, y_mag, -z_mag],
                  [-x_mag, y_mag, -z_mag],
                  [-x_mag, -y_mag, z_mag],
                  [x_mag, -y_mag, z_mag],
                  [x_mag, y_mag, z_mag],
                  [-x_mag, y_mag, z_mag]])

    if vec is not None and len(vec) > 3:
        changing = True
    else:
        changing = False

    def get_vec(b, i):
        if not changing:
            return b
        return b[i]


    fig = plt.figure(figsize=(15, 5))
    for i in range(len(dcm1)):
        plt.clf()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        dcm1[i] = dcm1[i].T  # must get [NB] rather than [BN], because the vertices are in the B frame.

        Z = (dcm1[i] @ V.T).T

        # plot vertices
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

        # list of sides' polygons of figure
        verts = [[Z[0],Z[1],Z[2],Z[3]],
         [Z[4],Z[5],Z[6],Z[7]],
         [Z[0],Z[1],Z[5],Z[4]],
         [Z[2],Z[3],Z[7],Z[6]],
         [Z[1],Z[2],Z[6],Z[5]],
         [Z[4],Z[7],Z[3],Z[0]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.quiver(0, 0, 0, dcm1[i][0, 0], dcm1[i][1, 0], dcm1[i][2, 0], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 1], dcm1[i][1, 1], dcm1[i][2, 1], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 2], dcm1[i][1, 2], dcm1[i][2, 2], length=4)
        if dcm_reference is not None:
            ax.quiver(0, 0, 0, dcm_reference[0, 0], dcm_reference[0, 1], dcm_reference[0, 2], length=4, color='r')
            ax.quiver(0, 0, 0, dcm_reference[1, 0], dcm_reference[1, 1], dcm_reference[1, 2], length=4, color='r')
            ax.quiver(0, 0, 0, dcm_reference[2, 0], dcm_reference[2, 1], dcm_reference[2, 2], length=4, color='r')

        if vec is not None:
            ve = get_vec(vec, i)
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=6, color='r')
            ax.quiver(0, 0, 0, -ve[0], -ve[1], -ve[2], length=6, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax = fig.add_subplot(2, 2, 2)
        ax.set_xlim(0, time.max())
        ax.set_ylim(plot_thing1.min(), plot_thing1.max())
        ax.plot(time[:i+1], plot_thing1[:i+1])
        ax.set_title(title1)
        ax.set_ylabel(ylabel1)
        ax.set_xlabel('time')

        ax = fig.add_subplot(2, 2, 4)
        ax.set_xlim(0, time.max())
        ax.set_ylim(plot_thing2.min(), plot_thing2.max())
        ax.plot(time[:i+1], plot_thing2[:i+1])
        ax.set_title(title2)
        ax.set_ylabel(ylabel2)
        ax.set_xlabel('time')

        plt.pause(0.01)


def animate_attitude_and_ground_track(dcm, lats, longs, dcm_reference=None, vec=None):
    dcm1 = dcm.copy()
    V = np.array([[-x_mag, -y_mag, -z_mag],
                  [x_mag, -y_mag, -z_mag],
                  [x_mag, y_mag, -z_mag],
                  [-x_mag, y_mag, -z_mag],
                  [-x_mag, -y_mag, z_mag],
                  [x_mag, -y_mag, z_mag],
                  [x_mag, y_mag, z_mag],
                  [-x_mag, y_mag, z_mag]])

    if vec is not None and len(vec) > 3:
        changing = True
    else:
        changing = False

    def get_vec(b, i):
        if not changing:
            return b
        return b[i]

    if dcm_reference is not None and len(dcm_reference) > 3:
        changing_ref = True
    else:
        changing_ref = False

    def get_dcm_ref(b, i):
        if not changing_ref:
            return b
        return b[i]

    fig = plt.figure(figsize=(18, 5))

    for i in range(len(dcm1)):
        plt.clf()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        dcm1[i] = dcm1[i].T  # must get [NB] rather than [BN], because the vertices are in the B frame.

        Z = (dcm1[i] @ V.T).T

        # plot vertices
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

        # list of sides' polygons of figure
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                 [Z[4], Z[5], Z[6], Z[7]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[2], Z[6], Z[5]],
                 [Z[4], Z[7], Z[3], Z[0]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts,
                                             facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.quiver(0, 0, 0, dcm1[i][0, 0], dcm1[i][1, 0], dcm1[i][2, 0], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 1], dcm1[i][1, 1], dcm1[i][2, 1], length=4)
        ax.quiver(0, 0, 0, dcm1[i][0, 2], dcm1[i][1, 2], dcm1[i][2, 2], length=4)

        if dcm_reference is not None:
            de = get_dcm_ref(dcm_reference, i)
            ax.quiver(0, 0, 0, de[0, 0], de[0, 1], de[0, 2], length=4, color='r')
            ax.quiver(0, 0, 0, de[1, 0], de[1, 1], de[1, 2], length=4, color='r')
            ax.quiver(0, 0, 0, de[2, 0], de[2, 1], de[2, 2], length=4, color='r')

        if vec is not None:
            ve = get_vec(vec, i)
            ve = 2 * ve / np.linalg.norm(ve)
            ax.quiver(0, 0, 0, ve[0], ve[1], ve[2], length=6, color='r')
            # ax.quiver(0, 0, 0, -ve[0], -ve[1], -ve[2], length=6, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-85, 85)
        ax.plot(longs[:i+1], lats[:i+1], '.')
        ax.set_ylabel('lats')
        ax.set_xlabel('lons')

        plt.pause(0.01)
