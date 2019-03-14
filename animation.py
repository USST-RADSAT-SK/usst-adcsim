import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def animate_attitude(dcm, dcm_reference):

    V = np.array([[-2, -1, -1],
                  [2, -1, -1],
                  [2, 1, -1],
                  [-2, 1, -1],
                  [-2, -1, 1],
                  [2, -1, 1],
                  [2, 1, 1],
                  [-2, 1, 1]])

    fig = plt.figure()

    for i in range(len(dcm)):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        Z = (dcm[i] @ V.T).T

        r = [-1, 1]
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
        ax.add_collection3d(Poly3DCollection(verts,
         facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.quiver(0, 0, 0, dcm[i][0, 0], dcm[i][1, 0], dcm[i][2, 0], length=4)
        ax.quiver(0, 0, 0, dcm[i][0, 1], dcm[i][1, 1], dcm[i][2, 1], length=4)
        ax.quiver(0, 0, 0, dcm[i][0, 2], dcm[i][1, 2], dcm[i][2, 2], length=4)

        ax.quiver(0, 0, 0, dcm_reference[0, 0], dcm_reference[1, 0], dcm_reference[2, 0], length=4, color='r')
        ax.quiver(0, 0, 0, dcm_reference[0, 1], dcm_reference[1, 1], dcm_reference[2, 1], length=4, color='r')
        ax.quiver(0, 0, 0, dcm_reference[0, 2], dcm_reference[1, 2], dcm_reference[2, 2], length=4, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.pause(0.01)


def animate_wheel_angular_velocity(time, wheel_angular_vel):
    f = plt.figure()

    for i in range(len(wheel_angular_vel)):
        plt.clf()
        a = f.add_subplot(111)
        a.set_xlim(0, 300)
        a.set_ylim(-20000, 16000)
        a.plot(time[:i+1], wheel_angular_vel[:i+1]*9.5493)
        a.set_title('reaction wheel angular velocities')
        a.set_ylabel('rpm')
        plt.pause(0.01)



def combination(time, wheel_angular_vel, dcm, dcm_reference):

    V = np.array([[-2, -1, -1],
                  [2, -1, -1],
                  [2, 1, -1],
                  [-2, 1, -1],
                  [-2, -1, 1],
                  [2, -1, 1],
                  [2, 1, 1],
                  [-2, 1, 1]])

    fig = plt.figure(figsize=(15, 5))
    for i in range(len(dcm)):
        plt.clf()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)

        Z = (dcm[i] @ V.T).T

        r = [-1, 1]
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
        ax.add_collection3d(Poly3DCollection(verts,
         facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.quiver(0, 0, 0, dcm[i][0, 0], dcm[i][1, 0], dcm[i][2, 0], length=4)
        ax.quiver(0, 0, 0, dcm[i][0, 1], dcm[i][1, 1], dcm[i][2, 1], length=4)
        ax.quiver(0, 0, 0, dcm[i][0, 2], dcm[i][1, 2], dcm[i][2, 2], length=4)

        ax.quiver(0, 0, 0, dcm_reference[0, 0], dcm_reference[1, 0], dcm_reference[2, 0], length=4, color='r')
        ax.quiver(0, 0, 0, dcm_reference[0, 1], dcm_reference[1, 1], dcm_reference[2, 1], length=4, color='r')
        ax.quiver(0, 0, 0, dcm_reference[0, 2], dcm_reference[1, 2], dcm_reference[2, 2], length=4, color='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax = fig.add_subplot(1, 2, 2)
        ax.set_xlim(0, 300)
        ax.set_ylim(-20000, 16000)
        ax.plot(time[:i+1], wheel_angular_vel[:i+1]*9.5493)
        ax.set_title('reaction wheel angular velocities')
        ax.set_ylabel('rpm')

        plt.pause(0.01)


if __name__ == "__main__":
    dcm_rn = np.load('dcm_rn.npy')
    dcm = np.load('dcm_array.npy')

    animate_attitude(dcm[::10], dcm_rn)

    plt.show()
#
# dcm = np.load('dcm_array.npy')
# dcm = dcm[:5000:10]
#
#
# Z = np.array([[-2, -1, -1],
#               [2, -1, -1],
#               [2, 1, -1],
#               [-2, 1, -1],
#               [-2, -1, 1],
#               [2, -1, 1],
#               [2, 1, 1],
#               [-2, 1, 1]])
#
# i = 0
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# ax.set_zlim(-4, 4)
#
# Z = (dcm[i] @ Z.T).T
#
# r = [-1,1]
#
# # X, Y = np.meshgrid(r, r)
# # plot vertices
# ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
#
# # list of sides' polygons of figure
# verts = [[Z[0],Z[1],Z[2],Z[3]],
#  [Z[4],Z[5],Z[6],Z[7]],
#  [Z[0],Z[1],Z[5],Z[4]],
#  [Z[2],Z[3],Z[7],Z[6]],
#  [Z[1],Z[2],Z[6],Z[5]],
#  [Z[4],Z[7],Z[3],Z[0]]]
#
# # plot sides
# ax.add_collection3d(Poly3DCollection(verts,
#  facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# ax.quiver(0, 0, 0, dcm[i][0, 0], dcm[i][1, 0], dcm[i][2, 0], length=4)
# ax.quiver(0, 0, 0, dcm[i][0, 1], dcm[i][1, 1], dcm[i][2, 1], length=4)
# ax.quiver(0, 0, 0, dcm[i][0, 2], dcm[i][1, 2], dcm[i][2, 2], length=4)
#
