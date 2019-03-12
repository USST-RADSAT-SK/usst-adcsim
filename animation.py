import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import transformations as tr


Z = np.array([[-1, -1, -1],
                  [1, -1, -1 ],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1 ],
                  [1, 1, 1],
                  [-1, 1, 1]])



fig = plt.figure()

for i in range(100):
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    v = np.array([0.5+0.00001*i, 0.5, 0.1])  # create a vector that represents euler angle rotation
    dcm_rn = tr.euler_angles_to_dcm(v, type='3-2-1')  # find dcm corresponding to euler angle rotation

    Z = (dcm_rn @ Z.T).T

    r = [-1,1]

    X, Y = np.meshgrid(r, r)
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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.pause(0.1)

plt.show()
