import numpy as np
import adcsim.transformations as trans
from adcsim.util import cross_product_operator


def triad_method(v_true, v_measured):
    """Uses the TRIAD method to determine the Direction Cosine Matrix (BN DCM)
    to transform from the true-vector frame to the measured-vector frame.

    NOTE: vectors v_measured[0] and v_true[0] should correspond to the
    measurement with the highest accuracy, while vectors v_measured[1] and
    v_true[1] should correspond to the measurement with the lowest accuracy.

    :param v_true: (numpy array, shape (2, 3))
        Two true vectors (v_true[0] and v_true[1]) in some frame of reference
        (usually the inertial frame).
    :param v_measured: (numpy array, shape (2, 3))
        Two measured vectors (v_measured[0] and v_measured[1]) in some frame of
        reference that is different than that of the true vectors (usually the
        body frame).
    :return: (numpy array, shape (3, 3))
        The DCM to transform from the measured-vector frame to the true-vector
        frame.
    """
    # Normalize vectors
    v_true[0] = v_true[0] / np.linalg.norm(v_true[0])
    v_true[1] = v_true[1] / np.linalg.norm(v_true[1])
    v_measured[0] = v_measured[0] / np.linalg.norm(v_measured[0])
    v_measured[1] = v_measured[1] / np.linalg.norm(v_measured[1])

    # Measured vectors in arbitrary f frame
    cross_prod_b = np.cross(v_measured[0], v_measured[1])
    fm1 = v_measured[0]
    fm2 = cross_prod_b / np.linalg.norm(cross_prod_b)
    dcm_fm = np.array([fm1, fm2, np.cross(fm1, fm2)])

    # True vectors in arbitrary f frame
    cross_prod_n = np.cross(v_true[0], v_true[1])
    ft1 = v_true[0]
    ft2 = cross_prod_n / np.linalg.norm(cross_prod_n)
    dcm_ft = np.array([ft1, ft2, np.cross(ft1, ft2)])

    return dcm_fm.T @ dcm_ft


def q_method(v_true, v_measured, w=None):
    """Uses Devonport's q-Method to determine the Direction Cosine Matrix (DCM)
    to transform from the true-vector frame to the measured-vector frame.

    :param v_true: (numpy array, shape (N, 3))
        The N true vectors (v_true[i] for i = 0, 1, ..., N-1) in some frame of
        reference (usually the inertial frame).
    :param v_measured: (numpy array, shape (N, 3))
        The N measured vectors (v_measured[i] for i = 0, 1, ..., N-1) in some
        frame of reference that is different than that of the true vectors
        (usually the body frame).
    :param w: (numpy array, shape (N,))
        Weights of the N measured vectors. This indicates how much accuracy is
        expected for each measurement/sensor.

        e.g. If using two vectors and the first vector measurement is twice as
        accurate as the second, then the ratio of weights for the first and
        second vectors should be 2. Thus, one can set
        w = np.array([2, 1]), w = np.array([100, 50]), w = np.array([3, 1.5]),
        or anything else with the right ratio.

        Note that weights do not need to be normalized.
        If w is None or not specified, weights are set as equal (all 1's) within
        the functions for each method.
    :return: (numpy array, shape (3, 3))
        The DCM to transform from the measured-vector frame to the true-vector
        frame.
    """
    if w is None:
        w = np.ones(len(v_true))

    b = np.zeros((3, 3))
    for i in range(len(v_true)):
        # Normalize vectors
        v_true[i] = v_true[i] / np.linalg.norm(v_true[i])
        v_measured[i] = v_measured[i] / np.linalg.norm(v_measured[i])

        b += w[i] * np.outer(v_measured[i], v_true[i])
    s = b + b.T
    sigma = np.trace(b)
    z = np.array([b[1, 2] - b[2, 1], b[2, 0] - b[0, 2], b[0, 1] - b[1, 0]])
    k = np.array([[sigma, z[0],            z[1],            z[2]],
                  [z[0],  s[0, 0] - sigma, s[0, 1],         s[0, 2]],
                  [z[1],  s[1, 0],         s[1, 1] - sigma, s[1, 2]],
                  [z[2],  s[2, 0],         s[2, 1],         s[2, 2] - sigma]])

    # Resulting Euler parameters/quaternions are given by the eigenvector
    # corresponding to the largest eigenvalue.
    e_vals, e_vects = np.linalg.eig(k)
    idx = np.argmax(e_vals)
    return trans.quaternions_to_dcm(e_vects[:, idx])


def quest_method(v_true, v_measured, w=None, n_iterations=3):
    """Uses the QUEST method to determine the Direction Cosine Matrix (DCM)
    to transform from the true-vector frame to the measured-vector frame.
    The QUEST method is an approximation of the q-method. It has lower accuracy
    but greater computational speed.

    :param v_true: (numpy array, shape (N, 3))
        The N true vectors (v_true[i] for i = 0, 1, ..., N-1) in some frame of
        reference (usually the inertial frame).
    :param v_measured: (numpy array, shape (N, 3))
        The N measured vectors (v_measured[i] for i = 0, 1, ..., N-1) in some
        frame of reference that is different than that of the true vectors
        (usually the body frame).
    :param w: (numpy array, shape (N,))
        Weights of the N measured vectors. This indicates how much accuracy is
        expected for each measurement/sensor.

        e.g. If using two vectors and the first vector measurement is twice as
        accurate as the second, then the ratio of weights for the first and
        second vectors should be 2. Thus, one can set
        w = np.array([2, 1]), w = np.array([100, 50]), w = np.array([3, 1.5]),
        or anything else with the right ratio.

        Note that weights do not need to be normalized.
        If w is None or not specified, weights are set as equal (all 1's) within
        the functions for each method.
    :param n_iterations: (int)
        The number of Newton-Raphson iterations to get the eigenvalue
        corresponding to the optimal solution of Whaba's problem.
    :return: (numpy array, shape (3, 3))
        The DCM to transform from the measured-vector frame to the true-vector
        frame.
    """
    if w is None:
        w = np.ones(len(v_true))

    b = np.zeros((3, 3))
    for i in range(len(v_true)):
        # Normalize vectors
        v_true[i] = v_true[i] / np.linalg.norm(v_true[i])
        v_measured[i] = v_measured[i] / np.linalg.norm(v_measured[i])

        b += w[i] * np.outer(v_measured[i], v_true[i])
    s = b + b.T
    sigma = np.trace(b)
    z = np.array([b[1, 2] - b[2, 1], b[2, 0] - b[0, 2], b[0, 1] - b[1, 0]])
    k = np.array([[sigma, z[0],            z[1],            z[2]],
                  [z[0],  s[0, 0] - sigma, s[0, 1],         s[0, 2]],
                  [z[1],  s[1, 0],         s[1, 1] - sigma, s[1, 2]],
                  [z[2],  s[2, 0],         s[2, 1],         s[2, 2] - sigma]])

    def f(x):
        return np.linalg.det(k - x * np.eye(4))

    def f_prime(x):
        """Returns the derivative of f(x)."""
        a0 = -(k[0, 0] * (k[1, 1] * (k[2, 2] + k[3, 3]) + k[2, 2] * k[3, 3])
               + k[1, 1] * k[2, 2] * k[3, 3])\
             + k[0, 0] * (k[1, 2] * k[2, 1] + k[1, 3] * k[3, 1] + k[2, 3] * k[3, 2])\
             + k[1, 1] * (k[0, 2] * k[2, 0] + k[0, 3] * k[3, 0] + k[2, 3] * k[3, 2])\
             + k[2, 2] * (k[0, 1] * k[1, 0] + k[0, 3] * k[3, 0] + k[1, 3] * k[3, 1])\
             + k[3, 3] * (k[0, 1] * k[1, 0] + k[0, 2] * k[2, 0] + k[1, 2] * k[2, 1])\
             - (k[0, 1] * (k[2, 0] * k[1, 2] + k[3, 0] * k[1, 3])
                + k[0, 2] * (k[1, 0] * k[2, 1] + k[3, 0] * k[2, 3])
                + k[0, 3] * (k[1, 0] * k[3, 1] + k[2, 0] * k[3, 2])
                + k[1, 2] * k[3, 1] * k[2, 3] + k[1, 3] * k[2, 1] * k[3, 2])
        a1 = 2 * (k[0, 0] * (k[1, 1] + k[2, 2] + k[3, 3])
                  + k[1, 1] * (k[2, 2] + k[3, 3])
                  + k[2, 2] * k[3, 3]
                  - (k[0, 1] * k[1, 0] + k[0, 2] * k[2, 0] + k[0, 3] * k[3, 0]
                     + k[1, 2] * k[2, 1] + k[1, 3] * k[3, 1] + k[2, 3] * k[3, 2]))
        return 4 * x ** 3 - 3 * np.trace(k) * x ** 2 + a1 * x + a0

    # Newton-Raphson method to get maximum lambda
    lambda_i = np.sum(w)    # lambda_0
    for i in range(n_iterations):
        lambda_i = lambda_i - f(lambda_i) / f_prime(lambda_i)

    return trans.crp_to_dcm(np.linalg.inv((lambda_i + sigma) * np.eye(3) - s) @ z)


def olae_method(v_true, v_measured, w=None):
    """Uses the Optimal Linear Attitude Estimator (OLAE) method to determine the
    Direction Cosine Matrix (DCM) to transform from the true-vector frame to the
    measured-vector frame.

    :param v_true: (numpy array, shape (N, 3))
        The N true vectors (v_true[i] for i = 0, 1, ..., N-1) in some frame of
        reference (usually the inertial frame).
    :param v_measured: (numpy array, shape (N, 3))
        The N measured vectors (v_measured[i] for i = 0, 1, ..., N-1) in some
        frame of reference that is different than that of the true vectors
        (usually the body frame).
    :param w: (numpy array, shape (N,))
        Weights of the N measured vectors. This indicates how much accuracy is
        expected for each measurement/sensor.

        e.g. If using two vectors and the first vector measurement is twice as
        accurate as the second, then the ratio of weights for the first and
        second vectors should be 2. Thus, one can set
        w = np.array([2, 1]), w = np.array([100, 50]), w = np.array([3, 1.5]),
        or anything else with the right ratio.

        Note that weights do not need to be normalized.
        If w is None or not specified, weights are set as equal (all 1's) within
        the functions for each method.
    :return: (numpy array, shape (3, 3))
        The DCM to transform from the measured-vector frame to the true-vector
        frame.
    """
    if w is None:
        w = np.ones(len(v_true))

    w_mat = np.diag(np.repeat(w, 3))
    s = np.nan * np.ones((3 * len(v_true), 3))
    for i in range(len(v_true)):
        s[i * 3:(i + 1) * 3, :] = cross_product_operator(v_measured[i]) + cross_product_operator(v_true[i])
    d = v_measured.flatten() - v_true.flatten()

    return trans.crp_to_dcm(np.linalg.inv(s.T @ w_mat @ s) @ s.T @ w_mat @ d)


def test_methods(v_true=None, v_measured=None, w=None, euler_angles_true=None):
    """Tests the different determination methods. This can easily be checked
    against the University of Colorado Boulder course notes.

    :param v_true: (numpy array, shape (N, 3))
        The N true vectors (v_true[i] for i = 0, 1, ..., N-1) in some frame of
        reference (usually the inertial frame).
    :param v_measured: (numpy array, shape (N, 3))
        The N measured vectors (v_measured[i] for i = 0, 1, ..., N-1) in some
        frame of reference that is different than that of the true vectors
        (usually the body frame).
    :param w: (numpy array, shape (N,))
        Weights of the N measured vectors. This indicates how much accuracy is
        expected for each measurement/sensor.

        e.g. If using two vectors and the first vector measurement is twice as
        accurate as the second, then the ratio of weights for the first and
        second vectors should be 2. Thus, one can set
        w = np.array([2, 1]), w = np.array([100, 50]), w = np.array([3, 1.5]),
        or anything else with the right ratio.

        Note that weights do not need to be normalized.
        If w is None or not specified, weights are set as equal (all 1's) within
        the functions for each method.
    :param euler_angles_true: (numpy array, shape (3,))
       The Euler angles describing the true attitude state (angle between the
       true-vector frame and the measured-vector frame). This allows for
       the comparison of each method to the true values.
       If None, the values from the University of Boulder Colorado course are
       used.
    :return: N/A (prints results to console).
    """
    # Default values are those from the University of Boulder Colorado course notes
    if v_true is None:
        v_true = np.array([[1, 0, 0], [0, 0, 1]])
    if v_measured is None:
        v_measured = np.array([[0.8190, -0.5282, 0.2242],
                               [-0.3138, -0.1584, 0.9362]])
    if euler_angles_true is None:
        euler_angles_true = np.array([30, 20, -10]) * np.pi / 180

    # True attitude state (to test method)
    dcm_mt_true = trans.euler_angles_to_dcm(euler_angles_true)

    # TRIAD
    if len(v_true) == 2:
        dcm_mt_triad = triad_method(v_true, v_measured)
        angle_triad, e_triad = trans.dcm_to_prv(dcm_mt_triad @ dcm_mt_true.T)
        print('TRIAD method error:\n   ' + str(angle_triad * 180 / np.pi) + ' degrees')

    # q-method
    dcm_mt_q = q_method(v_true, v_measured, w)
    angle_q, e_q = trans.dcm_to_prv(dcm_mt_q @ dcm_mt_true.T)
    print('q-method error:\n   ' + str(angle_q * 180 / np.pi) + ' degrees')

    # QUEST
    print('QUEST method error:')
    n_quest_iters = 4
    dcm_mt_quest = [quest_method(v_true, v_measured, w, n_iterations=i) for i in range(n_quest_iters)]
    for i in range(n_quest_iters):
        angle_quest, e_quest = trans.dcm_to_prv(dcm_mt_quest[i] @ dcm_mt_true.T)
        print('   ' + str(i) + ' iterations:')
        print('      ' + str(angle_quest * 180 / np.pi) + ' degrees')
        print('      ' + str((angle_quest - angle_q) * 180 / np.pi) +
              ' degrees (relative to q-method)')

    # OLAE
    dcm_mt_olae = olae_method(v_true, v_measured, w)
    angle_olae, e_olae = trans.dcm_to_prv(dcm_mt_olae @ dcm_mt_true.T)
    print('OLAE method error:\n   ' + str(angle_olae * 180 / np.pi) + ' degrees')


if __name__ == '__main__':
    # True vectors in inertial frame and two measured vectors in body frame
    v_true = np.array([[1, 0, 0], [0, 0, 1]])
    v_measured = np.array([[0.8190, -0.5282, 0.2242],
                           [-0.3138, -0.1584, 0.9362]])
    w = np.array([1, 1])

    # Test all methods
    test_methods(v_true, v_measured, w=w)
