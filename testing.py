import unittest
import transformations as tr
import util as ut
import numpy as np


class UtilitiesTests(unittest.TestCase):
    @staticmethod
    def test_random_dcm_1():
        dcm = ut.random_dcm()
        np.testing.assert_almost_equal(dcm @ dcm.T, np.identity(3))

    @staticmethod
    def test_random_dcm_2():
        dcm = ut.random_dcm()
        np.testing.assert_almost_equal(np.linalg.det(dcm), 1)

    @staticmethod
    def test_random_dcm_3():
        dcm = ut.random_dcm()
        for i in (0, 1, 2):
            np.testing.assert_almost_equal(np.linalg.norm(dcm[i]), 1)
            np.testing.assert_almost_equal(np.linalg.norm(dcm[:, i]), 1)

    @staticmethod
    def test_align_z_to_nadir():
        p = 100 * np.random.random(3)
        dcm = ut.align_z_to_nadir(pos_vec=p)
        p_norm = p/np.linalg.norm(p)
        np.testing.assert_almost_equal(dcm @ p_norm, np.array([0, 0, 1]))

    @staticmethod
    def test_initial_align_gravity_stabilization():
        p = np.random.random(3)
        p = p/np.linalg.norm(p)

        v = np.random.random(3)
        v = v / np.linalg.norm(v)

        dcm = ut.initial_align_gravity_stabilization(p, v)

        cross_track = ut.cross_product_operator(p) @ v
        cross_track = cross_track/np.linalg.norm(cross_track)

        np.testing.assert_almost_equal(dcm @ p, np.array([0, 0, 1]))
        np.testing.assert_almost_equal(dcm @ cross_track, np.array([0, 1, 0]))


class TransformationsTests(unittest.TestCase):
    @staticmethod
    def test_pvr_reverse():
        dcm1 = ut.random_dcm()
        angle, e = tr.dcm_to_prv(dcm1)
        dcm2 = tr.prv_to_dcm(angle, e)
        np.testing.assert_almost_equal(dcm1, dcm2)

    @staticmethod
    def test_pvr_valid_dcm():
        dcm1 = ut.random_dcm()
        angle, e = tr.dcm_to_prv(dcm1)
        dcm2 = tr.prv_to_dcm(angle, e)
        np.testing.assert_almost_equal(dcm2 @ dcm2.T, np.identity(3))
        np.testing.assert_almost_equal(np.linalg.det(dcm2), 1)
        for i in (0, 1, 2):
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[i]), 1)
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[:, i]), 1)

    @staticmethod
    def test_quaternions_reverse():
        dcm1 = ut.random_dcm()
        b = tr.dcm_to_quaternions_bad(dcm1)
        dcm2 = tr.quaternions_to_dcm(b)
        np.testing.assert_almost_equal(dcm1, dcm2)

    @staticmethod
    def test_quaternions_valid_dcm():
        dcm1 = ut.random_dcm()
        b = tr.dcm_to_quaternions_bad(dcm1)
        dcm2 = tr.quaternions_to_dcm(b)
        np.testing.assert_almost_equal(dcm2 @ dcm2.T, np.identity(3))
        np.testing.assert_almost_equal(np.linalg.det(dcm2), 1)
        for i in (0, 1, 2):
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[i]), 1)
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[:, i]), 1)

    @staticmethod
    def test_crp_reverse():
        dcm1 = ut.random_dcm()
        b = tr.dcm_to_crp(dcm1)
        dcm2 = tr.crp_to_dcm(b)
        np.testing.assert_almost_equal(dcm1, dcm2)

    @staticmethod
    def test_crp_valid_dcm():
        dcm1 = ut.random_dcm()
        b = tr.dcm_to_crp(dcm1)
        dcm2 = tr.crp_to_dcm(b)
        np.testing.assert_almost_equal(dcm2 @ dcm2.T, np.identity(3))
        np.testing.assert_almost_equal(np.linalg.det(dcm2), 1)
        for i in (0, 1, 2):
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[i]), 1)
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[:, i]), 1)

    @staticmethod
    def test_mrp_reverse():
        dcm1 = ut.random_dcm()
        b = tr.dcm_to_mrp(dcm1)
        dcm2 = tr.mrp_to_dcm(b)
        np.testing.assert_almost_equal(dcm1, dcm2)

    @staticmethod
    def test_mrp_valid_dcm():
        dcm1 = ut.random_dcm()
        b = tr.dcm_to_mrp(dcm1)
        dcm2 = tr.mrp_to_dcm(b)
        np.testing.assert_almost_equal(dcm2 @ dcm2.T, np.identity(3))
        np.testing.assert_almost_equal(np.linalg.det(dcm2), 1)
        for i in (0, 1, 2):
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[i]), 1)
            np.testing.assert_almost_equal(np.linalg.norm(dcm2[:, i]), 1)

    @staticmethod
    def test_dcm_to_quaternions_same_result():
        for i in range(30):  # this is a quick and poor way to check that sheppard's method works for all 4 cases
            dcm = ut.random_dcm()
            b1 = tr.dcm_to_quaternions_bad(dcm)
            b2 = tr.dcm_to_quaternions(dcm)
            np.testing.assert_almost_equal(b1, b2)

    @staticmethod
    def test_dcm_to_quaternions_on_singularity_case():
        e = 2 * np.random.random(3) - 1
        e = e / np.linalg.norm(e)  # random unit vector
        b1 = np.insert(e, 0, 0)
        dcm = tr.quaternions_to_dcm(b1)
        b2 = tr.dcm_to_quaternions(dcm)
        try:  # The direction on the unit vector doesnt matter, because the angle is 180 in this case of b[0] = 0
            np.testing.assert_almost_equal(b1, b2)
        except AssertionError:
            np.testing.assert_almost_equal(-b1, b2)


if __name__ == '__main__':
    unittest.main()
