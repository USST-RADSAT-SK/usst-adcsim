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
        b = tr.dcm_to_quaternions(dcm1)
        dcm2 = tr.quaternions_to_dcm(b)
        np.testing.assert_almost_equal(dcm1, dcm2)

    @staticmethod
    def test_quaternions_valid_dcm():
        dcm1 = ut.random_dcm()
        b = tr.dcm_to_quaternions(dcm1)
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
    def test_sheppards_method_same_result():
        for i in range(30):  # this is a quick and poor way to check that sheppard's method works for all 4 cases
            dcm = ut.random_dcm()
            b1 = tr.dcm_to_quaternions(dcm)
            b2 = tr.sheppards_method(dcm)
            np.testing.assert_almost_equal(b1, b2)

    @staticmethod
    def test_sheppards_method_on_singularity_case():
        e = 2 * np.random.random(3) - 1
        e = e / np.linalg.norm(e)  # random unit vector
        b1 = np.insert(e, 0, 0)
        dcm = tr.quaternions_to_dcm(b1)
        b2 = tr.sheppards_method(dcm)
        try:  # The direction on the unit vector doesnt matter, because the angle is 180 in this case of b[0] = 0
            np.testing.assert_almost_equal(b1, b2)
        except AssertionError:
            np.testing.assert_almost_equal(-b1, b2)


if __name__ == '__main__':
    unittest.main()
