import util as ut


def gravity_gradient(ue, R0, inertia):
    u = 3.986004418 * (10**14)
    return (3*u/(R0**3)) * (ut.cross_product_operator(ue) @ inertia @ ue)
