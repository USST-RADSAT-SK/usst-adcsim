import transformations as tr


def get_mrp_br(dcm_rn, sigma):
    dcm_br = tr.mrp_to_dcm(sigma) @ dcm_rn.T
    return tr.dcm_to_mrp(dcm_br)
