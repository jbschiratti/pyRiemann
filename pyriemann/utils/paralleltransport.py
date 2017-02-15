import numpy
from .base import expm, logm, invsqrtm, sqrtm


def paralleltransport(A_start, A_end, W):
    """ Computes the parallel transport of W along the geodesic segment from
    A_start to A_end.

    :param A_start: covariance matrix
        A_start is the base point of the geodesic between A_start and A_end
    :param A_end: covariance matrix
        A_end is the end point of the geodesic between A_start and A_end
    :param W: covariance matrix
        W is the tangent vector, in the tangent space at A_start, to be
        transported to the tangent space at A_end.
    :returns covariance matrix

    """
    isA_start = invsqrtm(A_start)
    sA_start = sqrtm(A_start)
    aux = numpy.dot(numpy.dot(isA_start, A_end), isA_start)
    aux = logm(aux)
    R = expm(0.5 * aux)
    C = numpy.dot(sA_start, R)
    T = numpy.dot(numpy.dot(C, W), C.transpose())
    return T