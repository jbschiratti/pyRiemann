import numpy
from .base import sqrtm, invsqrtm, logm, expm
from .paralleltransport import paralleltransport

###############################################################
# Tangent Space
###############################################################


def tangent_space(covmats, Cref, metric='riemann'):
    """Project a set of covariance matrices in the tangent space according to the given reference point Cref

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: The reference covariance matrix
    :param metric : str ; defaults to 'riemann'
    :returns: the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    Nf = int(Ne * (Ne + 1) / 2)
    T = numpy.empty((Nt, Nf))
    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne))[idx]
    for index in range(Nt):
        if metric is 'riemann':
            tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]), Cm12)
            tmp = logm(tmp)
            T[index, :] = numpy.multiply(coeffs, tmp[idx])
        elif metric is 'euclid':
            tmp = logm(covmats[index, :, :]) - logm(Cref)
            T[index, :] = numpy.multiply(coeffs, tmp[idx])
    return T

def tangent_space_parallel_transport(covmats, Cref_covmats, Cref_target,
                                     metric='riemann'):
    """ The tangent vectors in covemats (assumed to be tangent vectors at
    Cref_covemats) are transported to the tangent space at Cref_target.


    :param tangent vectors set
    :param Cref_covmats : covariance matrix
    :param Cref_target : covariance
    :param metric : str
    :returns tangent vectors set

    """
    Nt, Ne, Ne = covmats.shape
    isCref_covmats = invsqrtm(Cref_covmats)
    isCref_target = invsqrtm(Cref_target)
    idx = numpy.triu_indices_from(Cref_covmats)
    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne))[idx]
    Nf = int(Ne * (Ne + 1) / 2)
    T = numpy.empty((Nt, Nf))
    for index in range(Nt):
        if metric is 'riemann':
            tmp = numpy.dot(numpy.dot(isCref_covmats, covmats[index, :, :]),
                            isCref_covmats)
            tmp = logm(tmp)
            tmp = paralleltransport(Cref_covmats, Cref_target, tmp)
            tmp = numpy.dot(numpy.dot(isCref_target, tmp), isCref_target)
            T[index, :] = numpy.multiply(coeffs, tmp[idx])
        elif metric is 'euclid':
            tmp = logm(covmats[index, :, :]) - logm(Cref_covmats)
            T[index, :] = numpy.multiply(coeffs, tmp[idx])
    return T

def untangent_space(T, Cref):
    """Project a set of Tangent space vectors in the manifold according to the given reference point Cref

    :param T: the Tangent space , a matrix of Ntrials X (Nchannels * (Nchannels + 1)/2)
    :param Cref: The reference covariance matrix
    :returns: A set of Covariance matrix, Ntrials X Nchannels X Nchannels

    """
    Nt, Nd = T.shape
    Ne = int((numpy.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = numpy.triu_indices_from(Cref)
    covmats = numpy.empty((Nt, Ne, Ne))
    covmats[:, idx[0], idx[1]] = T
    for i in range(Nt):
        triuc = numpy.triu(covmats[i], 1) / numpy.sqrt(2)
        covmats[i] = (numpy.diag(numpy.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12, covmats[i]), C12)

    return covmats
