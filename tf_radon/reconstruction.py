import tensorflow as tf

from . import backprojection
from .filtering import FourierFilters, filter_projections


def fdk_reconstruction(projections: tf.Tensor, radon_args,
                       filter_name: str = 'ramp',
                       fourier_filters: FourierFilters = FourierFilters()) \
                       -> tf.Tensor:
    """ Reconstructs the projections using FDK algorithm.

    Parameters
    ----------
    projections: tf.Tensor
        Cone beam projections. Must be of shape [batch, channels, n_angles,
        det_rows, det_cols].
    radon_args: Tuple
        Additional arguments that specify the CT system setup and the volume.
        See `tf_radon.forward` and `tf_radon.backprojection`.
    filter_name: str
        Filter used in frequency domain filtering. Filters available:
        ramp (ram-lak), shepp-logan, cosine, hamming, hann.
    fourier_filters: FourierFilters
        A FourierFilters objects as cache for filter coefficients.

    Returns
    -------
    volume: tf.Tensor
        The reconstructed volume with shape [batch, channels, z, y, x].

    """
    # filter the projections
    filtered_projections = filter_projections(
        projections,
        filter_name,
        fourier_filters,
    )

    # backproject the filtered projections
    reco = backprojection(filtered_projections, *radon_args)

    # additional correction needed because of cone beams
    det_spacing_v = radon_args[3][1]
    src_dist = radon_args[4]
    det_dist = radon_args[5]
    src_det_dist = src_dist + det_dist
    reco *= det_spacing_v/src_det_dist*src_dist

    return reco
