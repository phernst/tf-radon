import numpy as np
import tensorflow as tf


class FourierFilters:
    def __init__(self):
        self.cache = dict()

    def get(self, size: int, filter_name: str):
        key = (size, filter_name)

        if key not in self.cache:
            ff = tf.constant(self.construct_fourier_filter(size, filter_name))
            self.cache[key] = ff

        return self.cache[key]

    @staticmethod
    def construct_fourier_filter(size, filter_name):
        """Construct the Fourier filter.

        This computation lessens artifacts and removes a small bias as
        explained in [1], Chap 3. Equation 61.

        Parameters
        ----------
        size: int
            filter size. Must be even.
        filter_name: str
            Filter used in frequency domain filtering. Filters available:
            ram-lak (ramp), shepp-logan, cosine, hamming, hann.

        Returns
        -------
        fourier_filter: ndarray
            The computed Fourier filter.

        References
        ----------
        .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
               Imaging", IEEE Press 1988.

        """
        filter_name = filter_name.lower()

        n = np.concatenate([np.arange(1, size // 2 + 1, 2, dtype=np.int32),
                            np.arange(size // 2 - 1, 0, -2, dtype=np.int32)])
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2

        # Computing the ramp filter from the fourier transform of its
        # frequency domain representation lessens artifacts and removes a
        # small bias as explained in [1], Chap 3. Equation 61
        fourier_filter = 2 * np.real(np.fft.fft(f))  # ramp filter
        if filter_name in ["ramp", "ram-lak"]:
            pass
        elif filter_name == "shepp-logan":
            # Start from first element to avoid divide by zero
            omega = np.pi * np.fft.fftfreq(size)[1:]
            fourier_filter[1:] *= np.sin(omega) / omega
        elif filter_name == "cosine":
            freq = np.linspace(0, np.pi, size, endpoint=False)
            cosine_filter = np.fft.fftshift(np.sin(freq))
            fourier_filter *= cosine_filter
        elif filter_name == "hamming":
            fourier_filter *= np.fft.fftshift(np.hamming(size))
        elif filter_name == "hann":
            fourier_filter *= np.fft.fftshift(np.hanning(size))
        else:
            print(
                f"[tf-radon] Error, unknown filter type '{filter_name}'"
                ", available filters are: 'ramp', 'shepp-logan', 'cosine', "
                "'hamming', 'hann'")

        return tf.constant(fourier_filter[None, None, :size//2+1])


def filter_projections(projections: tf.Tensor, filter_name: str = "ramp",
                       fourier_filters: FourierFilters = FourierFilters()) \
                       -> tf.Tensor:
    """ Applies (apodized) ramp filtering to cone beam projections.

    This is needed for, e.g., FDK reconstructions.

    Parameters
    ----------
    projections: tf.Tensor
        Cone beam projections. Must be of shape [batch, channels, n_angles,
        det_rows, det_cols].
    filter_name: str
        Filter used in frequency domain filtering. Filters available:
        ramp (ram-lak), shepp-logan, cosine, hamming, hann.
    fourier_filters: FourierFilters
        A FourierFilters objects as cache for filter coefficients.

    Returns
    -------
    filtered_projections: tf.Tensor
        The filtered cone beam projections.

    """
    projections = tf.transpose(projections, perm=(0, 1, 3, 2, 4))
    proj_shape = projections.shape
    projections = tf.reshape(
        projections,
        (
            tf.math.reduce_prod(proj_shape[:-2]),
            proj_shape[-2],
            proj_shape[-1],
        ),
    )
    size = projections.shape[-1]
    n_angles = projections.shape[-2]

    # Pad sinogram to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    tf_pad = tf.constant([[0, 0], [0, 0], [0, pad]])
    padded_sinogram = tf.pad(projections, tf_pad, constant_values=0)

    sino_fft = tf.signal.rfft(padded_sinogram)

    # get filter and apply
    f = tf.cast(fourier_filters.get(padded_size, filter_name), tf.complex64)
    filtered_sino_fft = sino_fft * f

    # Inverse fft
    filtered_sinogram = tf.signal.irfft(filtered_sino_fft)
    filtered_sinogram = filtered_sinogram[:, :, :-pad] * np.pi / (2 * n_angles)

    return tf.transpose(
        tf.reshape(
            tf.cast(filtered_sinogram, projections.dtype),
            proj_shape),
        perm=(0, 1, 3, 2, 4))
