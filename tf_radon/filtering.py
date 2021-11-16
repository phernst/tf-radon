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
        tf_pi = tf.constant(np.pi)

        n = tf.concat([tf.range(1, size / 2 + 1, 2, dtype=tf.int32),
                       tf.range(size / 2 - 1, 0, -2, dtype=tf.int32)], 0)
        f = tf.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (tf_pi * n) ** 2

        # Computing the ramp filter from the fourier transform of its
        # frequency domain representation lessens artifacts and removes a
        # small bias as explained in [1], Chap 3. Equation 61
        fourier_filter = 2 * tf.math.real(tf.signal.fft(f))  # ramp filter
        if filter_name in ["ramp", "ram-lak"]:
            pass
        elif filter_name == "shepp-logan":
            # Start from first element to avoid divide by zero
            omega = tf_pi * tf.constant(np.fft.fftfreq(size)[1:])
            fourier_filter[1:] *= tf.math.sin(omega) / omega
        elif filter_name == "cosine":
            freq = tf.constant(np.linspace(0, np.pi, size, endpoint=False))
            cosine_filter = tf.signal.fftshift(tf.math.sin(freq))
            fourier_filter *= cosine_filter
        elif filter_name == "hamming":
            fourier_filter *= tf.signal.fftshift(
                tf.signal.hamming_window(size, False))
        elif filter_name == "hann":
            fourier_filter *= tf.signal.fftshift(
                tf.signal.hann_window(size, False))
        else:
            print(
                f"[tf-radon] Error, unknown filter type '{filter_name}'"
                ", available filters are: 'ramp', 'shepp-logan', 'cosine', "
                "'hamming', 'hann'")

        return fourier_filter[None, None, :size//2+1]
