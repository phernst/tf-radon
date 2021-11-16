import sys

import nrrd
import numpy as np
import tensorflow as tf

import tf_radon
from tf_radon.reconstruction import fdk_reconstruction


def load_nrrd(path: str):
    volume, nrrd_header = nrrd.read(path)
    voxel_size = np.diag(nrrd_header['space directions'])
    return volume, tuple(abs(s) for s in voxel_size)


def hu2mu(volume: tf.Tensor, mu_water: float = 0.02) -> tf.Tensor:
    return (volume * mu_water)/1000 + mu_water


def mu2hu(volume: tf.Tensor, mu_water: float = 0.02) -> tf.Tensor:
    return (volume - mu_water)/mu_water * 1000


def main(nrrd_path: str):
    volume, voxel_size = load_nrrd(nrrd_path)
    volume = tf.constant(volume.transpose()[None, None])
    volume = hu2mu(volume)

    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    radon_args = (
        angles,
        volume[0, 0].shape,
        (620, 480),
        (.616, .616),
        1200.*3/4,
        1200.*1/4,
        (0., 0., 0.),
        voxel_size[::-1],
    )

    projections = tf_radon.forward(
        volume,
        *radon_args,
    )

    reco = fdk_reconstruction(projections, radon_args, 'hann')

    reco = mu2hu(reco)

    from matplotlib import pyplot as plt
    plt.imshow(reco[0, 0, reco.shape[2]//2].numpy(), vmin=-1000)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise RuntimeError("No nrrd file specified")
    main(sys.argv[1])
