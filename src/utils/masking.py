import jax
import jax.numpy as jnp
import numpy as np

from jax import random, vmap

from flax import linen as nn


def make_random_binary_mask_1D(key, shape, percent_zeros):
    B, N = shape
    num_zeros = np.rint(N * percent_zeros).astype(np.int32)

    # Generate random indices for placing the 1s
    indices = jnp.broadcast_to(jnp.arange(N), shape)
    random_indices = jax.random.permutation(key, indices, axis=-1, independent=True)[:, :num_zeros]
    random_indices = (random_indices + (jnp.arange(B) * N)[:, None]).flatten()

    # Create the binary mask with 1s at the selected indices
    binary_mask = jnp.ones(B * N, dtype=np.int8)
    binary_mask = binary_mask.at[random_indices].set(0)

    # Reshape to the desired shape
    binary_mask = binary_mask.reshape(shape)

    return binary_mask


def sample_random_binary_mask_1D(key, probs, percent_zeros):
    probs = jnp.squeeze(probs, axis=0)
    B, N = probs.shape
    percent_ones = 1 - percent_zeros
    num_patches_keep = int(N * percent_ones)

    def _sample(key, probs, N, M):
        binary_mask = nn.one_hot(
            random.choice(key, a=N, shape=(M,), p=probs, replace=False),
            num_classes=N
        ).sum(0)
        return binary_mask

    sample_rngs = random.split(key, B)
    return vmap(_sample, in_axes=(0, 0, None, None))(sample_rngs, probs, N, num_patches_keep)


def broadcast_to_2D_mask(mask_1D, obs_shape, P):
    """
    Broadcast a 1D bit mask to a 2D mask with shape `obs_shape`.

    mask_1D: jnp.Array  bit mask
    obs_shape: tuple    shape of full observation
    P: int              patch size
    G: int              grid_size
    """
    B = obs_shape[0]  # batch size
    H = W = int(mask_1D.shape[-1] ** 0.5)
    obs_mask = mask_1D.reshape(B, H, 1, W, 1, 1)
    obs_mask = jnp.broadcast_to(
        obs_mask,
        (B, H, P, W, P, obs_shape[-1])
    )
    return obs_mask.reshape(obs_shape)  # (B, H, W, C)
