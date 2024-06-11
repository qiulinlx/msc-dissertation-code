import functools

from typing import Sequence

import jax
import numpy as np

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal, glorot_uniform

from jax import numpy as jnp
from jax.lax import reshape


class DenseEmbed(nn.Module):
    """
    Dense vector embedding.
    """
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        _, B, input_dim = x.shape  # (seq, B, input_dim)
        I = jnp.eye(input_dim)
        x = I * jnp.broadcast_to(x[..., None], x.shape + (input_dim,))
        x = nn.Dense(
            self.embed_dim,
            kernel_init=glorot_uniform(),
        )(x)
        return x


class PatchEmbed(nn.Module):
    """
    Convolutional patch embedding.
    See: https://github.com/huggingface/pytorch-image-models/blob/3055411c1bde27c18ac8d654ce85c81731622754/timm/layers/patch_embed.py#L25
    """
    grid_size: Sequence[int]
    patch_size: int
    embed_dim: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        S, B, H, W, C = x.shape

        # forward pass
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size,) * 2,
            strides=(self.patch_size,) * 2,
            use_bias=self.bias,
            padding='SAME',
            kernel_init=glorot_uniform()
        )(x)
        x = reshape(x, (S, B, int(self.grid_size ** 2), self.embed_dim))  # NHWC -> NLC
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron.
    """
    embed_dim: int
    mlp_ratio: float
    act: str = 'gelu'

    @nn.compact
    def __call__(self, x):
        if self.act == 'gelu':
            activation = nn.activation.gelu
        elif self.act == 'relu':
            activation = nn.activation.relu
        elif self.act == 'tanh':
            activation = nn.activation.tanh
        else:
            raise ValueError(f'activation {self.act} not supported')

        x = nn.Dense(
            int(self.embed_dim * self.mlp_ratio),
            kernel_init=glorot_uniform()
        )(x)
        x = activation(x)
        x = nn.Dense(
            self.embed_dim,
            kernel_init=glorot_uniform()
        )(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention Block Module:
    Self Attention -> MLP

    See: https://github.com/huggingface/pytorch-image-models/blob/3055411c1bde27c18ac8d654ce85c81731622754/timm/models/vision_transformer.py#L113
    """
    embed_dim: int
    embed_dim_head: int
    num_heads: int
    mlp_ratio: float
    num_masked_patches: int
    attn_bias: bool = True
    ln_eps: float = 1e-6

    def setup(self):
        self.ln1 = nn.LayerNorm(epsilon=self.ln_eps)
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads * self.embed_dim_head,
            out_features=self.embed_dim,
            deterministic=True,
            kernel_init=nn.initializers.glorot_uniform(),
            use_bias=self.attn_bias,
            normalize_qk=False
        )
        self.ln2 = nn.LayerNorm(epsilon=self.ln_eps)
        self.mlp = MLP(self.embed_dim, self.mlp_ratio)

    @nn.compact
    def __call__(self, x, bit_mask=None):
        """
        x.shape:        (seq, B, L, E)
        bit_mask.shape: (seq, B, L)
        ->.shape:       (seq, B, E)
        """

        if bit_mask is not None:
            idxs = bit_mask.argsort(-1)[:, :, self.num_masked_patches:]
            x = jnp.take_along_axis(
                arr=x,
                indices=idxs[..., np.newaxis],
                axis=-2
            )

        # timm implementation: https://shorturl.at/jkBV4
        x_ln = self.ln1(x)
        x = x + self.attn(x_ln)
        x = x + self.mlp(self.ln2(x))
        x = x.mean(-2)
        return x


class GatedAttentionBlock(nn.Module):
    """
    Attention Block Module:
    Self Attention -> MLP

    See: https://github.com/huggingface/pytorch-image-models/blob/3055411c1bde27c18ac8d654ce85c81731622754/timm/models/vision_transformer.py#L113
    """
    embed_dim: int
    embed_dim_head: int
    num_heads: int
    mlp_ratio: float
    num_masked_patches: int
    attn_bias: bool = True
    ln_eps: float = 1e-6

    def setup(self):
        self.ln1 = nn.LayerNorm(epsilon=self.ln_eps)
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads * self.embed_dim_head,
            out_features=self.embed_dim,
            deterministic=True,
            kernel_init=nn.initializers.glorot_uniform(),
            use_bias=self.attn_bias,
            normalize_qk=False
        )
        self.gate_1 = nn.Dense(
            features=self.embed_dim,
            kernel_init=glorot_uniform()
        )
        self.ln2 = nn.LayerNorm(epsilon=self.ln_eps)
        self.mlp = MLP(self.embed_dim, self.mlp_ratio)
        self.gate_2 = nn.Dense(
            features=self.embed_dim,
            kernel_init=glorot_uniform()
        )

    @nn.compact
    def __call__(self, x, bit_mask=None):
        """
        x.shape:        (seq, B, L, E)
        bit_mask.shape: (seq, B, L)
        ->.shape:       (seq, B, E)
        """

        if bit_mask is not None:
            idxs = bit_mask.argsort(-1)[:, :, self.num_masked_patches:]
            x = jnp.take_along_axis(
                arr=x,
                indices=idxs[..., np.newaxis],
                axis=-2
            )

        # output gate: https://arxiv.org/pdf/1910.06764.pdf
        x_mha = nn.relu(self.attn(self.ln1(x)))
        x_gated = nn.sigmoid(self.gate_1(x))
        x = x + x_mha * x_gated

        x_mlp = nn.relu(self.mlp(self.ln2(x)))
        x_gated = nn.sigmoid(self.gate_2(x))
        x = x + x_mlp * x_gated
        x = x.mean(-2)

        return x


class ConvNet(nn.Module):
    """
    Convolutional Neural Network without pooling.
    """
    kernel_dims: Sequence[int]
    strides: Sequence[int]
    embed_dim: int
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x):
        if self.activation == 'relu':
            activation = nn.activation.relu
        elif self.activation == 'tanh':
            activation = nn.activation.tanh
        elif self.activation == 'gelu':
            activation = nn.activation.gelu
        else:
            raise ValueError(f'activation {self.activation} not supported')
        k1, k2 = self.kernel_dims
        s1, s2 = self.strides
        # x.shape = (S, B, 10, 10, 4)
        x = nn.Conv(
            32, (k1, k1), strides=(s1, s1),
            kernel_init=nn.initializers.glorot_uniform()
        )(x)
        x = activation(x)
        x = nn.Conv(
            64, (k2, k2), strides=(s2, s2),
            kernel_init=nn.initializers.glorot_uniform()
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))  # flatten, keep seq and batch dims
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        return x


class ScannedRNN(nn.Module):
    """
    Scanned RNN Breakdown
    ---------------------
    The full x tensor to be scanned has shape: (sequence, batch, features)
    The full dones tensor to be scanned has shape: (sequence,)

    However, since the function is wrapped in nn.transforms.scan, the
    `inputs` tuple contains `x_t` and `dones_t`; single slices along the time
    dimension to be used in a single pass through the RNN.

    `carry` always has shape (batch, features) as it is not scanned.

    Note: in RL, "batch" is "num envs"
    """

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=0, out_axes=0,  # NB: time dimension first!!
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, inputs):
        x_t, dones_t = inputs
        B, E = x_t.shape
        carry = jnp.where(  #  reset state if terminal state is reached
            dones_t[:, np.newaxis],  # make 2D: (B,) -> (B, 1)
            self.initialize_carry(B, E),
            carry,
        )
        carry, y = nn.GRUCell(E)(carry, x_t)  # y == x_{t+1}

        carry = jnp.asarray(carry, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        return carry, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorHead(nn.Module):
    """
    Special Multi-Layer Perceptron for Actor Head.
    """
    hidden_dim: int
    activation: str
    action_dim: int

    @nn.compact
    def __call__(self, x):
        if self.activation == 'relu':
            activation = nn.activation.relu
        elif self.activation == 'tanh':
            activation = nn.activation.tanh
        elif self.activation == 'gelu':
            activation = nn.activation.gelu
        else:
            raise ValueError(f'activation {self.activation} not supported')

        actor_mean = nn.Dense(
            features=self.hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)
        return actor_mean


class CriticHead(nn.Module):
    """
    Special Multi-Layer Perceptron for Critic Head.
    """
    hidden_dim: int
    activation: str

    @nn.compact
    def __call__(self, x):
        if self.activation == 'relu':
            activation = nn.activation.relu
        elif self.activation == 'tanh':
            activation = nn.activation.tanh
        elif self.activation == 'gelu':
            activation = nn.activation.gelu
        else:
            raise ValueError(f'activation {self.activation} not supported')

        critic = nn.Dense(
            features=self.hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)
        return critic
