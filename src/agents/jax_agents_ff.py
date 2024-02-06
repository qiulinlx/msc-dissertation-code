import math

from typing import Sequence, Dict

import distrax
import jax

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal, glorot_uniform

from jax import numpy as jnp

from src.agents.jax_modules import (
    DenseEmbed,
    PatchEmbed,
    ConvNet,
    AttentionBlock,
    ScannedRNN,
    ActorHead,
    CriticHead,
    MLP
)

from src.utils.masking import make_random_binary_mask_1D, broadcast_to_2D_mask

from src.utils.positional_encoding import get_2d_sincos_pos_embed


################
# Regular Agents
################

class RegularAgentDense(nn.Module):
    obs_shape: Sequence[int]  # unused
    action_dim: int
    noise_masking: bool
    embed_dim: int
    activation: str
    head_ratio: float
    mask_ratio: float  # unused
    continuous_actions: bool = False

    def setup(self):
        head_hidden_dim = int(self.embed_dim * self.head_ratio)
        self.body = MLP(
            embed_dim=self.embed_dim,
            act=self.activation,
            mlp_ratio=2.0
        )
        self.rnn = ScannedRNN()
        self.actor_head = ActorHead(head_hidden_dim, self.activation, self.action_dim)
        self.critic_head = CriticHead(head_hidden_dim, self.activation)

    @staticmethod
    def generate_mask(rng, obs_shape, mask_ratio, unused=None, add_noise=False):
        B, L = obs_shape
        rng_mask, rng_noise = jax.random.split(rng)
        mask = make_random_binary_mask_1D(rng, shape=(B, L), percent_zeros=mask_ratio)
        if add_noise:
            noise = jax.random.uniform(rng_noise, shape=mask.shape)
            mask = jnp.stack([mask, noise], axis=-1)
        return mask

    @nn.compact
    def __call__(self, hidden, inputs, get_latent_code=False):
        obs, dones = inputs

        # network body
        m = self.body(obs)

        # RNN layer
        rnn_in = (m, dones)
        hidden, x = self.rnn(hidden, rnn_in)

        # actor head
        actor_mean = self.actor_head(x)
        if self.continuous_actions:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:  # discrete actions
            pi = distrax.Categorical(logits=actor_mean)

        # critic head
        critic = self.critic_head(x)

        if get_latent_code:
            return hidden, pi, jnp.squeeze(critic, axis=-1), m
        return hidden, pi, jnp.squeeze(critic, axis=-1)


class RegularAgentCNN(nn.Module):
    obs_shape: Sequence[int]
    kernel_dims: Sequence[int]
    strides: Sequence[int]
    conv_features: Sequence[int]
    embed_dim: int
    activation: str
    head_ratio: float
    action_dim: int
    noise_masking: bool
    patch_size: int  # used in generate_mask
    mask_ratio: float  # used in generate_mask
    continuous_actions: bool = False

    def setup(self):
        head_hidden_dim = int(self.embed_dim * self.head_ratio)
        self.conv = ConvNet(self.kernel_dims, self.strides, self.embed_dim)
        self.rnn = ScannedRNN()
        self.actor_head = ActorHead(head_hidden_dim, self.activation, self.action_dim)
        self.critic_head = CriticHead(head_hidden_dim, self.activation)

    @staticmethod
    def generate_mask(rng, obs_shape, mask_ratio, patch_size, add_noise=False):
        B, H, W, C = obs_shape
        assert H == W, "Only square observations allowed"
        assert H % patch_size == 0, "Patch size must divide observation size"
        grid_size = math.ceil(H / patch_size)
        rng_mask, rng_noise = jax.random.split(rng, 2)
        mask_1D = make_random_binary_mask_1D(rng_mask, shape=(B, int(grid_size ** 2)), percent_zeros=mask_ratio)
        mask_2D = broadcast_to_2D_mask(mask_1D, obs_shape, patch_size)
        if add_noise:
            noise = jax.random.uniform(rng_noise, shape=mask_2D.shape)
            mask_2D = jnp.stack([mask_2D, noise], axis=-1)
        return mask_2D

    @nn.compact
    def __call__(self, hidden, inputs, get_latent_code=False):
        obs, dones = inputs

        # network body
        m = self.conv(obs)

        # RNN layer
        rnn_in = (m, dones)
        hidden, x = self.rnn(hidden, rnn_in)

        # actor head (from jax implementation)
        actor_mean = self.actor_head(x)
        if self.continuous_actions:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:  # discrete actions
            pi = distrax.Categorical(logits=actor_mean)

        # critic head
        critic = self.critic_head(x)

        if get_latent_code:
            return hidden, pi, jnp.squeeze(critic, axis=-1), m
        return hidden, pi, jnp.squeeze(critic, axis=-1)
