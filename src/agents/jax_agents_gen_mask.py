import math

from typing import Sequence, Dict

import distrax

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
    CriticHead
)

from src.utils.masking import sample_random_binary_mask_1D, broadcast_to_2D_mask

from src.utils.positional_encoding import get_2d_sincos_pos_embed


################
# Regular Agents
################

class RegularAgentDense(nn.Module):
    obs_shape: Sequence[int]  # unused
    mask_ratio: float  # unused
    embed_dim: int
    activation: str
    head_ratio: float
    action_dim: int
    num_heads: int  # unused
    mlp_ratio: float  # unused

    def setup(self):
        head_hidden_dim = int(self.embed_dim * self.head_ratio)
        self.dense = nn.Dense(
            self.embed_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )
        self.rnn = ScannedRNN()
        self.actor_head = ActorHead(head_hidden_dim, self.activation, self.action_dim)
        self.critic_head = CriticHead(head_hidden_dim, self.activation)

    @staticmethod
    def generate_mask(key, probs, mask_ratio, unused1, unused2):
        # probs: (B, N)
        return sample_random_binary_mask_1D(key, probs, percent_zeros=mask_ratio)

    @nn.compact
    def __call__(self, hidden, inputs):
        if self.activation == 'relu':
            activation = nn.activation.relu
        elif self.activation == 'tanh':
            activation = nn.activation.tanh
        elif self.activation == 'gelu':
            activation = nn.activation.gelu
        else:
            raise ValueError(f'activation {self.activation} not supported')

        obs, dones, masks = inputs
        x = obs * masks  # zero masking
        x = self.dense(x)
        x = activation(x)

        # RNN layer
        rnn_in = (x, dones)
        hidden, x = self.rnn(hidden, rnn_in)

        # actor head
        actor_mean = self.actor_head(x)
        pi = distrax.Categorical(logits=actor_mean)

        # critic head
        critic = self.critic_head(x)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class RegularAgentCNN(nn.Module):
    obs_shape: Sequence[int]
    embed_dim: int
    activation: str
    head_ratio: float
    action_dim: int
    patch_size: int  # unused
    mask_ratio: float  # unused
    num_heads: int  # unused
    mlp_ratio: float  # unused

    def setup(self):
        head_hidden_dim = int(self.embed_dim * self.head_ratio)
        self.conv = ConvNet(self.embed_dim)
        self.rnn = ScannedRNN()
        self.actor_head = ActorHead(head_hidden_dim, self.activation, self.action_dim)
        self.critic_head = CriticHead(head_hidden_dim, self.activation)

    @staticmethod
    def generate_mask(key, probs, mask_ratio, obs_shape, patch_size):
        # probs: (B, N)
        mask_1D = sample_random_binary_mask_1D(key, probs, percent_zeros=mask_ratio)
        return broadcast_to_2D_mask(mask_1D, obs_shape, patch_size)

    @nn.compact
    def __call__(self, hidden, inputs):
        obs, dones, masks = inputs
        S, B, H, W, C = obs.shape  # (seq, batch, H, W, C)

        # CONVOLUTIONAL LAYERS
        x = obs * masks  # zero masking
        x = self.conv(x)

        # RNN layer
        rnn_in = (x, dones)
        hidden, x = self.rnn(hidden, rnn_in)

        # actor head (from jax implementation)
        actor_mean = self.actor_head(x)
        pi = distrax.Categorical(logits=actor_mean)

        # critic head
        critic = self.critic_head(x)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


##################
# Attention Agents
##################

class AttentionAgentDense(nn.Module):
    obs_shape: Sequence[int]
    mask_ratio: float
    embed_dim: int
    activation: str
    num_heads: int
    head_dim: int
    mlp_ratio: float
    head_ratio: float
    action_dim: int

    def setup(self):
        head_hidden_dim = int(self.embed_dim * self.head_ratio)
        self.num_masked_features = int(self.obs_shape[0] * self.mask_ratio)
        self.embedding = DenseEmbed(self.embed_dim)
        self.attn_block = AttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            embed_dim_head=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            num_masked_patches=self.num_masked_features
        )
        self.rnn = ScannedRNN()
        self.mask_head = ActorHead(head_hidden_dim, self.activation, self.obs_shape[0])
        self.actor_head = ActorHead(head_hidden_dim, self.activation, self.action_dim)
        self.critic_head = CriticHead(head_hidden_dim, self.activation)

    @staticmethod
    def generate_mask(key, probs, mask_ratio, unused, unused2):
        # probs: (B, N)
        return sample_random_binary_mask_1D(key, probs, percent_zeros=mask_ratio)

    @nn.compact
    def __call__(self, hidden, inputs):
        obs, dones, masks = inputs

        # embedding layer
        x = self.embedding(obs)

        # attention -> MLP
        x = self.attn_block(x, masks)

        # RNN layer
        rnn_in = (x, dones)
        hidden, x = self.rnn(hidden, rnn_in)

        # actor head (from jax implementation)
        actor_mean = self.actor_head(x)
        pi = distrax.Categorical(logits=actor_mean)

        # mask head
        mask_logits = self.mask_head(x)
        mask_logits = nn.activation.tanh(mask_logits)
        mu = distrax.Multinomial(self.obs_shape[0] - self.num_masked_features, logits=mask_logits)

        # critic head
        critic = self.critic_head(x)

        return hidden, pi, mu, jnp.squeeze(critic, axis=-1)


class AttentionAgentCNN(nn.Module):
    obs_shape: Sequence[int]
    action_dim: int
    patch_size: int
    mask_ratio: float
    embed_dim: int
    num_heads: int
    head_dim: int
    mlp_ratio: float
    head_ratio: float
    activation: str


    def setup(self):
        H, W, C = self.obs_shape
        assert H == W, "Only square images are supported"
        assert H % self.patch_size == 0, "Patch size must divide observation size"
        grid_size = H // self.patch_size
        self.num_patches = int(grid_size ** 2)
        self.num_masked_patches = int(self.mask_ratio * self.num_patches)
        self.positional_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size)
        head_hidden_dim = int(self.embed_dim * self.head_ratio)
        self.embedding = PatchEmbed(
            grid_size=grid_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
        self.attn_block = AttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            embed_dim_head=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            num_masked_patches=self.num_masked_patches
        )
        self.rnn = ScannedRNN()
        self.actor_head = ActorHead(head_hidden_dim, self.activation, self.action_dim)
        self.mask_head = ActorHead(head_hidden_dim, self.activation, self.num_patches)
        self.critic_head = CriticHead(head_hidden_dim, self.activation)

    @staticmethod
    def generate_mask(key, probs, mask_ratio, unused1, unused2):
        # probs: (B, N)
        return sample_random_binary_mask_1D(key, probs, percent_zeros=mask_ratio)

    @nn.compact
    def __call__(self, hidden, inputs):
        obs, dones, masks = inputs

        # embedding + positional encoding
        x = self.embedding(obs)
        x = x + self.positional_encoding

        # attention -> MLP
        x = self.attn_block(x, masks)

        # RNN layer (GRU)
        rnn_in = (x, dones)
        hidden, x = self.rnn(hidden, rnn_in)

        # policy head
        actor_mean = self.actor_head(x)
        pi = distrax.Categorical(logits=actor_mean)

        # mask head
        mask_logits = self.mask_head(x)
        mask_logits = nn.activation.tanh(mask_logits)
        mu = distrax.Multinomial(self.num_patches - self.num_masked_patches, logits=mask_logits)

        # value head
        critic = self.critic_head(x)

        return hidden, pi, mu, jnp.squeeze(critic, axis=-1)
