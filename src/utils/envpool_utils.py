import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit

def discretise_actions(action):
    action_map = jnp.array([
      [0., 0., 0.],
      [1., 0., 0.],
      [-1., 0., 0.],
      [0., 1., 0.],
      [0., 0., 1.]
    ])
    return jnp.take(action_map, action, axis=0)


def grayscale_obs(obs):
    obs = obs / 255.0
    obs = jnp.dot(obs[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))
    return obs[..., None]  # add channel dim


def record_episode_stats(episode_stats, transition):
    obs, reward, done, truncated, info = transition
    done_any = done + truncated

    # update stats state
    new_episode_return = episode_stats.episode_returns + reward
    new_episode_length = episode_stats.episode_lengths + 1

    episode_stats = episode_stats.replace(
        timestep=episode_stats.timestep + 1,
        episode_returns=(new_episode_return) * (1 - done_any),
        episode_lengths=(new_episode_length) * (1 - done_any),
        returned_episode_returns=(
                episode_stats.returned_episode_returns * (1 - done_any) + new_episode_return * done_any
        ),
        returned_episode_lengths=(
                episode_stats.returned_episode_lengths * (1 - done_any) + new_episode_length * done_any
        ),
    )

    # replace info
    info = {
      "returned_episode_returns": episode_stats.returned_episode_returns,
      "returned_episode_lengths": episode_stats.returned_episode_lengths,
      "timestep": jnp.full(done.shape, episode_stats.timestep),
      "returned_episode": done_any
    }

    return episode_stats, done_any, info


def carracing_step(step_fn, env_state, action, episode_stats):
    # step in environment
    action = discretise_actions(action)
    env_state, (obs, reward, done, truncated, info) = step_fn(env_state, action)
    obs = grayscale_obs(obs)
    episode_stats, done_any, info = record_episode_stats(
        episode_stats, (obs, reward, done, truncated, info))

    return episode_stats, env_state, (obs, reward, done_any, info)
