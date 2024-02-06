from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
from src.jax_agents import RegularAgentDense, AttentionAgentDense, RegularAgentCNN, AttentionAgentCNN
from src.jax_modules import ScannedRNN
from src.purejaxrl.purejaxrl.wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    UpscaleMinAtarObservationWrapper
)

MINATAR_ENVS = [
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "SpaceInvaders-MinAtar",
    "Freeway-MinAtar",
    "Pong-misc",
]

DENSE_AGENTS = [
    "RegularAgentDense",
    "AttentionAgentDense"
]


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray
    info: jnp.ndarray


def make_agent_evaluation(config):
    # MAKE + INIT ENV
    env, env_params = gymnax.make(config["ENV_NAME"], **config["ENV_CONFIG"])
    if config["AGENT"] in DENSE_AGENTS:
        env = FlattenObservationWrapper(env)
    if config["ENV_NAME"] in MINATAR_ENVS:
        env = UpscaleMinAtarObservationWrapper(env)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(30)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_EVAL_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    # INIT NETWORK
    if config["AGENT"] == "AttentionAgentDense":
        agent = AttentionAgentDense
    elif config["AGENT"] == "RegularAgentDense":
        agent = RegularAgentDense
    elif config["AGENT"] == "AttentionAgentCNN":
        agent = AttentionAgentCNN
    elif config["AGENT"] == "RegularAgentCNN":
        agent = RegularAgentCNN
    else:
        raise ValueError(f"Unknown agent {config['AGENT']}")
    network = agent(
        obs_shape=env.observation_space(env_params).shape,
        action_dim=env.action_space(env_params).n,
        **config["AGENT_CONFIG"]
    )

    def agent_eval(params):
        def _env_step(runner_state, unused):
            params, env_state, last_obs, last_done, hstate, rng = runner_state
            rng, _rng = jax.random.split(rng)

            # GENERATE_MASKS
            last_mask = network.generate_mask(
                rng,
                last_obs.shape,
                config["MASK_RATIO"],
                config["AGENT_CONFIG"].get("patch_size", None),
                config["AGENT_CONFIG"].get("noise_masking", False)  # only used by RegularAgent*
            )
            rng, _rng = jax.random.split(rng)

            # SELECT ACTION
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :], last_mask[np.newaxis, :])
            hstate, pi, value = network.apply(params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_EVAL_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, env_params)
            transition = Transition(
                done, action, value, reward, log_prob, last_obs, last_mask, info
            )
            runner_state = (params, env_state, obsv, done, hstate, rng)
            return runner_state, transition

        runner_state = (
            params,
            env_state,
            obsv,
            jnp.zeros((config["NUM_EVAL_ENVS"]), dtype=bool),  # dones
            ScannedRNN.initialize_carry(  # hstate
                config["NUM_EVAL_ENVS"], config["AGENT_CONFIG"]["embed_dim"]),
            _rng,
        )

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_EVAL_STEPS"]
        )

        return traj_batch.info

    return agent_eval


def evaluate(config, train_state):
    mean_returns = []
    for i in range(min(10, config['NUM_CHILD_SEEDS'])):
        params = jax.tree_util.tree_map(lambda x: x[i], train_state.params)
        eval_agent_jit = jax.jit(make_agent_evaluation(config))
        eval_metric = eval_agent_jit(params)
        eps_per_env = eval_metric['returned_episode'].sum()
        mean_return = (eval_metric['returned_episode'] * eval_metric['returned_episode_returns']).sum() / eps_per_env

        R = jnp.round(mean_return, 2)
        mean_returns.append(R)
        print(f"Return (seed {i}): {R}")

    mean_return = np.mean(mean_returns)
    print(f"Mean return (all seeds): {mean_return}")

    return eval_metric, mean_return
