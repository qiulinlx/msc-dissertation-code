from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.training.train_state import TrainState

from src.purejaxrl.purejaxrl.wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    UpscaleMinAtarObservationWrapper
)

from src.agents.jax_agents import (
    RegularAgentDense,
    RegularAgentCNN,
    AttentionAgentDense,
    AttentionAgentCNN
)
from src.agents.jax_modules import ScannedRNN

MINATAR_ENVS = [
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "SpaceInvaders-MinAtar",
    "Freeway-MinAtar",
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


def make_ppo_train(config):
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"], **config["ENV_CONFIG"])
    if config['AGENT'] in DENSE_AGENTS:
        env = FlattenObservationWrapper(env)
    if config["ENV_NAME"] in MINATAR_ENVS:
        env = UpscaleMinAtarObservationWrapper(env, config["WRAPPER_CONFIG"]["resize_method"])
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
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

        # GENERATE INPUTS
        rng, _rng = jax.random.split(rng)
        init_mask = network.generate_mask(
            _rng,
            (config["NUM_ENVS"],) + env.observation_space(env_params).shape,
            config["AGENT_CONFIG"].get("mask_ratio"),
            config["AGENT_CONFIG"].get("patch_size", None),
            config["AGENT_CONFIG"].get("noise_masking", False)
        )
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            init_mask[np.newaxis, :]
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["AGENT_CONFIG"]["embed_dim"])
        rng, _rng = jax.random.split(rng)

        # INIT NETWORK PARAMS
        network_params = network.init(_rng, init_hstate, init_x)

        # MAKE TRAIN STATE
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["AGENT_CONFIG"]["embed_dim"])

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # GENERATE MASK
                rng, _rng = jax.random.split(rng)
                last_mask = network.generate_mask(
                    _rng,
                    last_obs.shape,
                    config["AGENT_CONFIG"].get("mask_ratio"),
                    config["AGENT_CONFIG"].get("patch_size", None),  # only used by AttentionAgentCNN
                    config["AGENT_CONFIG"].get("noise_masking", False)  # only used by RegularAgent*
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :], last_mask[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, last_mask, info
                )
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            #  generate mask
            rng, _rng = jax.random.split(rng)
            last_mask = network.generate_mask(
                _rng,
                last_obs.shape,
                config["AGENT_CONFIG"].get("mask_ratio"),
                config["AGENT_CONFIG"].get("patch_size", None),  # only used by AttentionAgentCNN
                config["AGENT_CONFIG"].get("noise_masking", False)  # only used by RegularAgent*
            )

            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :], last_mask[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done, traj_batch.mask)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    episode_lens = info["returned_episode_lengths"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                            info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, ",
                            f"episode length={episode_lens[t]}, ",
                            f"return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def make_agent_evaluation(config):
    # MAKE + INIT ENV
    env, env_params = gymnax.make(config["ENV_NAME"], **config["ENV_CONFIG"])
    if config["AGENT"] in DENSE_AGENTS:
        env = FlattenObservationWrapper(env)
    if config["ENV_NAME"] in MINATAR_ENVS:
        env = UpscaleMinAtarObservationWrapper(env, config["WRAPPER_CONFIG"]["resize_method"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(config["PARENT_SEED"])
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
                config["AGENT_CONFIG"].get("mask_ratio"),
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

        return traj_batch

    return agent_eval
