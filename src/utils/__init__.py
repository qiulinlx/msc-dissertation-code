import json
import os

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint

from flax.training import orbax_utils


def chkpt_save(train_state, fdir):
    chkpt_dir = f"{fdir}/chkpt"
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    ckptr = orbax.checkpoint.Checkpointer(
        orbax.checkpoint.PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    ckptr.save(
        chkpt_dir,
        train_state,
        save_args=orbax_utils.save_args_from_target(train_state), force=True
    )


def chkpt_load(train_state, fdir):
    ckptr = orbax.checkpoint.Checkpointer(
        orbax.checkpoint.PyTreeCheckpointHandler())  # A stateless object, can be created on the fly.
    ckptr.restore(
        fdir,
        item=train_state,
        restore_args=flax.training.orbax_utils.restore_args_from_target(train_state, mesh=None)
    )
    ckptr.restore(fdir, item=None)
    return train_state


def write_metric(metric, train, fdir):
    metric_dir = f"{fdir}/metrics"
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)
    for k in list(metric.keys()):
        metric_k = metric[k].copy()
        metric_shape = metric_k.shape
        if len(metric_shape) == 4:
            n_seeds, n_batches, n_steps, n_envs = metric_k.shape
            metric_k = metric_k.reshape(n_seeds, n_batches * n_steps, n_envs)
        else:
            n_batches, n_steps, n_envs = metric_k.shape
            metric_k = metric_k.reshape(n_batches * n_steps, n_envs)
        with open(f"{metric_dir}/{k}_{'train' if train else 'test'}.npy", "wb") as f:
            np.save(f, metric_k)


def write_config(config, fdir):
    config_cp = config.copy()
    with open(f"{fdir}/config.json", "w") as f:
        f.write(json.dumps(config_cp, indent=4))


def save_metric_plots(group_id, config, outs, fdir):
    plots_dir = f"{fdir}/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    ts = outs['metric']['timestep'].reshape(config["NUM_CHILD_SEEDS"], -1, config["NUM_ENVS"]).mean(-1)
    for var in ['returned_episode_returns', 'returned_episode_lengths']:
        y = outs['metric'][var].reshape(config["NUM_CHILD_SEEDS"], -1, config["NUM_ENVS"]).mean(-1)
        y_mean = y.mean(0)
        y_delta = y.std(0)
        y_min, y_max = y.min(0), y.max(0)
        y_under = jnp.clip(y_mean - y_delta, y_min, y_max)
        y_over = jnp.clip(y_mean + y_delta, y_min, y_max)

        plt.figure(figsize=(15, 7.5))
        plt.title(f"{var}\n{group_id}")
        plt.plot(y_mean, linewidth=2)
        plt.fill_between(ts[0, :], y_under, y_over, color='red', alpha=.3)
        # plt.show()
        plt.savefig(f"{plots_dir}/{var}.png")


def log_save_experiment(group_id, config, outs, exp_dir):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # save config
    print(f"Saving config for {group_id}...")
    write_config(config, exp_dir)

    # save train_state with orbax
    print(f"Saving train_state for {group_id}...")
    train_state = outs["runner_state"][0]
    chkpt_save(train_state, exp_dir)

    # save metrics
    print(f"Saving metrics for {group_id}...")
    write_metric(metric=outs["metric"], train=True, fdir=exp_dir)
    save_metric_plots(group_id, config, outs, exp_dir)


def evaluate(make_eval_fn, config, outs):
    train_state = outs["runner_state"][0]
    mean_returns = []
    for i in range(config['NUM_CHILD_SEEDS']):
        if config['NUM_CHILD_SEEDS'] == 1:
            params = train_state.params
        else:
            params = jax.tree_util.tree_map(lambda x: x[i], train_state.params)
        eval_agent_jit = jax.jit(make_eval_fn(config))
        eval_metric = eval_agent_jit(params)
        eps_per_env = eval_metric['returned_episode'].sum()
        mean_return = (eval_metric['returned_episode'] * eval_metric['returned_episode_returns']).sum() / max(eps_per_env, 1)

        R = jnp.round(mean_return, 2)
        mean_returns.append(R)
        print(f"Return (seed {i}): {R}")

    mean_return = np.mean(mean_returns)
    print(f"Mean return (all seeds): {mean_return}")

    return eval_metric, mean_return
