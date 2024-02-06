import argparse
import json
import time

import jax

from src.utils import log_save_experiment, evaluate
from src.utils.config import generate_config_list


def import_ppo(forward_fill: bool, generated_masking: bool, envpool: bool):
    if not forward_fill and not generated_masking:
        if not envpool:
            from src.ppo.ppo_jax import (
                make_ppo_train,
                make_agent_evaluation
            )
        elif envpool:
            from src.ppo.ppo_jax_envpool import (
                make_ppo_train,
                make_agent_evaluation
            )

    elif generated_masking and not forward_fill:
        if not envpool:
            from src.ppo.ppo_jax_gen_mask import (
                make_ppo_train,
                make_agent_evaluation
            )
        elif envpool:
            from src.ppo.ppo_jax_gen_mask_envpool import (
                make_ppo_train,
                make_agent_evaluation
            )

    elif forward_fill and not generated_masking:
        if not envpool:
            from src.ppo.ppo_jax_forward_fill import (
                make_ppo_train,
                make_agent_evaluation
            )
        elif envpool:
            from src.ppo.ppo_jax_forward_fill_envpool import (
                make_ppo_train,
                make_agent_evaluation
            )

    else:
        raise ValueError("Invalid combination of envpool and generated_masking")
    return make_ppo_train, make_agent_evaluation


def main(
        base_dir: str,
        env: str,
        agent: str,
        forward_fill: bool,
        noise_masking: bool,
        generated_masking:bool,
        resize_method: str,
        envpool: bool,
        seed: int
):
    config_list = generate_config_list(
        env_name=env,
        agent_name=agent,
        noise_masking=noise_masking,
        resize_method=resize_method,
        generated_masking=generated_masking
    )

    # experiment tag
    if generated_masking:
        experiment = "fixed_capacity_generated_masking"
    else:
        experiment = "fixed_capacity_random_masking"

    # import training and eval functions
    make_ppo_train, make_agent_evaluation = import_ppo(forward_fill, generated_masking, envpool)

    for config in config_list:
        if seed is not None:
            config["PARENT_SEED"] = seed
        group_id = "{}_PPO_maskratio={}_parentseed={}".format(
            config['AGENT'],
            config['AGENT_CONFIG']['mask_ratio'],
            config['PARENT_SEED'],
            int(time.time())
        )
        exp_dir = f"{base_dir}/{group_id}"

        print("-" * 50)
        print(f"Running {experiment} for {group_id} on {env}...")
        print(f"Will write to {exp_dir}...")
        print(f"CONFIG:\n{json.dumps(config, indent=4)}")

        rng = jax.random.PRNGKey(config["PARENT_SEED"])
        if not envpool:
            rng = jax.random.split(rng, config["NUM_CHILD_SEEDS"])
            train = jax.jit(jax.vmap(make_ppo_train(config)))
        else:
            train = jax.jit(make_ppo_train(config))
        outs = train(rng)
        log_save_experiment(group_id, config, outs, exp_dir)

        print(f"Evaluating agent for {group_id}...")

        eval_metric, mean_return = evaluate(
            make_eval_fn=make_agent_evaluation,
            config=config,
            outs=outs
        )

        print("DONE.")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AttentionAgent experiment')
    parser.add_argument('--base_dir', type=str)
    parser.add_argument('--env', type=str, choices=[
        'CartPole-v1',
        'Acrobot-v1',
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "SpaceInvaders-MinAtar",
        "Freeway-MinAtar",
        "Pong-misc",
        "CarRacing-v2"
    ])
    parser.add_argument('--agent', type=str, choices=[
        'AttentionAgentDense',
        'AttentionAgentCNN',
        'RegularAgentDense',
        'RegularAgentCNN',
    ])
    parser.add_argument('--forward_fill', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--noise_masking', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--generated_masking', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--resize_method', type=str, default="nearest", choices=[
        "nearest",
        "bicubic"
    ])
    parser.add_argument('--envpool', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
