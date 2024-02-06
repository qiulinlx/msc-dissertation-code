import json
import math


def generate_config_list(
        env_name: str,
        agent_name: str,
        noise_masking: bool,
        resize_method: str,
        generated_masking: bool,
):
    def load_config(env_name):
        with open("experiment_config.json", "r") as f:
            config = json.load(f)[env_name]
        return config

    env_config = load_config(env_name)
    agent_config = env_config[agent_name]

    # vision envs are 30x30x3, so we mask in increments of 3 (12%)
    if agent_name in ["RegularAgentCNN", "AttentionAgentCNN"]:
        H, W, C = env_config["obs_shape"]
        patch_size = agent_config["AGENT_CONFIG"]["patch_size"]
        assert H % patch_size == 0 and W % patch_size == 0
        num_sensors = (H // patch_size) * (W // patch_size)
        step_size = math.ceil(num_sensors / 10)
        mask_ratios = [i / num_sensors for i in range(0, num_sensors, step_size)]
    else:
        num_sensors = env_config["obs_shape"][0]
        assert num_sensors <= 10
        mask_ratios = [i / num_sensors for i in range(num_sensors)]

    if env_name == "CarRacing-v2":
        mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    config_list = []
    for ratio in mask_ratios:
        config = load_config(env_name)[agent_name]
        config["GENERATED_MASKING"] = generated_masking
        config["AGENT_CONFIG"]["mask_ratio"] = ratio

        if agent_name in ["RegularAgentDense", "RegularAgentCNN"]:
            config["AGENT_CONFIG"]["noise_masking"] = noise_masking

        if env_name.endswith("MinAtar"):
            config["WRAPPER_CONFIG"]["resize_method"] = resize_method

        config_list.append(config)

    return config_list
