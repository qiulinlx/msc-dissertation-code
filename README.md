# Description

This repository contains code for the experiments in my MSc dissertation titled "Self-Attention Policy Architectures for Reinforcement Learning Under Partial Observability".

Abstract:

_Intermittent unavailability of sensory signals due to sensor failure and/or latency is a problem encountered in production environments such as in large manufacturing plants, for example. Deep reinforcement learning offers a natural solution for process control and optimisation in such environments. However, a shortcoming of conventional agent policy architectures in this instance is an inability to handle variable-sized inputs composed of available sensory signals, thus requiring the imputation of unavailable sensory signals with data which necessarily constitutes noise. We explore self-attention-based policy architectures as a solution to this problem, demonstrating their robustness under conditions of high partial observability on different reinforcement learning benchmark tasks, and explore the advantages and disadvantages offered by our solution over conventional policy architectures. Additionally, we propose a novel hard attention mechanism, used in conjunction with our proposed policy architecture, enabling the agent to attend to the most salient sensory signals and allowing for greater interpretability of the agent's decision-making._

# Usage

Install the dependencies in `src/purejaxrl/requirements.txt`.

Experiments may be launched by running the `main.py` file and the boilerplate agent configs are in `src/experiment_config.json`.

Additional command line arguments are:

* `--base_dir`            The directory to which the training results will be written.
* `--env`                 The name of the environment you'd like to run (e.g. "CartPole-v2" or "Acrobot-v1")
* `--agent`               The name of the agent (e.g. `AttentionAgentDense`, `AttentionAgentCNN`, `RegularAgentDense`, `RegularAgentCNN`)
* `--forward_fill`        Whether or not to use the ‘forward-fill masking’ method of imputation (only compatible with `RegularAgentDense` and `RegularAgentCNN`)
* `--noise_masking`       Whether or not to use the ‘noise masking’ method of imputation (only compatible with `RegularAgentDense` and `RegularAgentCNN`)
* `--generated_masking`   Have the agent generate masks to sample the sensors directly as opposed to random sampling (only compatible with `AttentionAgentDense` and `AttentionAgentCNN`)
* `--envpool`             Whether the chosen environment is from [EnvPool](https://envpool.readthedocs.io/en/latest/index.html) (as opposed to [Gymnax](https://github.com/RobertTLange/gymnax))
* `--seed`                For setting the random seed       

Example: train the baseline convolutional agent on CarRacing-v2 with noise masking: 
```
python main.py
--env CarRacing-v2
--agent RegularAgentCNN
--noise_masking
--envpool
--base_dir '/path/to/save/stuff'
--seed 3
```
