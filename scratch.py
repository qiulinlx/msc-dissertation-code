import xminigrid
import jax.numpy as jnp
import jax
import jax.tree_util as jtu

import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

def show_img(img, dpi=32):
    plt.figure(dpi=dpi)
    plt.axis('off')
    plt.imshow(img)

key = jax.random.key(0)
key, reset_key = jax.random.split(key)

# to list available environments: xminigrid.registered_environments()
env, env_params = xminigrid.make("MiniGrid-Empty-8x8")
print("Observation shape:", env.observation_shape(env_params))
print("Num actions:", env.num_actions(env_params))

# fully jit-compatible step and reset methods
timestep = jax.jit(env.reset)(env_params, reset_key)
timestep = jax.jit(env.step)(env_params, timestep, action=0)

print("TimeStep shapes:", jtu.tree_map(jnp.shape, timestep))

show_img(env.render(env_params, timestep), dpi=64)