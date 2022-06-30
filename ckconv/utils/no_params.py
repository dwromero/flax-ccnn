import jax
import jax.numpy as jnp
from flax.training import train_state

def no_params(
        model_state: train_state.TrainState,
) -> int:
    """
    Calculates the number of parameters in the state of a flax.linen.Module.
    """
    # Get no params per leave
    params_per_leave = jax.tree_map(lambda x: sum(jnp.ravel(x).shape), model_state.params)
    # Flatten the PyTree and sum the size of all leaves
    no_params = sum(jax.tree_util.tree_flatten(params_per_leave)[0])
    return no_params
