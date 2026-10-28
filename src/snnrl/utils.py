import jax.numpy as jnp
import numpy as np

from snnrl.network import Weights

def load_results(path):
    saved = np.load(path)
    weights = Weights(**{k[2:]: jnp.asarray(saved[k]) for k in saved.files if k.startswith("w_")})
    history = {k[8:]: saved[k] for k in saved.files if k.startswith("history_")}
    return weights, history


def save_results(path, weights, history):
    path.parent.mkdir(exist_ok=True)
    np.savez(path,
             **{f"w_{k}": np.asarray(v) for k, v in vars(weights).items()},
             **{f"history_{k}": np.array([r.get(k, np.nan) for r in history], dtype=float)
                for k in sorted(set().union(*history))})