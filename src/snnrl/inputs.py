import jax
import jax.numpy as jnp


def poisson_spikes(key: jax.Array, rates: jax.Array, dt: float = 1.0) -> jax.Array:
    """
    Poisson spike trains with given (possibly time-varying) rates.

    Each input spikes at most once per time step, with probability rate * dt
    (a Bernoulli approximation of a Poisson process, as in Bellec et al. 2020).

    Parameters
    ----------
    key: jax.Array
        Random key.
    rates: jax.Array
        Firing rates in Hz, shape (T, n).
    dt: float
        Time step (ms).

    Returns
    -------
    jax.Array
        Spikes (0.0 or 1.0), shape (T, n).
    """
    p = rates * dt / 1000.0
    return (jax.random.uniform(key, rates.shape) < p).astype(jnp.float32)
