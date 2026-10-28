import jax
import jax.numpy as jnp

from dataclasses import dataclass


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


CARTPOLE_SCALES = (2.4, 2.0, 0.21, 3.0)
# Cart position: episode ends when |x| > 2.4 m
# Cart velocity: hand-tuned
# Pole angle: episode ends when |theta| > 12 degrees = 0.21 rad
# Pole angular velocity: hand-tuned


@dataclass(frozen=True)
class ObservationEncoder:
    """
    Encoding of an environment observation as network input.

    Each observation value is scaled by its scale and split into ON and OFF channels.
    Inputs are ordered as ON channels, OFF channels, background.
    Each channel is encoded as a population of n_population Poisson neurons.
    """

    scales: tuple[float, ...] = CARTPOLE_SCALES
    n_population: int = 10 # neurons per channel
    rate_max: float = 200.0 # rate of a channel at maximum value (Hz)
    rate_background: float = 10.0 # rate of the background population (Hz)

    @property
    def n_channels(self) -> int:
        """Number of ON and OFF channels."""
        return 2 * len(self.scales)

    @property
    def n_in(self) -> int:
        """Number of network inputs."""
        return self.n_population * (self.n_channels + 1)


def encode_observation(
    key: jax.Array,
    obs: jax.Array,
    n_steps: int,
    encoder: ObservationEncoder,
    dt: float = 1.0
) -> jax.Array:
    """
    Network input for one observation.

    Parameters
    ----------
    key: jax.Array
        Random key for the Poisson inputs.
    obs: jax.Array
        Observation (n_obs,).
    n_steps: int
        Network time steps per decision step.
    encoder: ObservationEncoder
        Encoding configuration.
    dt: float
        Time step (ms).

    Returns
    -------
    jax.Array
        Network input (n_steps, n_in).
    """
    n_obs = len(encoder.scales)

    # Identify the channel grouping and observation index for each of the n_in inputs
    channel = jnp.arange(encoder.n_in) // encoder.n_population
    obs_index = channel % n_obs

    # Scale observations
    scaled_obs = obs / jnp.asarray(encoder.scales)

    # ON channels (channel < n_obs) multiply by 1.0, OFF channels multiply by -1.0.
    sign = jnp.where(channel < n_obs, 1.0, -1.0)
    channel_rate = encoder.rate_max * jnp.clip(sign * scaled_obs[obs_index], 0.0, 1.0)

    # Overwrite the rates for the background channels
    rates = jnp.where(
        channel >= 2 * n_obs,
        encoder.rate_background,
        channel_rate
    )

    return poisson_spikes(
        key,
        jnp.broadcast_to(rates, (n_steps, encoder.n_in)),
        dt
    )
