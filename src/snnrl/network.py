import jax
import jax.numpy as jnp
from flax import struct

from snnrl.neuron import ALIFParams, ALIFState
from snnrl.neuron import init_state as init_neuron_state
from snnrl.neuron import step as neuron_step


# ============================================================
# PARAMETERS
# ============================================================

# -------------------------------------------------------------
# FIXED PARAMETERS
# -------------------------------------------------------------

@struct.dataclass
class NetworkParams:
    """Fixed parameters of the network."""

    neuron: ALIFParams # parameters of the recurrent population
    rec_mask: jax.Array # existing recurrent connections (1) or not (0), shape (n_rec, n_rec)
    tau_out: float = 20.0 # readout time constant (ms)

    @property
    def kappa(self) -> jax.Array:
        """Readout decay per time step."""
        return jnp.exp(-self.neuron.dt / self.tau_out)


def make_network_params(neuron: ALIFParams, **kwargs) -> NetworkParams:
    """
    Parameters of a fully connected recurrent network without self-connections.

    Parameters
    ----------
    neuron: ALIFParams
        Parameters of the recurrent population.
    **kwargs
        Other fields of NetworkParams.

    Returns
    -------
    NetworkParams
        Parameters of the network.
    """
    n_rec = neuron.beta.shape[0]
    rec_mask = 1.0 - jnp.eye(n_rec)
    return NetworkParams(neuron=neuron, rec_mask=rec_mask, **kwargs)


# -------------------------------------------------------------
# TRAINABLE PARAMETERS
# -------------------------------------------------------------

@struct.dataclass
class Weights:
    """Trainable weights, shapes (pre, post) so that the input is x @ w."""

    w_in: jax.Array # input -> recurrent, shape (n_in, n_rec)
    w_rec: jax.Array # recurrent -> recurrent, shape (n_rec, n_rec)
    w_out: jax.Array # recurrent -> readout, shape (n_rec, n_out)
    b_out: jax.Array # readout bias, shape (n_out,)


def init_weights(key: jax.Array, n_in: int, n_out: int, params: NetworkParams) -> Weights:
    """
    Initial weights following Bellec's cue accumulation code.

    w_in = N(0, 1/n_in) and w_rec = N(0, 1/n_rec), with missing connections (e.g. the diagonal) set to 0;
    w_out is Glorot-uniform (TensorFlow's default);
    b_out starts at 0.

    Parameters
    ----------
    key: jax.Array
        Random key.
    n_in: int
        Number of inputs.
    n_out: int
        Number of readout neurons.
    params: NetworkParams
        Parameters of the network.

    Returns
    -------
    Weights
        Initial weights.
    """
    n_rec = params.rec_mask.shape[0]
    key_in, key_rec, key_out = jax.random.split(key, 3)

    w_in = jax.random.normal(key_in, (n_in, n_rec)) / jnp.sqrt(n_in)

    w_rec = jax.random.normal(key_rec, (n_rec, n_rec)) / jnp.sqrt(n_rec) * params.rec_mask

    limit = jnp.sqrt(6 / (n_rec + n_out))
    w_out = jax.random.uniform(key_out, (n_rec, n_out), minval=-limit, maxval=limit)

    b_out = jnp.zeros(n_out)

    return Weights(w_in=w_in, w_rec=w_rec, w_out=w_out, b_out=b_out)


# ============================================================
# STATE
# ============================================================

@struct.dataclass
class NetworkState:
    neuron: ALIFState # state of the recurrent population
    z_filtered: jax.Array # spikes low-pass filtered with kappa, shape (n_rec,)


def init_state(params: NetworkParams) -> NetworkState:
    """Initial state: population at rest, no filtered spikes."""
    n_rec = params.rec_mask.shape[0]
    return NetworkState(neuron=init_neuron_state(n_rec), z_filtered=jnp.zeros(n_rec))


# ============================================================
# SIMULATION
# ============================================================

def step(
    state: NetworkState, 
    x: jax.Array, 
    weights: Weights, 
    params: NetworkParams
) -> tuple[NetworkState, jax.Array]:
    """
    Advance the simulation by one time step.

    Parameters
    ----------
    state: NetworkState
        State at time t.
    x: jax.Array
        Input x_{t+1}, shape (n_in,).
    weights: Weights
        Trainable weights.
    params: NetworkParams
        Fixed parameters of the network.

    Returns
    -------
    tuple[NetworkState, jax.Array]
        State at time t + 1 and readout y_{t+1}, shape (n_out,).
    """
    # 1. Compute the input current to the recurrent population
    # With e-prop, the recurrent input passes no gradient back to z_t
    z_rec = jax.lax.stop_gradient(state.neuron.z) if params.neuron.stop_z_gradient else state.neuron.z
    input_current = x @ weights.w_in + z_rec @ (weights.w_rec * params.rec_mask)

    # 2. Update the recurrent population
    neuron = neuron_step(state.neuron, input_current, params.neuron)

    # 3. Compute the filtered spikes
    z_filtered = params.kappa * state.z_filtered + (1 - params.kappa) * neuron.z

    # 4. Compute the readout
    y = z_filtered @ weights.w_out + weights.b_out

    return NetworkState(neuron=neuron, z_filtered=z_filtered), y


@jax.jit
def run(
    x: jax.Array,
    weights: Weights,
    params: NetworkParams,
    state: NetworkState | None = None
) -> tuple[NetworkState, NetworkState, jax.Array]:
    """
    Simulate the network driven by an input sequence.

    Parameters
    ----------
    x: jax.Array
        Inputs at each time step, shape (T, n_in).
    weights: Weights
        Trainable weights.
    params: NetworkParams
        Fixed parameters of the network.
    state: NetworkState, optional
        State before the first step. Defaults to the network at rest.

    Returns
    -------
    final_state: NetworkState
        State at time step T.
    states: NetworkState
        States at time steps 1..T, each field with a leading time axis.
    y: jax.Array
        Readouts at time steps 1..T, shape (T, n_out).
    """
    if state is None:
        state = init_state(params)

    def step_fn(state, xt):
        new_state, y = step(state, xt, weights, params)
        return new_state, (new_state, y)

    final_state, (states, y) = jax.lax.scan(step_fn, state, x)
    return final_state, states, y