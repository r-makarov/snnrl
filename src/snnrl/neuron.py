import jax
import jax.numpy as jnp
from flax import struct


# ============================================================
# PARAMETERS
# ============================================================

@struct.dataclass
class ALIFParams:
    beta: jax.Array # adaptation strength per neuron, shape (n,); 0 for LIF neurons
    tau_m: float = 20.0 # membrane time constant (ms)
    tau_a: float = 2000.0 # adaptation time constant (ms)
    v_th: float = 0.6 # baseline threshold
    gamma: float = 0.3 # dampening factor of the pseudo-derivative
    n_ref: int = 5 # refractory period (time steps)
    dt: float = 1.0 # time step (ms)
    # e-prop: no gradient through the spikes z_t where they enter
    # the next step's reset and recurrent input 
    # (the gradient through the adaptation is kept).
    stop_z_gradient: bool = struct.field(pytree_node=False, default=False)

    @property
    def alpha(self) -> jax.Array:
        """Membrane decay per time step."""
        return jnp.exp(-self.dt / self.tau_m)

    @property
    def rho(self) -> jax.Array:
        """Adaptation decay per time step."""
        return jnp.exp(-self.dt / self.tau_a)

    @property
    def is_alif(self) -> jax.Array:
        """True for ALIF neurons (beta > 0), shape (n,)."""
        return self.beta > 0


def make_params(n_lif: int, n_alif: int, beta: float = 0.0174, **kwargs) -> ALIFParams:
    """
    Parameters for a population of n_lif LIF neurons followed by n_alif ALIF neurons.

    Parameters
    ----------
    n_lif: int
        Number of LIF neurons (beta = 0).
    n_alif: int
        Number of ALIF neurons.
    beta: float
        Adaptation strength of the ALIF neurons. The default is Bellec's value for the
        cue accumulation task.
    
    Returns
    -------
    ALIFParams
        Parameters of the population.
    """
    betas = jnp.concatenate([jnp.zeros(n_lif), jnp.full(n_alif, beta)])
    return ALIFParams(beta=betas, **kwargs)


# ============================================================
# STATE
# ============================================================

@struct.dataclass
class ALIFState:
    v: jax.Array # membrane potential, shape (n,)
    a: jax.Array # threshold adaptation variable, shape (n,)
    z: jax.Array # spikes (0.0 or 1.0), shape (n,)
    r: jax.Array # remaining refractory steps, shape (n,)


def init_state(n: int) -> ALIFState:
    """
    Initial state of the population: at rest, no adaptation, no spikes, not refractory.

    Parameters
    ----------
    n: int
        Number of neurons in the population.

    Returns
    -------
    ALIFState
        Initial state, each field of shape (n,).
    """
    return ALIFState(
        v=jnp.zeros(n),
        a=jnp.zeros(n),
        z=jnp.zeros(n),
        r=jnp.zeros(n, dtype=jnp.int32),
    )


# ============================================================
# SIMULATION
# ============================================================

@jax.custom_jvp
def spike(v_scaled: jax.Array, gamma: float) -> jax.Array:
    """
    Heaviside step function with a triangular pseudo-derivative.

    Forward: z = H(v_scaled). 
    Backward: dz/dv_scaled = gamma * max(0, 1 - |v_scaled|).

    Parameters
    ----------
    v_scaled: jax.Array
        Distance to the threshold in units of the baseline threshold, (v - A) / v_th.
    gamma: float
        Dampening factor of the pseudo-derivative.

    Returns
    -------
    jax.Array
        Spikes (0.0 or 1.0), same shape as v_scaled.
    """
    return (v_scaled > 0).astype(v_scaled.dtype)


@spike.defjvp
def _spike_jvp(primals, tangents):
    """
    Custom Jacobian-vector product for the spike function.

    Parameters
    ----------
    primals: tuple[jax.Array, float]
        The primal inputs to the spike function: (v_scaled, gamma).
    tangents: tuple[jax.Array, float]
        The tangent inputs to the spike function: (v_scaled_dot, gamma_dot).

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The primal output (spikes) and the tangent output (spike gradients).
    """
    # The input values
    v_scaled, gamma = primals
    # A small change in each input
    v_scaled_dot, _ = tangents # gamma is a constant, its tangent is ignored
    # Compute z and z_dot
    z = spike(v_scaled, gamma)
    pseudo_derivative = gamma * jnp.maximum(0.0, 1.0 - jnp.abs(v_scaled))
    z_dot = pseudo_derivative * v_scaled_dot
    return z, z_dot


def threshold(a: jax.Array, params: ALIFParams) -> jax.Array:
    """Adaptive threshold A = v_th + beta * a, shape (n,)."""
    return params.v_th + params.beta * a


def step(state: ALIFState, input_current: jax.Array, params: ALIFParams) -> ALIFState:
    """
    Advance simulation by one time step.

    Parameters
    ----------
    state: ALIFState
        State at time t.
    input_current: jax.Array
        Input current I_{t+1}, shape (n,).
    params: ALIFParams
        Parameters of the population.

    Returns
    -------
    ALIFState
        State at time t + 1.
    """
    # 1. Update the membrane potential and adaptation variable
    # With e-prop, the reset passes no gradient back to z_t
    z_reset = jax.lax.stop_gradient(state.z) if params.stop_z_gradient else state.z
    v = params.alpha * state.v + input_current - params.v_th * z_reset
    a = params.rho * state.a + state.z

    # 2. Update the refractory counter (`where` over neurons)
    # For neurons that spiked, set the refractory counter to n_ref.
    # For neurons that did not spike, decrement the counter by 1, but not below 0.
    r = jnp.where(state.z > 0, params.n_ref, jnp.maximum(state.r - 1, 0))

    # 3 Compute the distance to the new threshold A 
    # normalized by the baseline threshold v_th
    v_scaled = (v - threshold(a, params)) / params.v_th
    
    # 4. Compute new spikes (`where` over neurons)
    # Outside the refractory period the gradient flows through `spike`
    # inside it it is 0
    z = jnp.where(r > 0, 0.0, spike(v_scaled, params.gamma))

    return ALIFState(v=v, a=a, z=z, r=r)


@jax.jit
def run(input_current: jax.Array, 
        params: ALIFParams, 
        state: ALIFState | None = None) -> tuple[ALIFState, ALIFState]:
    """
    Simulate the population driven by given input currents.

    Parameters
    ----------
    input_current: jax.Array
        Input current at each time step, shape (T, n).
    params: ALIFParams
        Parameters of the population.
    state: ALIFState, optional
        State before the first step. Defaults to the population at rest.

    Returns
    -------
    final_state: ALIFState
        State at time step T.
    states: ALIFState
        States at time steps 1..T, each field of shape (T, n).
    """
    if state is None:
        state = init_state(input_current.shape[1])

    def step_fn(state, current):
        new_state = step(state, current, params)
        return new_state, new_state

    final_state, states = jax.lax.scan(step_fn, state, input_current)
    return final_state, states
