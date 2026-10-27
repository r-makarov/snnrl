from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from snnrl.inputs import poisson_spikes


@dataclass(frozen=True)
class CueAccumulation:
    """
    Cue accumulation task (Bellec et al. 2020, Fig. 3).

    A trial has a cue period, a delay and a decision period.
    In the cue period, n_cues cues are shown one after another,
    each on the left or the right.
    In the decision period, the network has to report the side
    that got the majority of the cues.

    Inputs come from 4 populations of n_population Poisson neurons: 
    - left cues
    - right cues
    - decision period
    - background

    In each trial one side is favoured: each cue appears on it with probability 1 - p_group.

    All trials have the same array length, set by max_cues and max_delay. 
    Shorter trials start later, with silent input before them,
    so that the decision period is always the last t_decision steps.

    The fields set array shapes, so the task is passed to jitted functions as a static argument.
    """

    n_population: int = 10 # Poisson neurons per input population
    max_cues: int = 7 # largest number of cues in a trial
    t_cue: int = 100 # duration of a cue (ms)
    t_cue_spacing: int = 150 # time between the cue onsets (ms); 50 ms gap, 100 ms cue.
    min_delay: int = 1050 # shortest delay between the last cue and the decision period (ms)
    max_delay: int = 1050 # longest delay (ms)
    t_decision: int = 150 # duration of the decision period (ms)
    rate_input: float = 40.0 # rate of cue and decision populations while active (Hz)
    rate_background: float = 10.0 # rate of the background population (Hz)
    p_group: float = 0.3 # probability of a cue on the side not favoured in the trial

    @property
    def n_in(self) -> int:
        """Number of inputs."""
        return 4 * self.n_population

    @property
    def trial_length(self) -> int:
        """Number of time steps of every trial (dt = 1 ms)."""
        return self.max_cues * self.t_cue_spacing + self.max_delay + self.t_decision


@partial(jax.jit, static_argnames="task")
def sample_trial(
    key: jax.Array, 
    n_cues: int, 
    task: CueAccumulation
) -> tuple[jax.Array, jax.Array]:
    """
    Sample one trial of the cue accumulation task.

    Parameters
    ----------
    key: jax.Array
        Random key.
    n_cues: int
        Number of cues in the trial, odd and at most task.max_cues.
    task: CueAccumulation
        Task configuration.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Input spikes, shape (T, n_in), with inputs ordered as: left, right, decision and background;
        and the label, 0 (left) or 1 (right).
    """
    # 1. Split the key
    key_side, key_cues, key_delay, key_spikes = jax.random.split(key, 4)

    # 2. Build the sequence of cues.
    # Pick the favored side for this trial.
    p_right = jnp.where(jax.random.bernoulli(key_side), 1 - task.p_group, task.p_group)
    # Sample sides for all max_cues cues [T, F, ...]; T = right.
    right = jax.random.bernoulli(key_cues, p_right, (task.max_cues,))
    # Build a mask for the first n_cues cues; [T, T, T, F, F, F, F]
    shown = jnp.arange(task.max_cues) < n_cues

    # 3. Select the label.
    # If more than a half are right the label is 1 (right).
    label = (jnp.sum(right & shown) > n_cues / 2).astype(jnp.int32)

    # 4. Sample a delay and compute the start of the trial
    T = task.trial_length
    # Sample a delay between the last cue and the decision period (defaults to 1050 ms)
    delay = jax.random.randint(key_delay, (), task.min_delay, task.max_delay + 1)
    # The trial starts so that the decision period is always the last t_decision steps
    start = T - (n_cues * task.t_cue_spacing + delay + task.t_decision)

    # 5. Compute cue onsets
    t = jnp.arange(T)[:, None] # (T, 1)
    # Cue k is on during [onset_k, onset_k + t_cue), after a silent gap of t_cue_spacing - t_cue
    onsets = start + jnp.arange(task.max_cues) * task.t_cue_spacing + (task.t_cue_spacing - task.t_cue) # (max_cues,)

    # 6. Compute which cues are on at each time step
    cue_on = shown & (t >= onsets) & (t < onsets + task.t_cue) # (T, max_cues)
    left_on = jnp.any(cue_on & ~right, axis=1, keepdims=True) # (T, 1)
    right_on = jnp.any(cue_on & right, axis=1, keepdims=True) # (T, 1)
    decision_on = (t >= T - task.t_decision) # (T, 1)
    background_on = (t >= start) # (T, 1)

    # 7. Compute the rate matrix (T, n_in) as an outer product.
    # population is [0, 0, ..., 1, 1, ..., 2, 2, ..., 3, 3, ...]
    population = jnp.arange(task.n_in) // task.n_population
    rates = (
        task.rate_input * left_on * (population == 0)
        + task.rate_input * right_on * (population == 1)
        + task.rate_input * decision_on * (population == 2)
        + task.rate_background * background_on * (population == 3)
    ) # (T, n_in)
    
    # Note: check the issue with stack/concatenate in XLA (jax 0.10.2)

    return poisson_spikes(key_spikes, rates), label


@partial(jax.jit, static_argnames=("batch_size", "task"))
def sample_batch(
    key: jax.Array,
    batch_size: int,
    n_cues: int,
    task: CueAccumulation
) -> tuple[jax.Array, jax.Array]:
    """
    Sample a batch of trials with the same number of cues.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Input spikes, shape (batch_size, T, n_in), and labels, shape (batch_size,).
    """
    keys = jax.random.split(key, batch_size)
    return jax.vmap(lambda key: sample_trial(key, n_cues, task))(keys)
