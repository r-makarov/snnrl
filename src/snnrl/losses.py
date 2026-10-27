import jax
import jax.numpy as jnp
import optax


def decision_cross_entropy(y: jax.Array, label: jax.Array, t_decision: int) -> jax.Array:
    """
    Softmax cross-entropy of the readouts against the label during the decision period.

    The readouts are the logits at every step of the decision period (the last t_decision steps);
    the loss is averaged over trials and these steps.

    Parameters
    ----------
    y: jax.Array
        Readouts, shape (batch, T, n_out).
    label: jax.Array
        Correct class of each trial, shape (batch,).
    t_decision: int
        Duration of the decision period (time steps).

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    logits = y[:, -t_decision:] # (batch, t_decision, n_out)
    labels = jnp.broadcast_to(label[:, None], logits.shape[:2])
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def decision_accuracy(y: jax.Array, label: jax.Array, t_decision: int) -> jax.Array:
    """
    Fraction of trials whose decision is correct.

    The decision is the class with the largest readout averaged over the decision period.

    Parameters
    ----------
    y: jax.Array
        Readouts, shape (batch, T, n_out).
    label: jax.Array
        Correct class of each trial, shape (batch,).
    t_decision: int
        Duration of the decision period (time steps).

    Returns
    -------
    jax.Array
        Scalar accuracy.
    """
    decision = jnp.argmax(y[:, -t_decision:].mean(axis=1), axis=-1)
    return jnp.mean(decision == label)


def rate_regularization(
    z: jax.Array, rate_target: float = 10.0, dt: float = 1.0
) -> jax.Array:
    """
    Squared deviation of each neuron's firing rate from a target, summed over neurons.

    Rate of each neuron is averaged over all trials and time steps
    (Bellec et al. 2020, Supplementary Note 2).

    Parameters
    ----------
    z: jax.Array
        Spikes (batch, T, n_rec).
    rate_target: float
        Target firing rate (Hz).
    dt: float
        Time step (ms).

    Returns
    -------
    jax.Array
        Scalar loss, with rates in spikes per ms.
    """
    # Average over all trials and time steps.
    rate = z.mean(axis=(0, 1)) # (n_rec,)
    rate_per_ms = rate / dt # convert per step -> per ms
    rate_target_per_ms = rate_target / 1000.0 # Hz -> per ms
    return jnp.sum((rate_per_ms - rate_target_per_ms) ** 2)
