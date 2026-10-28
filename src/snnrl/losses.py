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


def actor_loss(logits: jax.Array, actions: jax.Array, advantages: jax.Array, mask: jax.Array) -> jax.Array:
    """
    Advantage-weighted negative log-probability of the taken actions, mean over valid steps.

    Parameters
    ----------
    logits: jax.Array
        Policy logits, shape (batch, T, n_actions).
    actions: jax.Array
        Taken actions, shape (batch, T).
    advantages: jax.Array
        Advantage of each action, without gradient, shape (batch, T).
    mask: jax.Array
        Steps that count (1) or not (0), shape (batch, T).

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, actions) # -log pi(a_t), (batch, T)
    return (advantages * nll * mask).sum() / mask.sum()


def critic_loss(values: jax.Array, returns: jax.Array, mask: jax.Array) -> jax.Array:
    """
    Half squared error of the value predictions, mean over valid steps.

    Parameters
    ----------
    values: jax.Array
        Value prediction, shape (batch, T).
    returns: jax.Array
        Target returns, on the same scale as value, shape (batch, T).
    mask: jax.Array
        Steps that count (1) or not (0), shape (batch, T).

    Returns
    -------
    jax.Array
        Scalar loss.
    """
    return 0.5 * (((returns - values) ** 2) * mask).sum() / mask.sum()


def entropy_regularization(logits: jax.Array, mask: jax.Array) -> jax.Array:
    """
    Negative entropy of the policy, mean over valid steps.

    Minimizing it keeps the policy from becoming deterministic too early
    (E_H in Bellec et al. 2020, Supplementary Note 5).

    Parameters
    ----------
    logits: jax.Array
        Policy logits, shape (batch, T, n_actions).
    mask: jax.Array
        Steps that count (1) or not (0), shape (batch, T).

    Returns
    -------
    jax.Array
        Scalar loss: minus the mean entropy (nats).
    """
    log_probs = jax.nn.log_softmax(logits)
    entropy = -(jnp.exp(log_probs) * log_probs).sum(-1)
    return -(entropy * mask).sum() / mask.sum()


def rate_regularization(
    z: jax.Array, rate_target: float = 10.0, dt: float = 1.0, mask: jax.Array | None = None
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
    mask: jax.Array, optional
        Time steps that count (1) or not (0), shape (batch, T); e.g. padding after the end of an episode.
        Defaults to all steps.

    Returns
    -------
    jax.Array
        Scalar loss, with rates in spikes per ms.
    """
    if mask is None:
        mask = jnp.ones(z.shape[:2])
    # Average over all valid trials and time steps.
    rate = (z * mask[..., None]).sum(axis=(0, 1)) / mask.sum() # (n_rec,)
    rate_per_ms = rate / dt # convert per step -> per ms
    rate_target_per_ms = rate_target / 1000.0 # Hz -> per ms
    return jnp.sum((rate_per_ms - rate_target_per_ms) ** 2)
