import jax
import optax

from snnrl import network
from snnrl.losses import decision_accuracy, decision_cross_entropy, rate_regularization
from snnrl.tasks import CueAccumulation, sample_batch


def run_batch(
    x: jax.Array, weights: network.Weights, params: network.NetworkParams
) -> tuple[network.NetworkState, network.NetworkState, jax.Array]:
    """
    Run the network on a batch of input sequences (batch, T, n_in).
    """
    return jax.vmap(network.run, in_axes=(0, None, None))(x, weights, params)


def loss_fn(
    weights: network.Weights,
    x: jax.Array,
    label: jax.Array,
    params: network.NetworkParams,
    task: CueAccumulation,
    c_reg: float,
    rate_target: float,
) -> tuple[jax.Array, dict]:
    """
    Cross-entropy during the decision period with the firing-rate regularization.

    Returns
    -------
    tuple[jax.Array, dict]
        Scalar loss and metrics.
    """
    _, states, y = run_batch(x, weights, params)
    z = states.neuron.z
    cross_entropy = decision_cross_entropy(y, label, task.t_decision)
    rate_reg = rate_regularization(z, rate_target, params.neuron.dt)
    loss = cross_entropy + c_reg * rate_reg
    accuracy = decision_accuracy(y, label, task.t_decision)
    population_rate = z.mean() * 1000.0 / params.neuron.dt
    metrics = {
        "loss": loss,
        "cross_entropy": cross_entropy,
        "rate_reg": rate_reg,
        "accuracy": accuracy,
        "rate": population_rate,
    }
    return loss, metrics


def make_optimizer(learning_rate: float) -> optax.GradientTransformation:
    """Adam on w_in, w_rec and w_out; the readout bias b_out stays fixed."""
    labels = network.Weights(w_in="train", w_rec="train", w_out="train", b_out="fixed")
    return optax.multi_transform(
        {"train": optax.adam(learning_rate), "fixed": optax.set_to_zero()}, labels
    )


def make_train_step(params, task, optimizer, batch_size, c_reg, rate_target):
    """
    Jitted training step: sample a batch, compute the gradient, update the weights.

    The returned function has the signature
    train_step(weights, opt_state, key, n_cues) -> (weights, opt_state, metrics).
    """

    @jax.jit
    def train_step(weights, opt_state, key, n_cues):
        x, label = sample_batch(key, batch_size, n_cues, task)
        grads, metrics = jax.grad(loss_fn, has_aux=True)(
            weights, x, label, params, task, c_reg, rate_target
        )
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        return optax.apply_updates(weights, updates), opt_state, metrics

    return train_step


def make_evaluate(params, task, batch_size, c_reg, rate_target):
    """
    Jitted evaluation on a fresh batch, without updating the weights.

    The returned function has the signature
    evaluate(weights, key, n_cues) -> metrics.
    """

    @jax.jit
    def evaluate(weights, key, n_cues):
        x, label = sample_batch(key, batch_size, n_cues, task)
        _, metrics = loss_fn(weights, x, label, params, task, c_reg, rate_target)
        return metrics

    return evaluate


def train_cue_accumulation(
    key: jax.Array,
    weights: network.Weights,
    params: network.NetworkParams,
    task: CueAccumulation,
    curriculum: tuple[int, ...] = (1, 3, 5, 7),
    n_iter: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 5e-3,
    c_reg: float = 1.0,
    rate_target: float = 10.0,
    max_error: float = 0.08,
    validate_every: int = 10,
    n_confirm: int = 8,
    stop_at_last_stage: bool = True,
) -> tuple[network.Weights, list[dict]]:
    """
    Train the network on the cue accumulation task with BPTT and a curriculum.

    Every validate_every iterations, the error is measured on one fresh batch.
    If it is below max_error, it is measured again on n_confirm fresh batches;
    if their mean error is also below max_error, training moves to the next number of cues in the curriculum.
    At the last stage, training stops there if stop_at_last_stage is True, and otherwise continues for n_iter iterations.

    Parameters
    ----------
    key: jax.Array
        Random key.
    weights: network.Weights
        Initial weights.
    params: network.NetworkParams
        Fixed parameters of the network.
    task: CueAccumulation
        Task configuration.
    curriculum: tuple[int, ...]
        Numbers of cues, in the order they are trained.
    n_iter: int
        Maximum number of weight updates.
    batch_size: int
        Trials per weight update.
    learning_rate: float
        Optimizer learning rate.
    c_reg: float
        Weight of the firing-rate regularization.
    rate_target: float
        Target firing rate (Hz).
    max_error: float
        Error below which the network moves to the next curriculum stage.
    validate_every: int
        Iterations between validations.
    n_confirm: int
        Number of batches used to confirm a low validation error.
    stop_at_last_stage: bool
        Stop as soon as the last stage of the curriculum is passed.

    Returns
    -------
    tuple[network.Weights, list[dict]]
        Trained weights and training history.
    """
    optimizer = make_optimizer(learning_rate)
    opt_state = optimizer.init(weights)
    train_step = make_train_step(params, task, optimizer, batch_size, c_reg, rate_target)
    evaluate = make_evaluate(params, task, batch_size, c_reg, rate_target)

    stage = 0
    history = []
    for iteration in range(n_iter):
        key, key_train, key_val, key_confirm = jax.random.split(key, 4)
        n_cues = curriculum[stage]
        # Training
        weights, opt_state, metrics = train_step(weights, opt_state, key_train, n_cues)
        record = {"iteration": iteration, "n_cues": n_cues, **{k: float(v) for k, v in metrics.items()}}
        # Validation
        if iteration % validate_every == 0:
            metrics = evaluate(weights, key_val, n_cues)
            record["val_error"] = 1.0 - float(metrics["accuracy"])
            if record["val_error"] < max_error:
                keys = jax.random.split(key_confirm, n_confirm)
                errors = [1.0 - float(evaluate(weights, k, n_cues)["accuracy"]) for k in keys]
                record["confirm_error"] = sum(errors) / n_confirm
                # Next stage if the error is low enough
                if record["confirm_error"] < max_error:
                    if stage < len(curriculum) - 1:
                        stage += 1
                    elif stop_at_last_stage:
                        history.append(record)
                        break

        history.append(record)
    return weights, history
