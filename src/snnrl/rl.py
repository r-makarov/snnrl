from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct

from snnrl import network
from snnrl.losses import actor_loss, critic_loss, entropy_regularization, rate_regularization
from snnrl.inputs import ObservationEncoder, encode_observation


N_ACTIONS = 2


# ============================================================
# ROLLOUT
# ============================================================

@struct.dataclass
class Episode:
    """Experience of one episode, or a batch of episodes with a leading batch axis."""

    obs: jax.Array # observation at each step (max_steps, n_obs)
    x: jax.Array # network inputs (max_steps * n_steps, n_in)
    readout: jax.Array # readouts at the end of each decision window (max_steps, n_out)
    actions: jax.Array # shape (max_steps,)
    rewards: jax.Array # 0 after the episode ended (max_steps,)
    valid: jax.Array # decision steps before the episode ended (max_steps,)
    truncated: jax.Array # the episode was still running at the last decision step


@partial(jax.jit, static_argnames=("env", "encoder", "n_steps", "max_steps", "greedy"))
def rollout(
    key: jax.Array,
    weights: network.Weights,
    params: network.NetworkParams,
    env,
    env_params,
    encoder: ObservationEncoder,
    n_steps: int = 5,
    max_steps: int = 500,
    greedy: bool = False,
) -> Episode:
    """
    Run one episode in closed loop: observation -> network -> action -> environment.

    At each decision step the observation is encoded and held for n_steps network steps;
    the action is sampled from the softmax of the policy readouts at the last of these steps
    (or, if greedy, the action with the largest readout is taken).
    The episode is padded to max_steps decision steps.
    Steps after the episode ended are marked as not valid.

    Parameters
    ----------
    key: jax.Array
        Random key.
    weights: network.Weights
        Network weights.
    params: network.NetworkParams
        Fixed parameters of the network.
    env
        gymnax environment.
    env_params
        Parameters of the environment.
    encoder: ObservationEncoder
        Observation encoder configuration.
    n_steps: int
        Network time steps per decision step.
    max_steps: int
        Maximum number of decision steps.
    greedy: bool
        Take the action with the largest policy readout instead of sampling.

    Returns
    -------
    Episode
        Experience of the episode.
    """
    key_reset, key_steps = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)

    def decision_step(carry, step_key):
        obs, env_state, net_state, done = carry
        key_input, key_action, key_env = jax.random.split(step_key, 3)
        x = encode_observation(key_input, obs, n_steps, encoder)
        net_state, _, y = network.run(x, weights, params, net_state)
        logits = y[-1, :N_ACTIONS] # at the last time step, Value excluded
        action = jnp.argmax(logits) if greedy else jax.random.categorical(key_action, logits)
        next_obs, env_state, reward, step_done, _ = env.step(key_env, env_state, action, env_params)
        valid = ~done
        carry = (next_obs, env_state, net_state, done | step_done)
        return carry, (obs, x, y[-1], action, reward * valid, valid)

    init = (obs, env_state, network.init_state(params), jnp.array(False))

    _, (observations, x, readout, actions, rewards, valid) = jax.lax.scan(
        decision_step, init, jax.random.split(key_steps, max_steps)
    )

    return Episode(
        obs=observations,
        x=x.reshape(max_steps * n_steps, encoder.n_in),
        readout=readout,
        actions=actions,
        rewards=rewards,
        valid=valid,
        truncated=valid[-1],
    )


def rollout_batch(
    key, batch_size, weights, params, env, env_params, encoder, n_steps=5, max_steps=500, greedy=False
):
    """Run batch_size independent episodes in parallel."""
    keys = jax.random.split(key, batch_size)
    return jax.vmap(
        lambda k: rollout(k, weights, params, env, env_params, encoder, n_steps, max_steps, greedy)
    )(keys)


# ============================================================
# LOSS
# ============================================================

def discounted_returns(
    rewards: jax.Array, truncated: jax.Array, last_value: jax.Array, gamma: float
) -> jax.Array:
    """
    Discounted returns G_t = r_t + gamma * G_{t+1} of one episode.

    For an episode cut by the time limit, the return after the last step
    is estimated by the value at the last step.
    For an episode that ended, it is 0.

    Parameters
    ----------
    rewards: jax.Array
        Rewards, 0 after the episode ended (T,).
    truncated: jax.Array
        The episode was still running at the last step.
    last_value: jax.Array
        Value estimate at the last step.
    gamma: float
        Discount factor.

    Returns
    -------
    jax.Array
        Returns (T,).
    """

    def step_fn(next_return, reward):
        g = reward + gamma * next_return
        return g, g

    _, returns = jax.lax.scan(
        step_fn,
        jnp.where(truncated, last_value, 0.0),
        rewards,
        reverse=True
    )
    return returns



def split_readout(y: jax.Array, n_steps: int) -> tuple[jax.Array, jax.Array]:
    """
    Policy logits and value prediction from the readouts.
    """
    readout = y[..., n_steps - 1 :: n_steps, :] # last readout of each decision step
    policy_readout = readout[..., :N_ACTIONS]
    value_readout = readout[..., N_ACTIONS]
    return policy_readout, value_readout


def actor_critic_loss(
    weights: network.Weights,
    episodes: Episode,
    params: network.NetworkParams,
    n_steps: int = 5,
    gamma: float = 0.99,
    c_value: float = 0.5,
    c_entropy: float = 0.01,
    c_reg: float = 1.0,
    rate_target: float = 10.0,
) -> tuple[jax.Array, dict]:
    """
    Actor-critic loss of a batch of episodes, computed by running the network on the stored inputs.

        E = E_policy + c_value * E_value + c_entropy * E_entropy + c_reg * E_reg

    The value readout predicts the scaled return G_t * (1 - gamma).

    Parameters
    ----------
    weights: network.Weights
        Network weights.
    episodes: Episode
        Batch of episodes.
    params: network.NetworkParams
        Fixed parameters of the network.
    n_steps: int
        Network time steps per decision step.
    gamma: float
        Discount factor per decision step.
    c_value: float
        Coefficient of the value loss.
    c_entropy: float
        Coefficient of the entropy regularization.
    c_reg: float
        Coefficient of the firing-rate regularization.
    rate_target: float
        Target firing rate (Hz).

    Returns
    -------
    tuple[jax.Array, dict]
        Scalar loss and metrics.
    """
    # 1. Run the network on the stored inputs to get the states and readouts.
    _, states, y = jax.vmap(network.run, in_axes=(0, None, None))(episodes.x, weights, params)
    logits, value_scaled = split_readout(y, n_steps)
    value = value_scaled / (1 - gamma) # in units of returns

    # 2. Compute returns and advantages
    returns = jax.vmap(discounted_returns, in_axes=(0, 0, 0, None))(
        episodes.rewards, episodes.truncated, jax.lax.stop_gradient(value[:, -1]), gamma
    )
    returns_scaled = returns * (1 - gamma) # in units of value readout
    advantages = jax.lax.stop_gradient(returns - value)

    # 3. Compute the mask of valid decision steps
    mask = episodes.valid.astype(jnp.float32)

    # 4. Compute the loss
    policy_loss = actor_loss(logits, episodes.actions, advantages, mask)
    value_loss = critic_loss(value_scaled, returns_scaled, mask)
    entropy_reg = entropy_regularization(logits, mask)
    step_mask = jnp.repeat(mask, n_steps, axis=1) # (batch, max_steps * n_steps)
    rate_reg = rate_regularization(states.neuron.z, rate_target, params.neuron.dt, step_mask)

    loss = policy_loss + c_value * value_loss + c_entropy * entropy_reg + c_reg * rate_reg

    # 5. Compute the mean firing rate over valid steps and neurons
    valid_spikes = states.neuron.z * step_mask[..., None]
    n_valid = step_mask.sum() * states.neuron.z.shape[-1] # -1 is the neuron dimension
    mean_rate = valid_spikes.sum() / n_valid * 1000.0 / params.neuron.dt # Hz

    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": -entropy_reg, # mean policy entropy
        "rate_reg": rate_reg,
        "episode_return": episodes.rewards.sum(axis=1).mean(),
        "rate": mean_rate,
    }
    return loss, metrics


# ============================================================
# TRAINING
# ============================================================

def make_train_step(params, env, env_params, encoder, optimizer, batch_size, n_steps, max_steps, loss_kwargs):
    """
    Jitted training step.
    Run a batch of episodes, compute the actor-critic gradient, update the weights.

    The returned function has the signature
    train_step(weights, opt_state, key) -> (weights, opt_state, metrics).
    """

    @jax.jit
    def train_step(weights, opt_state, key):
        episodes = rollout_batch(key, batch_size, weights, params, env, env_params, encoder, n_steps, max_steps)
        grads, metrics = jax.grad(actor_critic_loss, has_aux=True)(
            weights, episodes, params, n_steps, **loss_kwargs
        )
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        return optax.apply_updates(weights, updates), opt_state, metrics

    return train_step


def train_actor_critic(
    key: jax.Array,
    weights: network.Weights,
    params: network.NetworkParams,
    env,
    env_params,
    encoder: ObservationEncoder,
    n_iter: int = 500,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    n_steps: int = 5,
    max_steps: int = 500,
    **loss_kwargs,
) -> tuple[network.Weights, list[dict]]:
    """
    Train the network in closed loop with the actor-critic loss
    (BPTT, or e-prop if params.neuron.stop_z_gradient is set).

    Parameters
    ----------
    key: jax.Array
        Random key.
    weights: network.Weights
        Initial weights (all of them are trained).
    params: network.NetworkParams
        Fixed parameters of the network.
    env, env_params
        gymnax environment and its parameters.
    encoder: ObservationEncoder
        Observation encoding.
    n_iter: int
        Number of weight updates.
    batch_size: int
        Episodes per weight update.
    learning_rate: float
        Adam learning rate.
    n_steps: int
        Network time steps per decision step.
    max_steps: int
        Maximum number of decision steps per episode.
    **loss_kwargs
        Arguments of actor_critic_loss (gamma, c_value, c_entropy, c_reg, rate_target).

    Returns
    -------
    tuple[network.Weights, list[dict]]
        Trained weights and training history.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)
    train_step = make_train_step(
        params, env, env_params, encoder, optimizer, batch_size, n_steps, max_steps, loss_kwargs
    )
    history = []
    for iteration in range(n_iter):
        key, step_key = jax.random.split(key)
        weights, opt_state, metrics = train_step(weights, opt_state, step_key)
        history.append({"iteration": iteration, **{k: float(v) for k, v in metrics.items()}})
    return weights, history
