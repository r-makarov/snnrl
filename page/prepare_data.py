import json
from pathlib import Path

import gymnax
import jax
import jax.numpy as jnp
import numpy as np

from snnrl import network
from snnrl.inputs import ObservationEncoder
from snnrl.network import init_weights, make_network_params
from snnrl.neuron import make_params
from snnrl.rl import rollout, split_readout
from snnrl.utils import load_results

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "results"
WEIGHTS = RESULTS / "06_cartpole_eprop.npz"
OUTPUT = ROOT / "data" / "episode_player.json"


RATE_MAX = 200.0 # Hz, must match the training of WEIGHTS
N_STEPS = 5 # network time steps per decision step
GAMMA = 0.99
EPISODE_SEEDS = (42, 1, 0) # hand-picked seeds for the three networks, to get different episode lengths
GRID_COLUMNS = 10 # neurons of each type are shown as a block of rows x GRID_COLUMNS
BLOCK_GAP = 0.4 # vertical gap between the LIF and the ALIF block


# ======================================================================
# EPISODE
# ======================================================================

def run_episode(episode_key, weights, params, env, env_params, encoder):
    """One episode in closed loop, the input spikes and the network activity during it."""
    ep = rollout(episode_key, weights, params, env, env_params, encoder, N_STEPS)
    length = int(ep.valid.sum())
    _, states, y = network.run(ep.x, weights, params)

    z = np.asarray(states.neuron.z[: length * N_STEPS])
    rate = np.asarray(states.z_filtered[N_STEPS - 1 :: N_STEPS][:length]) * 1000.0 # Hz, end of each decision
    x = np.asarray(ep.x[: length * N_STEPS]) # input spikes (length * N_STEPS, n_in)

    # Spike count of each input population (ON, OFF, bg) in each decision window
    population = np.arange(encoder.n_in) // encoder.n_population
    counts = x.reshape(length, N_STEPS, encoder.n_in).sum(axis=1) # (length, n_in)
    input_counts = np.stack([counts[:, population == k].sum(axis=1) for k in range(population.max() + 1)], axis=1)

    logits, value_scaled = split_readout(y[: length * N_STEPS], N_STEPS)
    spike_times, spike_neurons = np.nonzero(z)
    input_times, input_ids = np.nonzero(x)
    return {
        "length": length,
        "cart_x": np.asarray(ep.obs[:length, 0]),
        "pole_angle": np.asarray(ep.obs[:length, 2]),
        "input_counts": input_counts,
        "rate": rate,
        "p_right": np.asarray(jax.nn.softmax(logits, axis=-1)[:, 1]),
        "action": np.asarray(ep.actions[:length]), # 0: push left, 1: push right
        "value": np.asarray(value_scaled) / (1 - GAMMA),
        "spike_times": spike_times,
        "spike_neurons": spike_neurons,
        "input_times": input_times,
        "input_ids": input_ids,
    }


# ======================================================================
# LAYOUT
# ======================================================================

def correlation_with(rate, signal):
    """Correlation of each neuron's rate (decision steps x neurons) with a signal, 0 for silent neurons."""
    centered = rate - rate.mean(axis=0)
    s = signal - signal.mean()
    norm = np.sqrt((centered**2).sum(axis=0) * (s**2).sum())
    return np.where(norm > 0, (centered * s[:, None]).sum(axis=0) / np.where(norm > 0, norm, 1.0), 0.0)


def neuron_positions(correlation, is_alif):
    """
    Recurrent neurons on a grid:
    LIF in a block on top, ALIF in a block below.
    In each block, columns go from left to right by correlation with P(right),
    so that "push-left" neurons are on the left.
    """
    x, y = np.zeros(len(is_alif)), np.zeros(len(is_alif))
    for alif in (False, True):
        neurons = np.flatnonzero(is_alif == alif)
        order = neurons[np.argsort(correlation[neurons])]
        n_rows = int(np.ceil(len(order) / GRID_COLUMNS))
        rank = np.arange(len(order))
        top = (n_rows - 1) if alif else (2 * n_rows - 1 + BLOCK_GAP)
        x[order] = rank // n_rows
        y[order] = top - rank % n_rows
    return x, y


def side_nodes(n_channels, height, gap=2.0):
    """
    Input populations in a row above the grid and readouts in a row below it.
    Inputs (ON, OFF, background):
    OFF channels on the left, background in the middle, ON channels on the right.
    Readouts (push left, push right, value):
    push left on the left, value in the middle, push right on the right.
    """
    slots = np.linspace(0, GRID_COLUMNS - 1, n_channels + 1)
    half = n_channels // 2
    input_x = np.concatenate([slots[half + 1 :], slots[:half], slots[half : half + 1]])
    right = GRID_COLUMNS - 1
    return {
        "input_nodes": {"x": input_x, "y": np.full(n_channels + 1, height + gap)},
        "readout_nodes": {"x": np.array([0.0, right, right / 2]), "y": np.full(3, -gap)},
    }


# ======================================================================
# MAIN
# ======================================================================

def to_json(value):
    """Convert NumPy arrays and scalars for json.dumps."""
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        array = np.asarray(value)
        return np.round(array.astype(float), 4).tolist() if array.dtype.kind == "f" else array.tolist()
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    env, env_params = gymnax.make("CartPole-v1")
    encoder = ObservationEncoder(rate_max=RATE_MAX)
    params = make_network_params(make_params(n_lif=50, n_alif=50, stop_z_gradient=True))
    is_alif = np.asarray(params.neuron.is_alif)
    networks = {
        "trained_42": load_results(WEIGHTS)[0],
        "trained_1": load_results(WEIGHTS)[0],
        "untrained_0": init_weights(jax.random.key(0), encoder.n_in, 3, params),
    }

    data = {"n_population": encoder.n_population}
    for (name, weights), seed in zip(networks.items(), EPISODE_SEEDS):
        episode_key = jax.random.key(seed)
        episode = run_episode(episode_key, weights, params, env, env_params, encoder)
        correlation = correlation_with(episode["rate"], episode["p_right"])
        x, y = neuron_positions(correlation, is_alif)
        data[name] = {
            "episode": episode,
            "neurons": {"x": x, "y": y, "alif": is_alif, "correlation": correlation},
            **side_nodes(encoder.n_channels, y.max()),
        }
        print(f"{name}: episode length {episode['length']}, {len(episode['spike_times'])} network spikes, "
              f"{len(episode['input_times'])} input spikes, mean rate {episode['rate'].mean():.1f} Hz")

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(to_json(data)))
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
