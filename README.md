# snnrl

Recurrent spiking neural networks trained in closed loop on reinforcement learning tasks, in [JAX](https://github.com/jax-ml/jax). The network is the RSNN of Bellec et al. (2020): leaky integrate-and-fire (LIF) and adaptive (ALIF) neurons with leaky readouts, trained with backpropagation through time (BPTT) or e-prop. Written with [Optax](https://github.com/google-deepmind/optax), with environments from [gymnax](https://github.com/RobertTLange/gymnax).

This is a pilot project and work in progress.

## Interactive page

👉 [r-makarov.github.io/snnrl](https://r-makarov.github.io/snnrl/), a network trained with e-prop balancing a pole. Episode explorer built with Bokeh.

## Repository layout

Each step is a short module in `src/snnrl/` with a notebook in `notebooks/` that validates it. The model and learning equations are in [src/snnrl/README.md](src/snnrl/README.md).

```
src/snnrl/
├── neuron.py      # LIF/ALIF neurons, pseudo-derivative
├── network.py     # input, recurrent and readout layers
├── inputs.py      # Poisson spikes, observation encoding
├── tasks.py       # cue accumulation task
├── losses.py      # cross-entropy, actor-critic losses, regularization
├── supervised.py  # training on cue accumulation
├── rl.py          # closed-loop rollouts, actor-critic training
└── utils.py       # saving and loading results
notebooks/         # one notebook per step, trained weights in results/
page/              # scripts that build the interactive page
data/              # episodes shown on the page
docs/              # the page (GitHub Pages)
```

## Steps

| Step | Notebook | Status |
|---|---|---|
| LIF/ALIF neurons | [01_neuron.ipynb](notebooks/01_neuron.ipynb) | ✅ |
| Recurrent network | [02_network.ipynb](notebooks/02_network.ipynb) | ✅ |
| Cue accumulation task | [03_cue_accumulation.ipynb](notebooks/03_cue_accumulation.ipynb) | ✅ |
| Cue accumulation with BPTT | [04_bptt.ipynb](notebooks/04_bptt.ipynb) | ✅ |
| Cue accumulation with e-prop, compared with BPTT | [05_eprop.ipynb](notebooks/05_eprop.ipynb) | ✅ |
| CartPole (Barto et al., 1983) in closed loop, actor-critic with BPTT and e-prop | [06_cartpole.ipynb](notebooks/06_cartpole.ipynb) | ✅ |
| Foraging task (Zeraati et al., 2025) | | Planned |
| Spatially embedded network with a diffusing learning signal (Barretto-Bittar et al., 2026) | | Planned |
| Two-compartment neurons (soma and dendrite) | | Planned |

## Installation

```
conda env create -f environment.yaml
conda activate snnrl
pip install -e .
```

## References

1. Bellec G., Scherr F., Subramoney A., Hajek E., Salaj D., Legenstein R., Maass W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications* 11:3625. [doi:10.1038/s41467-020-17236-y](https://doi.org/10.1038/s41467-020-17236-y)
2. Barto A. G., Sutton R. S., Anderson C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics* SMC-13:834–846. [doi:10.1109/TSMC.1983.6313077](https://doi.org/10.1109/TSMC.1983.6313077)
3. Zeraati R., Oña Jodar T., Safavi S., Cruz B., Poo C., Dayan P. (2025). Optimal foraging by learning the world model. *Proceedings of the Conference on Cognitive Computational Neuroscience (CCN)*.
4. Barretto-Bittar J., Levina A., Giannakakis E., Zeraati R. (2026). Diffusion of neuromodulators for temporal credit assignment. *Proceedings of the National Academy of Sciences* 123(38):e2608831123. [doi:10.1073/pnas.2608831123](https://doi.org/10.1073/pnas.2608831123)
