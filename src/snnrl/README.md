## Neuron

`neuron.py` describes a population of LIF and ALIF neurons following Bellec et al. (2020).

LIF and ALIF neurons form one population that differs only in $\beta$. $\beta = 0$ gives a LIF neuron, whose threshold stays at $v_{th}$.

**Membrane potential**

measured relative to rest ($v_{rest} = 0$)

$$v_{t+1} = \alpha \, v_t + I_{t+1} - v_{th} \, z_t, \qquad \alpha = e^{-\Delta t / \tau_m}$$

- $v$ — membrane potential (1)
- $I_{t+1}$ — input current
- $\alpha$ — membrane decay per time step, $\tau_m$ — membrane time constant
- $- v_{th} \, z_t$ — reset by the baseline threshold after each spike



**Threshold adaptation**

$$ a_{t+1} = \rho \, a_t + z_t, \qquad \rho = e^{-\Delta t / \tau_a}$$

- $a$ — threshold adaptation variable
- $\rho$ — adaptation decay per time step, $\tau_a$ — adaptation time constant

**Adaptive threshold**

$$ A_t = v_{th} + \beta \, a_t$$

- $v_{th}$ — baseline threshold
- $\beta$ — adaptation strength

**Spikes**

$$ z_t = H\left(v_t - A_t\right)$$

- $z$ — spike output (0 or 1)
- $H$ — Heaviside step function
- Refractoriness: after a spike, $z$ is forced to 0 for the next $n_{ref}$ steps

**Pseudo-derivative** (used in place of $dz/dv$ by BPTT and e-prop)

$$ \psi_t = \frac{\gamma}{v_{th}} \max\left(0, \; 1 - \left| \frac{v_t - A_t}{v_{th}} \right| \right)$$

- $\gamma$ — dampening factor
- $\psi = 0$ while the neuron is refractory

## Network

`network.py` connects an input layer, one recurrent layer of the neurons above, and a readout layer of leaky non-spiking neurons. 

Synapses are instantaneous (delta) current synapses: a synapse is a weight, and a presynaptic spike adds its weight to the postsynaptic input current.

Indices: $i$ — input, $j$ — recurrent neuron, $k$ — readout neuron.

**Input layer**

$$x_t \in \{0, 1\}^{n_{in}}$$

- $x_t$ — activity of the input layer (e.g. Poisson spike trains)

**Recurrent layer**

$$I_{j,t+1} = \sum_i W^{in}_{ij} \, x_{i,t+1} + \sum_k W^{rec}_{kj} \, M_{kj} \, z_{k,t}$$

- $I_{j,t+1}$ — input current of recurrent neuron $j$, the $I_{t+1}$ of the neuron model
- $W^{in}$ — input weights, stored as (pre, post).
- $W^{rec}$ — recurrent weights, stored as (pre, post).
- $M$ — connectivity mask: 1 for existing connections, 0 for missing ones (no self-connections)
- $z_{k,t}$ — spikes of the recurrent population (with a delay of one time step)

**Readout layer**

$$\bar z_{j,t} = \kappa \, \bar z_{j,t-1} + (1 - \kappa) \, z_{j,t}, \qquad \kappa = e^{-\Delta t / \tau_{out}}$$

- $\bar z$ — low-pass filtered spikes (`z_filtered`), approximately the instantaneous firing rate in
  spikes per time step
- $\kappa$ — readout decay per time step, $\tau_{out}$ — readout time constant

$$y_{k,t} = \sum_j W^{out}_{jk} \, \bar z_{j,t} + b_k$$

- $y_k$ — output of readout neuron $k$, a leaky integrator of the recurrent spikes. Filtering the
  spikes before weighting them gives the same $y$ and keeps the bias out of the leaky sum
- $W^{out}$, $b$ — readout weights and bias