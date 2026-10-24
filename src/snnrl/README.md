`neuron.py` describes a population of LIF and ALIF neurons following Bellec et al. (2020).

## Model

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

$$ A^t = v_{th} + \beta \, a_t$$

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
