import json
from pathlib import Path

import numpy as np
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, CustomJS, Div, FixedTicker, HoverTool, Range1d, Select, Slider,
                          Span, Toggle)
from bokeh.palettes import Inferno256
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.transform import linear_cmap

# Paths
ROOT = Path(__file__).resolve().parents[1]
EPISODE_DATA = ROOT / "data" / "episode_player.json"
OUTPUT = ROOT / "docs" / "index.html"

# Page layout
PAGE_WIDTH = 1140
LEFT_WIDTH = 520
RIGHT_WIDTH = PAGE_WIDTH - LEFT_WIDTH

# Labels
NETWORKS = {"trained_42": "trained (seed 42)", "trained_1": "trained (seed 1)", "untrained_0": "untrained (seed 0)"}
INPUT_LABELS = ["x+", "v+", "θ+", "ω+", "x-", "v-", "θ-", "ω-", "bg"]
READOUT_LABELS = ["push left", "push right", "value"]
SPEEDS = {"160": "0.25x", "80": "0.5x", "40": "1x"} # ms between frames

# Colors
LIF_COLOR = "royalblue"
ALIF_COLOR = "darkorange"
ACTION_COLOR = "forestgreen"
OBSERVATION_COLORS = ["darkorchid", "violet", "seagreen", "yellowgreen"]
BACKGROUND_COLOR = "gray"
RATE_MAX = 80.0 # Hz, top of the color scale of the neurons
COUNT_MAX = 8 # spikes per decision step, top of the color scale of the input populations
VALUE_MAX = 100.0 # top of the color scale of the value readout, 1 / (1 - gamma)

NODE_SIZE = 20
LABEL_MARGIN = 1.4 # room for the input and readout labels in the network panel (data units)
POLE_LENGTH = 1.0 # drawing length (CartPole's pole has half-length 0.5)
CRITICAL_ANGLE = 12 * 2 * np.pi / 360 # CartPole ends the episode when the pole leans further
CART_HALF_WIDTH = 0.25
ARROW_LENGTH = 0.45
N_STEPS = 5 # network time steps (ms) per decision step


# ======================================================================
# TEXTS
# ======================================================================

TITLE = """
<h2>Recurrent spiking network balancing a pole</h2>
"""

TEXT = """

<p style="margin-bottom: 10px">
GitHub repository: <a href="https://github.com/r-makarov/snnrl" target="_blank">https://github.com/r-makarov/snnrl</a>
</p>

<p style="margin-bottom: 10px">
<b>About:</b>
A recurrent network of 50 leaky integrate-and-fire (LIF) and 50 adaptive (ALIF) spiking neurons (Bellec et al., 2020) controls CartPole (Barto et al., 1983) in closed loop.
At every decision step the observation enters the network as Poisson spike trains:
each ON and OFF channel of cart position x, cart velocity v, pole angle θ and pole angular velocity ω is a population of 10 input neurons whose rate grows with the channel's value (up to 200 Hz), and a background population (bg) fires at 10 Hz.
After 5 ms of network time, three leaky readout neurons give the policy (pushing the cart left or right) and the value of the state.
The network was implemented from scratch in JAX, and its neuron model was checked numerically against the independent JAX implementation of Barretto-Bittar et al. (2026).
It was trained by reward with e-prop (Bellec et al., 2020), an approximation of backpropagation through time (BPTT) that is local in time.
</p>

<p style="margin-bottom: 10px">
<b>How to use it:</b>
Move the slider, step with the arrow buttons or press Play.
In the network panel, activity flows from top to bottom.
The squares at the top are the input populations, colored by their number of spikes in the current 5 ms.
The upper raster shows the individual input spikes.
Each circle is a recurrent neuron, colored by its firing rate (LIF neurons in the upper block, ALIF neurons in the lower block, ordered from left to right by how their activity correlates with pushing right).
The squares at the bottom are the readouts, colored by the probability of pushing left or right and by the value.
On the cart, the arrows show the probability of each action and, in green, the action taken.
The dashed lines mark the angle at which the episode ends.
Compare the episodes: the trained network usually balances the pole for the full 500 steps, but not always (episode seed 1), while the untrained network loses it within a few steps.
</p>
"""

REFERENCES = """
<h3>References</h3>
<ol>
<li>
    Bellec G., Scherr F., Subramoney A., Hajek E., Salaj D., Legenstein R., Maass W. (2020).
    A solution to the learning dilemma for recurrent networks of spiking neurons. <i>Nature Communications</i> 11:3625.
    <a href="https://doi.org/10.1038/s41467-020-17236-y" target="_blank">doi:10.1038/s41467-020-17236-y</a>
</li>
<li>
    Barto A. G., Sutton R. S., Anderson C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems.
    <i>IEEE Transactions on Systems, Man, and Cybernetics</i> SMC-13:834–846.
    <a href="https://doi.org/10.1109/TSMC.1983.6313077" target="_blank">doi:10.1109/TSMC.1983.6313077</a>
</li>
<li>
    Barretto-Bittar J., Levina A., Giannakakis E., Zeraati R. (2026). Diffusion of neuromodulators for temporal credit assignment.
    <i>Proceedings of the National Academy of Sciences</i> 123(38):e2608831123.
    <a href="https://doi.org/10.1073/pnas.2608831123" target="_blank">doi:10.1073/pnas.2608831123</a>
</li>
</ol>
"""


# ======================================================================
# CALLBACKS
# ======================================================================

# Show decision step t of the current episode in all views
FRAME_CALLBACK = """
const e = episodes[select.value]
const t = Math.min(slider.value, e.length - 1)
const x = e.cart_x[t], angle = e.pole_angle[t], p = e.p_right[t], a = e.action[t]
cart.data = {x: [x], y: [0]}
pole.data = {x0: [x], y0: [0], x1: [x + pole_length * Math.sin(angle)], y1: [pole_length * Math.cos(angle)]}
const reach = [-1, 1].map(s => x + s * pole_length * Math.sin(critical_angle))
limits.data = {...limits.data, x0: [x, x], x1: reach}
const start = [x - half_width, x + half_width], end = [x - half_width - arrow_length, x + half_width + arrow_length]
arrows.data = {...arrows.data, x0: start, x1: end, alpha: [0.15 + 0.85 * (1 - p), 0.15 + 0.85 * p],
               color: [a === 0 ? action_color : 'gray', a === 1 ? action_color : 'gray']}
neurons.data = {...neurons.data, rate: e.rate[t]}
inputs.data = {...inputs.data, value: e.input_counts[t]}
readouts.data = {...readouts.data, value: [1 - p, p, e.value[t] / value_max]}
raster_cursor.location = t * n_steps
input_cursor.location = t * n_steps
status.text = `step ${t + 1} of ${e.length} &nbsp;|&nbsp; P(right) = ${p.toFixed(2)}`
    + ` &nbsp;|&nbsp; action: ${a === 1 ? 'push right' : 'push left'} &nbsp;|&nbsp; value = ${e.value[t].toFixed(0)}`
"""

# Switch between the trained and the untrained network
NETWORK_CALLBACK = """
const e = episodes[select.value], g = graphs[select.value]
neurons.data = {...g, rate: e.rate[0]}
spikes.data = e.spikes
input_spikes.data = e.input_spikes
slider.end = e.length - 1
slider.value = 0
"""

# Play or pause: advance the slider every interval until the end of the episode.
# The speed is fixed while playing, since the interval is set when playback starts.
PLAY_CALLBACK = """
speed.disabled = toggle.active
if (toggle.active) {
    toggle.label = 'Pause'
    if (slider.value >= slider.end) { slider.value = 0 }
    window.snnrl_timer = setInterval(() => {
        if (slider.value >= slider.end) {
            toggle.active = false
        } else {
            slider.value = slider.value + 1
        }
    }, Number(speed.value))
} else {
    toggle.label = 'Play'
    clearInterval(window.snnrl_timer)
}
"""

# One decision step backward (step = -1) or forward (step = 1)
STEP_CALLBACK = """
slider.value = Math.min(slider.end, Math.max(slider.start, slider.value + step))
"""


# ======================================================================
# DERIVED DATA
# ======================================================================

def episode_columns(episode, n_population):
    """
    Per-episode columns for the browser: frames and rasters.
    Input spikes are colored by the observation value of their population (ON and OFF channels alike), background spikes in gray.
    """
    n_obs = len(OBSERVATION_COLORS)
    population = np.asarray(episode["input_ids"]) // n_population
    color = [BACKGROUND_COLOR if k == 2 * n_obs else OBSERVATION_COLORS[k % n_obs] for k in population]
    return {
        **episode,
        "spikes": {"t": episode["spike_times"], "neuron": episode["spike_neurons"]},
        "input_spikes": {"t": episode["input_times"], "input": episode["input_ids"], "color": color},
    }


def neuron_columns(neurons):
    """Columns of the neuron source of one network."""
    alif = np.asarray(neurons["alif"], dtype=bool)
    return {
        "x": neurons["x"],
        "y": neurons["y"],
        "neuron": list(range(len(alif))),
        "type": ["ALIF" if a else "LIF" for a in alif],
        "outline": [ALIF_COLOR if a else LIF_COLOR for a in alif],
        "correlation": neurons["correlation"],
    }


def cart_frame(e, t):
    """Cart, pole, critical-angle and arrow columns of decision step t, as FRAME_CALLBACK computes them."""
    x, angle, p, a = e["cart_x"][t], e["pole_angle"][t], e["p_right"][t], e["action"][t]
    reach = POLE_LENGTH * np.sin(CRITICAL_ANGLE)
    return {
        "cart": {"x": [x], "y": [0]},
        "pole": {"x0": [x], "y0": [0], "x1": [x + POLE_LENGTH * np.sin(angle)], "y1": [POLE_LENGTH * np.cos(angle)]},
        "limits": {"x0": [x, x], "y0": [0, 0], "x1": [x - reach, x + reach],
                   "y1": [POLE_LENGTH * np.cos(CRITICAL_ANGLE)] * 2},
        "arrows": {"x0": [x - CART_HALF_WIDTH, x + CART_HALF_WIDTH],
                   "x1": [x - CART_HALF_WIDTH - ARROW_LENGTH, x + CART_HALF_WIDTH + ARROW_LENGTH],
                   "y": [0, 0], "angle": [np.pi / 2, -np.pi / 2],
                   "alpha": [0.15 + 0.85 * (1 - p), 0.15 + 0.85 * p],
                   "color": [ACTION_COLOR if a == 0 else "gray", ACTION_COLOR if a == 1 else "gray"]},
    }


# ======================================================================
# VIEW
# ======================================================================

class View:
    """View of the episodes in the browser."""

    def __init__(self, data, network_names):
        """Data as written by prepare_data.py"""
        self.network_names = network_names
        self.network = next(iter(network_names))
        self.n_population = data["n_population"] # input neurons per population
        self.episodes = {name: episode_columns(data[name]["episode"], self.n_population) for name in network_names}
        self.graphs = {name: neuron_columns(data[name]["neurons"]) for name in network_names}
        self.side_nodes = data[self.network] # input and readout positions are the same for all networks
        self.figures = {}
        self.sources = {}
        self.widgets = {}
        self.spans = {}

    # Data
    def _create_sources(self):
        """Sources of the first decision step (cart-pole, node activity) and of the whole episode (rasters)."""
        e = self.episodes[self.network]
        p = e["p_right"][0]
        for name, columns in cart_frame(e, 0).items():
            self.sources[name] = ColumnDataSource(data=columns)
        self.sources["neurons"] = ColumnDataSource(data={**self.graphs[self.network], "rate": e["rate"][0]})
        inputs = self.side_nodes["input_nodes"]
        self.sources["inputs"] = ColumnDataSource(data={
            "x": inputs["x"], "y": inputs["y"], "label": INPUT_LABELS, "value": e["input_counts"][0]})
        readouts = self.side_nodes["readout_nodes"]
        self.sources["readouts"] = ColumnDataSource(data={
            "x": readouts["x"], "y": readouts["y"], "label": READOUT_LABELS,
            "value": [1 - p, p, e["value"][0] / VALUE_MAX]})
        self.sources["spikes"] = ColumnDataSource(data=e["spikes"])
        self.sources["input_spikes"] = ColumnDataSource(data=e["input_spikes"])

    # Figures
    @staticmethod
    def _hide_axes(fig):
        fig.axis.visible = False
        fig.grid.visible = False

    def _create_cartpole_figure(self):
        """Track, cart, pole, critical angles and action arrows at the current decision step."""
        fig = figure(title="CartPole", width=RIGHT_WIDTH, height=240, x_range=Range1d(-2.8, 2.8),
                     y_range=Range1d(-0.4, 1.3), tools="", toolbar_location=None)
        self._hide_axes(fig)
        fig.segment(x0=-2.4, y0=-0.1, x1=2.4, y1=-0.1, line_color="gray", line_width=2)
        for x in (-2.4, 2.4):
            fig.segment(x0=x, y0=-0.2, x1=x, y1=0.0, line_color="gray", line_width=2)
        fig.segment(x0="x0", y0="y0", x1="x1", y1="y1", source=self.sources["limits"], line_color="gray",
                    line_dash="dashed", line_width=1.5)
        fig.rect(x="x", y="y", width=2 * CART_HALF_WIDTH, height=0.2, source=self.sources["cart"],
                 fill_color="black", line_color=None)
        fig.segment(x0="x0", y0="y0", x1="x1", y1="y1", source=self.sources["pole"], line_color="#8c564b",
                    line_width=6, line_cap="round")
        arrows = self.sources["arrows"]
        fig.segment(x0="x0", y0="y", x1="x1", y1="y", source=arrows, line_color="color", line_alpha="alpha",
                    line_width=5)
        fig.scatter(x="x1", y="y", source=arrows, marker="triangle", angle="angle", size=16,
                    fill_color="color", fill_alpha="alpha", line_color=None)
        self.figures["cartpole"] = fig

    def _create_network_figure(self):
        """Inputs (top), recurrent neurons (grid) and readouts (bottom), colored by their activity."""
        # From the readouts at the bottom to the inputs at the top, with room for their labels
        bottom = min(self.side_nodes["readout_nodes"]["y"]) - LABEL_MARGIN
        top = max(self.side_nodes["input_nodes"]["y"]) + LABEL_MARGIN
        fig = figure(title="Network", width=RIGHT_WIDTH, height=500, x_range=Range1d(-1.6, 10.6),
                     y_range=Range1d(bottom, top), tools="", toolbar_location=None)
        self._hide_axes(fig)
        renderer = fig.scatter(x="x", y="y", source=self.sources["inputs"], size=NODE_SIZE, marker="square",
                               line_color="gray", fill_color=linear_cmap("value", Inferno256, 0, COUNT_MAX))
        fig.add_tools(HoverTool(renderers=[renderer], tooltips=[
            ("input population", "@label"), ("spikes in this step", "@value"),
        ]))
        fig.text(x="x", y="y", text="label", source=self.sources["inputs"], y_offset=-16, text_align="center",
                 text_baseline="bottom", text_font_size="11pt")
        renderer = fig.scatter(x="x", y="y", source=self.sources["neurons"], size=NODE_SIZE, marker="circle",
                               line_color="outline", line_width=2,
                               fill_color=linear_cmap("rate", Inferno256, 0, RATE_MAX))
        fig.add_tools(HoverTool(renderers=[renderer], tooltips=[
            ("neuron", "@neuron (@type)"), ("rate", "@rate{0} Hz"), ("correlation with P(right)", "@correlation{0.00}"),
        ]))
        fig.scatter(x="x", y="y", source=self.sources["readouts"], size=NODE_SIZE, marker="square",
                    line_color="gray", fill_color=linear_cmap("value", Inferno256, 0, 1))
        fig.text(x="x", y="y", text="label", source=self.sources["readouts"], y_offset=16,
                 text_align="center", text_baseline="top", text_font_size="11pt")
        # Block labels at the vertical centre of each block
        neurons = self.graphs[self.network]
        is_alif = np.array(neurons["type"]) == "ALIF"
        y = np.asarray(neurons["y"])
        fig.text(x=[-1.0, -1.0], y=[y[~is_alif].mean(), y[is_alif].mean()], text=["LIF", "ALIF"], angle=np.pi / 2,
                 text_align="center", text_baseline="middle", text_font_size="10pt",
                 text_color=[LIF_COLOR, ALIF_COLOR])
        self.figures["network"] = fig

    def _create_raster_figure(self):
        """Spikes of all recurrent neurons during the episode."""
        fig = figure(title="Network spikes", width=LEFT_WIDTH, height=260, tools="xpan,xwheel_zoom,reset",
                     x_axis_label="network time (ms)", y_axis_label="neuron", y_range=Range1d(-1, 100))
        fig.scatter(x="t", y="neuron", source=self.sources["spikes"], size=2, color="black", alpha=0.6)
        self.spans["raster"] = Span(location=0, dimension="height", line_color="red", line_width=1.5)
        fig.add_layout(self.spans["raster"])
        self.figures["raster"] = fig

    def _create_input_raster_figure(self):
        """Spikes of all input neurons during the episode, grouped by population."""
        n_in = len(INPUT_LABELS) * self.n_population
        fig = figure(title="Input spikes", width=LEFT_WIDTH, height=200, tools="xpan,xwheel_zoom,reset",
                     x_range=self.figures["raster"].x_range, y_range=Range1d(n_in, -1))
        fig.scatter(x="t", y="input", source=self.sources["input_spikes"], size=2, color="color", alpha=0.8)
        centers = [(k + 0.5) * self.n_population - 0.5 for k in range(len(INPUT_LABELS))]
        fig.yaxis.ticker = FixedTicker(ticks=centers)
        fig.yaxis.major_label_overrides = {c: label for c, label in zip(centers, INPUT_LABELS)}
        fig.ygrid.visible = False
        for k in range(1, len(INPUT_LABELS)):
            fig.add_layout(Span(location=k * self.n_population - 0.5, dimension="width", line_color="lightgray"))
        self.spans["input_raster"] = Span(location=0, dimension="height", line_color="red", line_width=1.5)
        fig.add_layout(self.spans["input_raster"])
        self.figures["input_raster"] = fig

    # Widgets
    def _create_widgets(self):
        """Create the widgets and add their callbacks."""
        e = self.episodes[self.network]
        self.widgets["network"] = Select(title="Network", value=self.network, name="network",
                                         options=list(self.network_names.items()), width=150)
        self.widgets["play"] = Toggle(label="Play", button_type="success", width=70, name="play", align="end")
        self.widgets["speed"] = Select(title="Speed", value="40", name="speed", options=list(SPEEDS.items()),
                                       width=70)
        self.widgets["back"] = Button(label="◀", width=40, name="back", align="end")
        self.widgets["slider"] = Slider(title="Decision step", start=0, end=e["length"] - 1, value=0, step=1,
                                        width=170, name="slider", align="end")
        self.widgets["forward"] = Button(label="▶", width=40, name="forward", align="end")
        p, a = e["p_right"][0], e["action"][0]
        self.widgets["status"] = Div(width=RIGHT_WIDTH, text=(
            f"step 1 of {e['length']} &nbsp;|&nbsp; P(right) = {p:.2f} &nbsp;|&nbsp; "
            f"action: {'push right' if a == 1 else 'push left'} &nbsp;|&nbsp; value = {e['value'][0]:.0f}"))

        frame = CustomJS(
            args=dict(episodes=self.episodes, select=self.widgets["network"], slider=self.widgets["slider"],
                      cart=self.sources["cart"], pole=self.sources["pole"], limits=self.sources["limits"],
                      arrows=self.sources["arrows"], neurons=self.sources["neurons"],
                      inputs=self.sources["inputs"], readouts=self.sources["readouts"],
                      raster_cursor=self.spans["raster"], input_cursor=self.spans["input_raster"],
                      status=self.widgets["status"],
                      pole_length=POLE_LENGTH, critical_angle=CRITICAL_ANGLE, half_width=CART_HALF_WIDTH,
                      arrow_length=ARROW_LENGTH, action_color=ACTION_COLOR, value_max=VALUE_MAX, n_steps=N_STEPS),
            code=FRAME_CALLBACK,
        )
        self.widgets["slider"].js_on_change("value", frame)
        self.widgets["network"].js_on_change("value", CustomJS(
            args=dict(episodes=self.episodes, graphs=self.graphs, select=self.widgets["network"],
                      slider=self.widgets["slider"], neurons=self.sources["neurons"], spikes=self.sources["spikes"],
                      input_spikes=self.sources["input_spikes"]),
            code=NETWORK_CALLBACK,
        ))
        self.widgets["network"].js_on_change("value", frame)
        for name, step in [("back", -1), ("forward", 1)]:
            self.widgets[name].js_on_event("button_click", CustomJS(
                args=dict(slider=self.widgets["slider"], step=step), code=STEP_CALLBACK))
        self.widgets["play"].js_on_change("active", CustomJS(
            args=dict(toggle=self.widgets["play"], slider=self.widgets["slider"], speed=self.widgets["speed"]),
            code=PLAY_CALLBACK,
        ))

    # Page layout
    def create_layout(self):
        """Create the sources, figures and widgets, and return the layout."""
        self._create_sources()
        self._create_cartpole_figure()
        self._create_network_figure()
        self._create_raster_figure()
        self._create_input_raster_figure()
        self._create_widgets()
        controls = row(self.widgets["network"], self.widgets["play"], self.widgets["speed"],
                       self.widgets["back"], self.widgets["slider"], self.widgets["forward"])
        left = column([Div(text=TEXT, width=LEFT_WIDTH), self.figures["input_raster"], self.figures["raster"]])
        right = column([controls, self.figures["cartpole"], self.widgets["status"], self.figures["network"]], align='end')
        return column(
            Div(text=TITLE, width=LEFT_WIDTH),
            row(left, right),
            Div(text=REFERENCES, width=PAGE_WIDTH),
            styles={"margin": "0 auto"},
        )


def main():
    data = json.loads(EPISODE_DATA.read_text())
    layout = View(data, NETWORKS).create_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(file_html(layout, CDN, "Recurrent spiking network balancing a pole"))


if __name__ == "__main__":
    main()
