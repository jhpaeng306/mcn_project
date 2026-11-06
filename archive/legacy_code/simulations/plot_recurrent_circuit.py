"""Render the four-population V1 recurrent circuit used by the simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch, FancyBboxPatch


EXCITATORY = "#0B2D5C"
INHIBITORY = "#7A1F1F"
TEXT = "#172033"
INPUT_FILL = "#ECEFF3"
INPUT_EDGE = "#657080"


def _connection(
    ax: plt.Axes,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    inhibitory: bool = False,
    curvature: float = 0.0,
    source_shrink: float = 31.0,
    target_shrink: float = 31.0,
    linewidth: float = 2.8,
) -> None:
    """Draw one continuous connection with a pointed head or flat T-bar."""
    color = INHIBITORY if inhibitory else EXCITATORY
    arrowstyle = (
        ArrowStyle.BracketB(widthB=0.9, lengthB=0.0)
        if inhibitory
        else ArrowStyle.CurveFilledB(head_length=0.45, head_width=0.28)
    )
    patch = FancyArrowPatch(
        source,
        target,
        arrowstyle=arrowstyle,
        connectionstyle=f"arc3,rad={curvature}",
        shrinkA=source_shrink,
        shrinkB=target_shrink,
        mutation_scale=18.0,
        linewidth=linewidth,
        color=color,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    ax.add_patch(patch)


def _self_connection(
    ax: plt.Axes,
    center: tuple[float, float],
    *,
    inhibitory: bool = False,
    side: str = "top",
) -> None:
    """Draw a recurrent self-connection outside a population node."""
    x, y = center
    if side == "top":
        start = (x - 0.18, y + 0.29)
        end = (x + 0.18, y + 0.29)
        curvature = -2.4
    elif side == "right":
        start = (x + 0.29, y + 0.18)
        end = (x + 0.29, y - 0.18)
        curvature = -2.4
    else:
        start = (x - 0.29, y - 0.18)
        end = (x - 0.29, y + 0.18)
        curvature = -2.4
    _connection(
        ax,
        start,
        end,
        inhibitory=inhibitory,
        curvature=curvature,
        source_shrink=0.0,
        target_shrink=0.0,
        linewidth=2.7,
    )


def render(output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 8.0), facecolor="white")
    ax.set_facecolor("white")

    positions = {
        "E": (0.0, 1.25),
        "PV": (2.25, 1.25),
        "SST": (2.25, -1.15),
        "VIP": (0.0, -1.15),
    }

    # Direct bottom-up visual input: E and PV only.
    input_center = (1.125, 2.47)
    _connection(
        ax,
        (0.72, 2.03),
        positions["E"],
        curvature=-0.04,
        source_shrink=0.0,
        target_shrink=31.0,
    )
    _connection(
        ax,
        (1.53, 2.03),
        positions["PV"],
        curvature=0.04,
        source_shrink=0.0,
        target_shrink=31.0,
    )

    # Excitatory recurrent output from E to every population.
    _self_connection(ax, positions["E"], side="left")
    _connection(ax, positions["E"], positions["PV"], curvature=-0.15)
    _connection(ax, positions["E"], positions["SST"], curvature=-0.12)
    _connection(ax, positions["E"], positions["VIP"], curvature=-0.15)

    # PV inhibits E and itself.
    _connection(
        ax,
        positions["PV"],
        positions["E"],
        inhibitory=True,
        curvature=-0.15,
    )
    _self_connection(ax, positions["PV"], inhibitory=True, side="right")

    # SST inhibits E, PV, and VIP.
    _connection(
        ax,
        positions["SST"],
        positions["E"],
        inhibitory=True,
        curvature=-0.12,
    )
    _connection(
        ax,
        positions["SST"],
        positions["PV"],
        inhibitory=True,
        curvature=-0.15,
    )
    _connection(
        ax,
        positions["SST"],
        positions["VIP"],
        inhibitory=True,
        curvature=-0.15,
    )

    # VIP closes the disinhibitory loop by inhibiting SST.
    _connection(
        ax,
        positions["VIP"],
        positions["SST"],
        inhibitory=True,
        curvature=-0.15,
    )

    for name, center in positions.items():
        color = EXCITATORY if name == "E" else INHIBITORY
        node = Circle(center, 0.34, facecolor=color, edgecolor=color, linewidth=2.0, zorder=4)
        ax.add_patch(node)
        ax.text(
            *center,
            name,
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold",
            zorder=5,
        )

    input_box = FancyBboxPatch(
        (input_center[0] - 0.82, input_center[1] - 0.42),
        1.64,
        0.84,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=INPUT_FILL,
        edgecolor=INPUT_EDGE,
        linewidth=1.8,
        zorder=4,
    )
    ax.add_patch(input_box)
    ax.text(
        *input_center,
        "Visual stimulus\n$\\varepsilon^2$ scale\n$2k_0$ and/or $2\\omega_0$",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=11.5,
        zorder=5,
    )

    # Compact arrowhead legend.
    legend_y = -2.05
    _connection(
        ax,
        (-1.75, legend_y),
        (-0.90, legend_y),
        source_shrink=0.0,
        target_shrink=0.0,
        linewidth=2.8,
    )
    ax.text(-0.72, legend_y, "Excitatory", va="center", color=TEXT, fontsize=11.5)
    _connection(
        ax,
        (0.45, legend_y),
        (1.30, legend_y),
        inhibitory=True,
        source_shrink=0.0,
        target_shrink=0.0,
        linewidth=2.8,
    )
    ax.text(1.50, legend_y, "Inhibitory", va="center", color=TEXT, fontsize=11.5)

    ax.set_title(
        "Four-population V1 recurrent circuit",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        pad=18,
    )
    ax.set_xlim(-1.55, 3.80)
    ax.set_ylim(-2.35, 3.28)
    ax.set_aspect("equal")
    ax.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    render(Path(__file__).with_name("v1_recurrent_circuit.png"))
