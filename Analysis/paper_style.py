"""AOAS-compliant Matplotlib style for every figure that goes into the paper.

The rule this module exists to enforce: a figure is authored at the size it
will occupy on the printed page, so LaTeX never rescales it.  With
``\\includegraphics[width=\\textwidth]{fig.pdf}`` on a figure that is already
``TEXTWIDTH_IN`` wide the scale factor is exactly 1, which means the 9 pt tick
labels set here really are 9 pt in the journal and a 1.1 pt curve really is
1.1 pt.  Authoring at 10 x 5 in and letting LaTeX shrink to 0.55 was what
turned nominal 15 pt labels into 8 pt and 0.6 pt grid lines into 0.33 pt.

AOAS requirements encoded here:
  * vector PDF only (raster only where the content is genuinely an image);
  * 8-10 pt text at final size;
  * Computer Modern, matching the manuscript body font;
  * no curve thinner than ~0.7 pt;
  * no in-figure title -- the caption carries that information;
  * colour is never the only channel: arms and components also differ by
    line style and marker.
"""

from __future__ import annotations

from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page geometry.  AOAS (imsart) text block is 5.5 in wide.
# ---------------------------------------------------------------------------
TEXTWIDTH_IN = 5.5
WIDE = TEXTWIDTH_IN           # a full-width main figure
NARROW = 0.7 * TEXTWIDTH_IN   # the 0.7\textwidth option

# Type 42 keeps the text selectable and searchable in the submitted PDF and
# avoids the Type 3 fonts some journal pipelines reject outright.
_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    # cmr10 has no U+2212; without this every negative tick warns and drops
    # its minus sign.
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "figure.labelsize": 9,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "patch.linewidth": 0.7,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "axes.grid": False,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
}

# Grayscale-safe secondary channel.  Index these by arm label or component.
ARM_LINESTYLES = {"control": (0, (4.0, 1.8)), "treatment": "-"}
ARM_MARKERS = {"control": "s", "treatment": "o"}
COMPONENT_LINESTYLES = (
    "-",
    (0, (4.0, 1.8)),
    (0, (1.2, 1.2)),
    (0, (5.0, 1.5, 1.0, 1.5)),
    (0, (3.0, 1.2, 1.0, 1.2, 1.0, 1.2)),
)
COMPONENT_MARKERS = ("o", "s", "^", "D", "v")


def apply(**overrides):
    """Install the AOAS style globally.  Safe to call more than once."""
    mpl.rcParams.update(_RC)
    if overrides:
        mpl.rcParams.update(overrides)
    return mpl.rcParams


def figsize(width=WIDE, aspect=0.62, *, fraction=None):
    """Figure size in inches at final printed size.

    ``aspect`` is height/width.  ``fraction`` overrides ``width`` as a
    fraction of the text block, so ``figsize(fraction=0.5)`` is a half-width
    panel.
    """
    if fraction is not None:
        width = fraction * TEXTWIDTH_IN
    return (float(width), float(width) * float(aspect))


def panel_label(axis, letter, *, x=-0.02, y=1.04, **kwargs):
    """Put an AOAS-style "(a)" label above the top-left corner of a panel."""
    axis.text(
        x, y, f"({letter})", transform=axis.transAxes,
        ha="right", va="bottom", fontsize=9, **kwargs,
    )
    return axis


def label_panels(axes, *, start=0, **kwargs):
    """Label a grid of axes (a), (b), (c), ... in row-major order."""
    letters = cycle("abcdefghijklmnopqrstuvwxyz")
    for _ in range(start):
        next(letters)
    flat = axes.ravel() if hasattr(axes, "ravel") else list(axes)
    for axis in flat:
        panel_label(axis, next(letters), **kwargs)
    return axes


def style_arm(label, color):
    """Keyword arguments giving an arm its colour *and* its line style."""
    key = str(label).lower()
    return {
        "color": color,
        "linestyle": ARM_LINESTYLES.get(key, "-"),
        "marker": ARM_MARKERS.get(key, "o"),
    }


def save(fig, path_stem, *, formats=("pdf",)):
    """Save a figure as vector PDF only (AOAS: no PNG for plots)."""
    from pathlib import Path

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


# Quiet empirical clouds over arm-coloured clinical density contours.
DENSITY_POINT_SIZE = 1.2
DENSITY_POINT_ALPHA = 0.025


def clinical_density_cmap(arm):
    """White-to-arm density scale; CLC/Treatment green, SC/Control red."""
    from matplotlib.colors import LinearSegmentedColormap
    from Analysis.paper_config import ARM_COLORS
    labels = {"CLC": 1, "Treatment": 1, "SC": 0, "Control": 0, 1: 1, 0: 0}
    if arm not in labels:
        raise ValueError(f"Unknown randomized arm: {arm!r}")
    label = labels[arm]
    return LinearSegmentedColormap.from_list(
        f"clinical_density_{label}", ["white", ARM_COLORS[label]], N=256,
    )
