"""Notebook style + descriptive-statistics helpers.

The cross-play notebook imports from here so fonts, colors, CI conventions and
the ties-excluded win-rate definition are encoded once.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap as bs

# ----------------------------------------------------------------------------- style

FULL_WIDTH = 6.0   # inches, ~= feupteses text block at \linewidth
HALF_WIDTH = 3.0


def apply_thesis_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
    })


# Okabe-Ito-derived colorblind palette (matches seaborn "colorblind" hues used
# in the exploration notebooks).
_BLUE = "#0173b2"
_ORANGE = "#de8f05"
_GREEN = "#029e73"
_RED = "#d55e00"
_PURPLE = "#cc78bc"
_GRAY = "#949494"
_YELLOW = "#ece133"
_SKY = "#56b4e9"

FAMILY_COLORS = {"gemma": _BLUE, "qwen": _GREEN, "ministral": _ORANGE, "mistral": _ORANGE}
PERSONA_COLORS = {"default": _BLUE, "desperate": _ORANGE, "cunning": _GREEN}
STRATEGY_COLORS = {
    "Default": _GRAY,
    "Self-Refine": _BLUE,
    "Team (homo)": _ORANGE,
    "Team (hetero)": _GREEN,
}
COND_COLORS = {"DD": _GRAY, "RR": _GREEN, "DR": _SKY, "RD": _BLUE}

SIZE_ORDER = ["very_small", "small", "medium"]
SIZE_LABEL = {"very_small": "4–9B", "small": "12–14B", "medium": "24–27B"}
GAME_ORDER = ["Trading", "Ultimatum", "BuySell"]

FAMILY_LABEL = {"gemma": "Gemma", "qwen": "Qwen", "ministral": "Mistral", "mistral": "Mistral"}


def family_of(model_name: str) -> str:
    n = model_name.lower()
    for fam in ("gemma", "qwen", "ministral"):
        if fam in n:
            return fam
    if "mistral" in n:
        return "ministral"
    return n


# ------------------------------------------------------------------------ statistics

def wilson_ci(k, n, conf=0.95):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (np.nan, np.nan)
    from scipy.stats import norm  # scipy ships with the env; lazy import
    z = norm.ppf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, n_boot=10000, conf=0.95, seed=0):
    """Percentile bootstrap CI for the mean. Deterministic (fixed seed)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - conf) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def win_rate(wins, losses):
    """Ties-excluded win rate (per Chapter 3): wins / (wins + losses).

    Returns (rate, k, n) so callers can attach a Wilson CI."""
    n = wins + losses
    return (np.nan if n == 0 else wins / n), wins, n


# ------------------------------------------------------------------------- plotting

def heatmap(ax, matrix, row_labels, col_labels, fmt="{:.2f}", cmap="Blues",
            vmin=None, vmax=None, na_color="#f0f0f0"):
    """Annotated heatmap (plain matplotlib stand-in for sns.heatmap)."""
    m = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(m)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(na_color)
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    lo = vmin if vmin is not None else np.nanmin(m)
    hi = vmax if vmax is not None else np.nanmax(m)
    mid = lo + 0.55 * (hi - lo)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isnan(m[i, j]):
                continue
            color = "white" if m[i, j] > mid else "black"
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                    color=color, fontsize=8)
    return im


def errbars_from_ci(centers, cis):
    """Convert [(lo,hi),...] into the 2xN yerr array matplotlib expects."""
    centers = np.asarray(centers, dtype=float)
    lo = np.array([c[0] for c in cis], dtype=float)
    hi = np.array([c[1] for c in cis], dtype=float)
    # CI bounds can undershoot the centre by float error at p in {0, 1}
    return np.clip(np.vstack([centers - lo, hi - centers]), 0, None)


# --------------------------------------------------------------------------- output

def save_fig(fig, slug):
    path = os.path.join(bs.FIG_DIR, f"{slug}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, bs.REPO_ROOT)}")
    return path


def write_table(latex, slug):
    path = os.path.join(bs.TAB_DIR, f"{slug}.tex")
    with open(path, "w") as f:
        f.write(latex.rstrip() + "\n")
    print(f"  wrote {os.path.relpath(path, bs.REPO_ROOT)}")
    return path


def fmt_ci(value, ci, fmt="{:.2f}"):
    """'$0.68~[0.61, 0.74]$'-style cell (math mode so minus signs typeset)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"${fmt.format(value)}~[{fmt.format(ci[0])}, {fmt.format(ci[1])}]$"


class Numbers:
    """Collects headline numbers and emits a numbers_<section>.tex macro file.

    Prose in chapter4.tex references these macros (e.g. \\ResMedTradingCompRetry)
    so re-running the pipeline can never desynchronize text and figures.
    """

    def __init__(self, section_slug):
        self.section_slug = section_slug
        self._macros = []

    def add(self, name, value, fmt="{:.2f}"):
        if not name.isalpha():
            raise ValueError(f"macro name must be alphabetic: {name}")
        text = fmt.format(value) if isinstance(value, (int, float)) else str(value)
        self._macros.append((name, text))

    def write(self):
        path = os.path.join(bs.TAB_DIR, f"numbers_{self.section_slug}.tex")
        with open(path, "w") as f:
            f.write("% Auto-generated by notebook style helpers — do not edit.\n")
            for name, text in self._macros:
                f.write(f"\\newcommand{{\\{name}}}{{{text}}}\n")
        print(f"  wrote {os.path.relpath(path, bs.REPO_ROOT)} ({len(self._macros)} macros)")
        return path
