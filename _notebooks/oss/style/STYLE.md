# Cross-Play Notebook Style Helpers

This folder contains the small style module used by
`_notebooks/oss/1_cross_play_benchmark.ipynb`.

The old thesis figure-generation pipeline has been removed. This folder is now
only a notebook-local dependency:

```
_notebooks/oss/style/
  _bootstrap.py   # finds the repo root and sets import paths
  style.py        # matplotlib theme, colours, CIs, heatmap, small output helpers
  STYLE.md        # this note
```

## How The Notebook Uses It

The notebook adds this folder to `sys.path`:

```python
sys.path.insert(0, str(ROOT_DIR / "_notebooks" / "oss" / "style"))
```

and imports:

```python
import style
from style import wilson_ci, errbars_from_ci
```

Keep `style.py` and `_bootstrap.py` with the notebook unless the notebook is
rewritten to inline or replace those helpers.

## Available Helpers

- `apply_thesis_style()` sets the shared matplotlib defaults.
- `FULL_WIDTH` and `HALF_WIDTH` define common figure sizes.
- `SIZE_ORDER`, `SIZE_LABEL`, and `GAME_ORDER` provide canonical labels.
- `wilson_ci(k, n)` returns Wilson intervals for proportions.
- `bootstrap_ci(values, seed=0)` returns deterministic bootstrap intervals.
- `errbars_from_ci(centers, cis)` converts CIs into matplotlib error bars.
- `heatmap(...)` renders annotated matplotlib heatmaps without seaborn.

`style.py` still includes `save_fig`, `write_table`, and `Numbers` for
compatibility, but the cross-play notebook does not rely on them for its core
analysis.

## Conventions

- Use the tier labels from `SIZE_LABEL`: `4–9B`, `12–14B`, `24–27B`.
- Use Wilson intervals for proportions and bootstrap intervals for payoff means.
- Win rates should remain ties-excluded, matching Chapter 3.
- Prefer the notebook's local analysis over resurrecting deleted figure scripts.
