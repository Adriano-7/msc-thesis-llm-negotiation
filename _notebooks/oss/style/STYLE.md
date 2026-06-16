# OSS Notebook Style Helpers

This folder contains the small style module used by
the main notebooks under `_notebooks/oss`.

The old thesis figure-generation pipeline has been removed. This folder is now
only a notebook-local dependency:

```
_notebooks/oss/style/
  _bootstrap.py   # finds the repo root and sets import paths
  style.py        # matplotlib theme, colours, CIs, heatmap, small output helpers
  STYLE.md        # this note
```

## How The Notebooks Use It

Each notebook adds this folder to `sys.path`:

```python
sys.path.insert(0, str(ROOT_DIR / "_notebooks" / "oss" / "style"))
```

and imports:

```python
import style
from style import wilson_ci, errbars_from_ci
```

Keep `style.py` and `_bootstrap.py` with the notebooks unless they are
rewritten to inline or replace those helpers.

## Available Helpers

- `apply_thesis_style()` sets the shared sans-serif matplotlib report skin:
  colorblind cycle, muted axis text, value-axis gridlines, and constrained
  layout.
- `FULL_WIDTH` and `HALF_WIDTH` define common figure sizes.
- `SIZE_ORDER`, `SIZE_LABEL`, and `GAME_ORDER` provide canonical labels.
- `wilson_ci(k, n)` returns Wilson intervals for proportions.
- `bootstrap_ci(values, seed=0)` returns deterministic bootstrap intervals.
- `errbars_from_ci(centers, cis)` converts CIs into matplotlib error bars.
- `heatmap(...)` renders annotated matplotlib heatmaps without seaborn.

`style.py` still includes `save_fig`, `write_table`, and `Numbers` for
compatibility, but the current notebooks use local PNG writers for their core
analysis figures.

## Conventions

- Use the tier labels from `SIZE_LABEL`: `4–9B`, `12–14B`, `24–27B`.
- Use Wilson intervals for proportions and bootstrap intervals for payoff means.
- Win rates should remain ties-excluded, matching Chapter 3.
- Prefer each notebook's local analysis over resurrecting deleted figure scripts.
