"""Source code and figure-generation utilities for the accompanying abstract.

Reproduces the figure and the demo from "Representation can dissociate from
function and behaviour in task-optimised vision networks" (CCN 2026).

Submodules are not imported eagerly, so the figure path (``results``,
``colors``, ``plotting``, ``figure``) stays free of torch.
"""

__all__ = [
    "colors",
    "figure",
    "imagenet",
    "losses",
    "metrics",
    "models",
    "plotting",
    "results",
    "runs",
    "similarity",
]

__version__ = "1.0.0"
