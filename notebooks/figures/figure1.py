import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 1

    Builds the three-panel figure of the abstract from the committed result
    tables in `results/` and writes `figures/figure1.pdf` (the version in
    the abstract) and `figures/figure1-preview.png` (the README teaser).

    * **A**: Representational similarity (RS), top-1 accuracy (ACC) and
      error consistency (EC) against the target RS for ResNet-50 trained
      towards a ResNet-50 reference (three similarity measures) and for
      AlexNet trained towards macaque IT responses.
    * **B**: Distributions of the classifier (`fc`) weights of three
      ResNet-50 networks from (A) against the torchvision reference.
    * **C**: AlexNet / CIFAR-10 networks trained to match baseline
      networks' RS and accuracy at a different EC.

    This notebook loads the result tables and generates and saves the
    figure using `dissociation.figure`. Run it headless with
    `uv run --locked python notebooks/figures/figure1.py`.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    from dissociation import figure
    from dissociation import results

    NOTEBOOK_DIR = mo.notebook_dir() or Path.cwd()
    REPO_ROOT = NOTEBOOK_DIR.parents[1]
    RESULTS_DIR = REPO_ROOT / "results"
    OUTPUT_DIR = REPO_ROOT / "figures"
    OUTPUT_STEM = "figure1"
    return OUTPUT_DIR, OUTPUT_STEM, RESULTS_DIR, figure, results


@app.cell
def _(RESULTS_DIR, results):
    curves = results.load_sweeps(RESULTS_DIR / "sweeps.csv")
    panel_b = results.load_panel_b(RESULTS_DIR / "fc_weight_violins.npz")
    panel_c = results.load_panel_c(RESULTS_DIR / "panel_c_runs.csv")
    return curves, panel_b, panel_c


@app.cell
def _(curves, figure, panel_b, panel_c):
    fig = figure.build_figure(curves, panel_b, panel_c)
    fig
    return (fig,)


@app.cell
def _(OUTPUT_DIR, OUTPUT_STEM, fig, figure):
    written = [
        figure.save_figure(fig, OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"),
        figure.save_figure(fig, OUTPUT_DIR / f"{OUTPUT_STEM}-preview.png"),
    ]
    print("\n".join(str(path) for path in written))
    return


if __name__ == "__main__":
    app.run()
