"""Generate the four project architecture diagrams for ``docs/images/``.

Produces PNGs at 300 dpi using matplotlib (Agg backend, headless-safe):

- ``docs/images/pipeline_diagram.png``  — end-to-end pipeline flow
- ``docs/images/data_flow.png``         — data artifacts through the stages
- ``docs/images/ml_workflow.png``       — ML workflow (split → models → winner)
- ``docs/images/folder_architecture.png`` — repository folder tree

Usage:
    python scripts/generate_diagrams.py

The diagrams are static snapshots of the implemented pipeline (M1–M11); the
text matches ``README.md`` / ``docs/*.md`` so documentation never drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DPI = 300
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"

# Palette consistent with the rest of the project's figures.
_BLUE = "#2E86AB"
_AMBER = "#F5A623"
_RED = "#C1121F"
_DARK = "#22313F"
_LIGHT = "#F4F6F7"


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = _BLUE,
    fontsize: int = 9,
) -> None:
    """Draw a rounded box with centered text."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=_DARK,
        facecolor=color,
        alpha=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="white",
        fontweight="bold",
        wrap=True,
    )


def _arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    text: str = "",
) -> None:
    """Draw an arrow between two points with an optional label."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=_DARK,
        )
    )
    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=_DARK,
            style="italic",
        )


def _new_axes(title: str, wide: bool = False) -> tuple[plt.Figure, plt.Axes]:
    """Figure with no visible axes and a title."""
    figure = plt.figure(figsize=(12 if wide else 10, 6.5))
    ax = figure.add_axes((0.02, 0.02, 0.96, 0.88))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    figure.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    return figure, ax


def _pipeline_diagram(path: Path) -> None:
    """End-to-end pipeline: raw data to risk map."""
    figure, ax = _new_axes("End-to-End Pipeline — Conflict Escalation Forecasting", wide=True)
    steps = [
        ("data/raw\nACLED CSVs", _BLUE),
        ("Ingest + Validate\n(M3–M4)", _BLUE),
        ("Features\n(M5)", _BLUE),
        ("Labels\n(M6)", _BLUE),
        ("Chronological Split\n(M7)", _BLUE),
        ("LightGBM\n(M8)", _AMBER),
        ("XGBoost +\nCompare (M9)", _AMBER),
        ("Winner →\nescalation_best.pkl", _RED),
        ("SHAP\n(M10)", _RED),
        ("Risk Map +\nVisuals (M11)", _RED),
    ]
    n = len(steps)
    w, gap = 0.82, 0.16
    x0 = (10 - (n * w + (n - 1) * gap)) / 2
    y, h = 2.4, 1.3
    for i, (label, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, label, color, fontsize=8)
        if i < n - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _data_flow(path: Path) -> None:
    """Data artifacts flowing through the stages."""
    figure, ax = _new_axes("Data Flow — Artifacts per Stage", wide=True)
    stages = [
        ("raw CSV\n127,353 rows", "cleaned_events\n127,052 × 14", "features\n44,146 × 37",
         "labeled_features\n43,981 × 38", "split_train 30,790\nsplit_val 6,522\nsplit_test 6,669"),
    ]
    x0, w, gap, y, h = 0.4, 1.7, 0.35, 3.1, 1.5
    for i, label in enumerate(stages[0]):
        _box(ax, x0 + i * (w + gap), y, w, h, label, _BLUE, fontsize=8)
        if i < 4:
            _arrow(ax, x0 + i * (w + gap) + w, y + h / 2, x0 + (i + 1) * (w + gap), y + h / 2)
    flow = [
        ("loader + validation", "feature_engineer", "label_engineer", "chronological_split"),
    ]
    y2 = 1.2
    for i, label in enumerate(flow[0]):
        _box(ax, x0 + i * (w + gap) + 0.15, y2, w - 0.3, 0.8, label, _AMBER, fontsize=8)
        if i < 3:
            _arrow(ax, x0 + i * (w + gap) + w - 0.05, y2 + 0.4, x0 + (i + 1) * (w + gap) + 0.15, y2 + 0.4)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _ml_workflow(path: Path) -> None:
    """ML workflow: split → train → compare → winner → SHAP → visualize."""
    figure, ax = _new_axes("ML Workflow — Train / Validate / Select / Explain")
    nodes = [
        ("Chronological split\n(no shuffle, no leakage)", 1.0, 4.6),
        ("LightGBM\nseed 42", 3.9, 4.6),
        ("XGBoost\nseed 42, same data", 6.8, 4.6),
        ("Compare: F1 → PR-AUC →\nBrier → simplicity", 4.0, 2.4),
        ("Winner: XGBoost\nthreshold 0.25", 6.8, 2.4),
        ("SHAP explainability\n(test window)", 4.0, 0.4),
        ("Risk map + visualizations", 7.6, 0.4),
    ]
    for label, x, y in nodes:
        _box(ax, x - 1.05, y - 0.45, 2.1, 0.9, label, _BLUE if y > 3 else (_RED if "Winner" in label else _AMBER), fontsize=8)
    _arrow(ax, 2.15, 4.6, 2.85, 4.6)
    _arrow(ax, 5.0, 4.6, 5.75, 4.6)
    _arrow(ax, 4.0, 4.15, 4.0, 2.85)          # LGBM down to compare
    _arrow(ax, 7.6, 4.15, 7.6, 3.4)           # XGB down
    _arrow(ax, 5.75, 2.4, 6.05, 2.4)          # compare → winner
    _arrow(ax, 7.15, 2.4, 5.0, 0.85)          # winner → SHAP
    _arrow(ax, 5.9, 0.4, 6.6, 0.4)            # SHAP → visualize
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _folder_architecture(path: Path) -> None:
    """Repository folder tree rendered as nested boxes."""
    figure, ax = _new_axes("Repository Folder Architecture")
    rows = [
        ("config.py", 0, 0), ("run_pipeline.py", 0, 0), ("requirements.txt", 0, 0),
        ("data/", 1, 0), ("  raw/ + processed/", 2, 0),
        ("models/", 1, 0), ("  escalation_best.pkl etc.", 2, 0),
        ("reports/", 1, 0), ("  maps/ figures/ dashboard/ *.md", 2, 0),
        ("docs/", 1, 0), ("  architecture.md model.md usage.md results.md", 2, 0),
        ("src/", 1, 0), ("  data_loader · data_validation · feature_engineer · label_engineer", 2, 0),
        ("  split · models · explainability · visualization · pipeline", 2, 0),
        ("tests/", 1, 0), ("  test_*.py (243 tests)", 2, 0),
        ("notebooks/", 1, 0), ("  (M13 — executed notebooks)", 2, 0),
    ]
    y = 5.5
    for label, indent, _ in rows:
        color = _BLUE if indent == 0 else (_AMBER if indent == 1 else _LIGHT)
        text_color = "white" if indent < 2 else _DARK
        x = 0.3 + indent * 0.4
        w = 9.4 - indent * 0.4
        ax.text(
            x + 0.05,
            y,
            label,
            fontsize=8 if indent == 2 else 9,
            fontfamily="monospace",
            color=text_color,
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=color,
                edgecolor=_DARK,
                linewidth=0.8,
                alpha=0.9,
            ),
        )
        y -= 0.28
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def main() -> int:
    """Render all four diagrams into ``docs/images/``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _pipeline_diagram(OUT_DIR / "pipeline_diagram.png")
    _data_flow(OUT_DIR / "data_flow.png")
    _ml_workflow(OUT_DIR / "ml_workflow.png")
    _folder_architecture(OUT_DIR / "folder_architecture.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
