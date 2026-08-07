"""Executable end-to-end pipeline.

Stage 1 (current milestone): ingest -> validate -> district master -> save.
Later milestones add feature engineering, labels, split, training,
evaluation, explainability, and visualization behind the same CLI.

Usage:
    python run_pipeline.py [--stage ingest]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import config
from src import (
    data_loader,
    data_validation,
    explainability,
    feature_engineer,
    label_engineer,
    logging_config,
    pipeline,
    split,
)
from src.exceptions import ConflictForecastError, DataLoadError


def run_ingest_stage() -> dict[str, object]:
    """Run data ingestion + validation and persist the cleaned artifacts.

    Returns:
        A summary dict (row counts, dates, geo units, written files).
    """
    logger = logging_config.get_logger("run_pipeline")
    logger.info("=== Stage: ingest ===")
    raw = data_loader.load_raw_data()
    clean = data_validation.validate_dataset(raw)
    master = data_validation.build_district_master(clean)
    written = data_loader.save_clean_outputs(clean, master, config.DATA_PROCESSED_DIR)

    summary: dict[str, object] = {
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(clean)),
        "geo_units": int(clean["geo_unit"].nunique()),
        "countries": int(clean["country"].nunique()),
        "date_start": str(clean["event_date"].min().date()),
        "date_end": str(clean["event_date"].max().date()),
        "written": {name: [str(p) for p in paths] for name, paths in written.items()},
    }
    logger.info("Ingest summary: %s", summary)
    return summary


def run_features_stage() -> dict[str, object]:
    """Build the engineered feature table from the cleaned dataset.

    Reads ``cleaned_events.parquet`` (produced by the ingest stage), builds
    all PRD §11.3 features, validates them, and writes
    ``features.parquet``/``features.csv`` plus the feature summary report.

    Raises:
        ConflictForecastError: if the cleaned dataset is missing or invalid.
    """
    logger = logging_config.get_logger("run_pipeline")
    logger.info("=== Stage: features ===")
    clean_path = config.DATA_PROCESSED_DIR / f"{config.CLEANED_EVENTS_FILE}.parquet"
    if not clean_path.is_file():
        raise DataLoadError(
            f"Cleaned dataset not found at {clean_path} — run the 'ingest' stage first."
        )
    clean = pd.read_parquet(clean_path)
    features = feature_engineer.build_features(clean)  # validates internally
    written = data_loader.save_dataframe(
        features, config.FEATURES_FILE, config.DATA_PROCESSED_DIR
    )
    summary_path = feature_engineer.write_feature_summary(
        features, config.REPORTS_DIR / config.FEATURE_SUMMARY_FILE
    )
    summary: dict[str, object] = {
        "feature_rows": int(len(features)),
        "feature_columns": int(features.shape[1]),
        "geo_units": int(features["geo_unit"].nunique()),
        "date_start": str(features["event_date"].min().date()),
        "date_end": str(features["event_date"].max().date()),
        "written": [str(p) for p in written],
        "summary_report": str(summary_path),
    }
    logger.info("Features stage summary: %s", summary)
    return summary


def run_labels_stage() -> dict[str, object]:
    """Attach escalation labels to the feature table and persist the result.

    Reads ``features.parquet`` and ``cleaned_events.parquet``, builds the
    PRD §11.2 labels (future-only), validates them, and writes
    ``labeled_features.parquet``/``labeled_features.csv`` plus the label
    summary report and timeline PNG.

    Raises:
        ConflictForecastError: if a required dataset is missing or invalid.
    """
    logger = logging_config.get_logger("run_pipeline")
    logger.info("=== Stage: labels ===")
    labeled = label_engineer.build_labeled_dataset()
    written = data_loader.save_dataframe(
        labeled, config.LABELED_FEATURES_FILE, config.DATA_PROCESSED_DIR
    )
    summary_path = label_engineer.write_label_summary(
        labeled,
        config.REPORTS_DIR / config.LABEL_SUMMARY_FILE,
        timeline_path=config.REPORTS_DIR / config.LABEL_TIMELINE_FILE,
    )
    positives = int(labeled[config.LABEL_COLUMN].sum())
    summary: dict[str, object] = {
        "labeled_rows": int(len(labeled)),
        "geo_units": int(labeled["geo_unit"].nunique()),
        "date_start": str(labeled["event_date"].min().date()),
        "date_end": str(labeled["event_date"].max().date()),
        "positives": positives,
        "positive_rate": 100 * positives / len(labeled),
        "written": [str(p) for p in written],
        "summary_report": str(summary_path),
    }
    logger.info("Labels stage summary: %s", summary)
    return summary


def run_split_stage() -> dict[str, object]:
    """Split the labeled dataset into chronological train/val/test subsets.

    Reads ``labeled_features.parquet`` (produced by the labels stage), cuts
    it over the date axis at the ``SPLIT_RATIOS`` quantiles (no shuffle),
    validates strict temporal separation, and writes ``split_{train,val,test}``
    plus the split summary report.

    Raises:
        ConflictForecastError: if the labeled dataset is missing or invalid.
    """
    logger = logging_config.get_logger("run_pipeline")
    logger.info("=== Stage: split ===")
    labeled_path = config.DATA_PROCESSED_DIR / f"{config.LABELED_FEATURES_FILE}.parquet"
    if not labeled_path.is_file():
        raise DataLoadError(
            f"Labeled dataset not found at {labeled_path} — run the 'labels' stage first."
        )
    labeled = pd.read_parquet(labeled_path)
    splits = split.chronological_split(labeled)
    written: dict[str, list[Path]] = {}
    for name, part in splits.items():
        fname = f"{config.SPLIT_FILE_PREFIX}_{name}"
        written[name] = data_loader.save_dataframe(
            part, fname, config.DATA_PROCESSED_DIR
        )
    summary_path = split.write_split_summary(
        splits, config.REPORTS_DIR / config.SPLIT_SUMMARY_FILE
    )
    date_col = config.SPLIT_DATE_COLUMN
    summary: dict[str, object] = {
        "total_rows": int(len(labeled)),
        "geo_units": int(labeled["geo_unit"].nunique()),
        "date_start": str(labeled[date_col].min().date()),
        "date_end": str(labeled[date_col].max().date()),
        "splits": {name: int(len(part)) for name, part in splits.items()},
        "cut_dates": {
            name: (
                str(part[date_col].min().date()),
                str(part[date_col].max().date()),
            )
            for name, part in splits.items()
        },
        "written": {name: [str(p) for p in paths] for name, paths in written.items()},
        "summary_report": str(summary_path),
    }
    logger.info("Split stage summary: %s", summary)
    return summary


def run_train_stage() -> dict[str, object]:
    """Train LightGBM on the chronological splits and persist artifacts.

    Delegates to :func:`src.pipeline.train_stage` (plan M8): loads
    ``split_train``/``split_val``, trains with the configured imbalance
    handling, reports validation metrics, and saves
    ``models/escalation_lgbm.pkl`` + ``models/manifest.json``.

    Raises:
        ConflictForecastError: if the splits are missing or training fails.
    """
    logger = logging_config.get_logger("run_pipeline")
    logger.info("=== Stage: train ===")
    summary = pipeline.train_stage()
    logger.info("Train stage summary: %s", summary)
    return summary


def run_compare_stage() -> dict[str, object]:
    """Train XGBoost and run the LightGBM vs XGBoost comparison.

    Delegates to :func:`src.pipeline.compare_stage` (plan M9): verifies the
    saved LightGBM is unchanged, trains XGBoost on identical data, runs the
    full metric comparison + threshold sweep + baselines, picks the winner
    per PRD priority, and writes ``escalation_xgb.pkl``,
    ``escalation_best.pkl``, ``model_comparison.json`` and
    ``reports/model_comparison.md``.

    Raises:
        ConflictForecastError: if the splits or trained model are missing.
    """
    logger = logging_config.get_logger("run_pipeline")
    logger.info("=== Stage: compare ===")
    summary = pipeline.compare_stage()
    logger.info("Compare stage summary: %s", summary)
    return summary


def run_explain_stage() -> dict[str, object]:
    """Run SHAP explainability on the winning model.

    Delegates to :func:`src.explainability.explain_stage` (plan M11 → M10):
    loads ``escalation_best.pkl`` + the test window, computes TreeExplainer
    SHAP values, and writes the summary/bar/waterfall/dependence plots under
    ``reports/shap/`` plus ``reports/shap_summary.md`` with the top-20
    features, interpretations, risk drivers, and local explanations.

    Raises:
        ConflictForecastError: if the model/split is missing or SHAP fails.
    """
    logger = logging_config.get_logger("run_pipeline")
    logger.info("=== Stage: explain ===")
    summary = explainability.explain_stage()
    logger.info("Explain stage summary: %s", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Conflict escalation forecasting pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=("ingest", "features", "labels", "split", "train", "compare", "explain"),
        default="ingest",
        help="Pipeline stage to run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on a handled domain error, 2 on an unexpected error.
    """
    args = parse_args(argv)
    logging_config.setup_logging(log_dir=config.LOGS_DIR, log_file=config.LOGS_FILE)
    try:
        config.validate_config()
        if args.stage == "ingest":
            summary = run_ingest_stage()
        elif args.stage == "features":
            summary = run_features_stage()
        elif args.stage == "labels":
            summary = run_labels_stage()
        elif args.stage == "split":
            summary = run_split_stage()
        elif args.stage == "compare":
            summary = run_compare_stage()
        elif args.stage == "explain":
            summary = run_explain_stage()
        else:
            summary = run_train_stage()
        if args.stage == "train":
            print(
                f"Pipeline stage 'train' complete: validation F1 = "
                f"{summary['validation_metrics']['f1']:.4f} "
                f"(n_train={summary['n_train']}, n_val={summary['n_val']})."
            )
        elif args.stage == "compare":
            print(
                f"Pipeline stage 'compare' complete: winner = {summary['winner']} "
                f"(best val F1 {summary['best_f1']:.4f}), operating threshold = "
                f"{summary['operating_threshold']:.2f}. "
                f"Best model saved to {summary['best_model_file']}."
            )
        elif args.stage == "explain":
            top1 = summary["top_features"][0]
            print(
                f"Pipeline stage 'explain' complete: {len(summary['plots'])} plots + "
                f"{summary['summary_report']}; top driver = "
                f"{top1['feature']} (mean |SHAP| {top1['mean_abs_shap']:.4f})."
            )
        else:
            row_key = {
                "ingest": "cleaned_rows",
                "features": "feature_rows",
                "labels": "labeled_rows",
                "split": "total_rows",
            }[args.stage]
            print(
                f"Pipeline stage '{args.stage}' complete: "
                f"{summary[row_key]} rows, "
                f"{summary['geo_units']} geo units, "
                f"{summary['date_start']}..{summary['date_end']}."
            )
        return 0
    except ConflictForecastError as exc:
        logging_config.get_logger("run_pipeline").error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:
        logging_config.get_logger("run_pipeline").exception("Unexpected failure")
        print("ERROR: unexpected failure — see logs/project.log", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
