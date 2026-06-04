####
# Offensiveness-TRIGGERING prediction from video metadata (non-learned baselines).
#
# Five reference baselines for the same binary task (predict `offensive_trigger`):
#   - random                : fully random labels, p(1)=0.5. 5 runs + mean/std.
#   - topic_majority         : assign every video the most common label in its topic.
#   - topic_majority_random  : within each topic, sample label 1 with probability
#                             equal to that topic's positive rate (e.g. a topic
#                             that is 60% offensive -> p(1)=0.6). 5 runs + mean/std.
#   - all_positive           : assign label 1 (triggering) to every video.
#   - all_negative           : assign label 0 (non-triggering) to every video.
#
# Output layout mirrors the LLM pipeline: results/<baseline>/<tag>.csv + <tag>.txt.
####

import os
import argparse

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

N_RUNS = 5          # random assignments for the random / topic_majority_random baselines
SEED = 42           # base seed; run i uses SEED + i

TOPIC_COL = "video_topic"

# `offensive_trigger` is the binary ground truth (1 = video triggers offensive
# comments, 0 = it does not). `video_topic` is needed by the topic baselines.
REQUIRED_COLUMNS = ["offensive_trigger"]


@dataclass
class BaselineSpec:
    name: str
    kind: str                    # "random" | "topic_majority" | "topic_random" | "constant"
    n_runs: int = 1              # >1 for the random baselines
    const_value: Optional[int] = None   # 0/1 for the constant baselines
    needs_topic: bool = False


# Analogous to the LLM REGISTRY: pick what to run with --baselines.
REGISTRY = {
    "random": BaselineSpec("random", kind="random", n_runs=N_RUNS),
    "topic_majority": BaselineSpec("topic_majority", kind="topic_majority", needs_topic=True),
    "topic_majority_random": BaselineSpec(
        "topic_majority_random", kind="topic_random", n_runs=N_RUNS, needs_topic=True
    ),
    "all_positive": BaselineSpec("all_positive", kind="constant", const_value=1),
    "all_negative": BaselineSpec("all_negative", kind="constant", const_value=0),
}


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def load_dataset(input_path: str, type_filter: str = "all", need_topic: bool = False):
    """Read the per-video CSV, optionally filter by gold/silver, and use the
    dataset's `offensive_trigger` column directly as the binary ground truth.
    Videos whose label is missing (undefined ground truth) are dropped."""
    df = pd.read_csv(input_path)

    required = list(REQUIRED_COLUMNS) + ([TOPIC_COL] if need_topic else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if type_filter != "all":
        if "type" not in df.columns:
            raise KeyError("--type filter requested but no 'type' column in the dataset.")
        before = len(df)
        df = df[df["type"].astype(str) == type_filter].copy()
        print(f"Type filter '{type_filter}': {len(df)}/{before} videos kept")

    # Label comes straight from the dataset (no tau / percentage thresholding).
    label = pd.to_numeric(df["offensive_trigger"], errors="coerce")
    n_dropped = int(label.isna().sum())
    if n_dropped:
        print(f"Dropping {n_dropped} videos with no offensive_trigger label — undefined ground truth")
    df = df[label.notna()].copy()
    df["true_label"] = pd.to_numeric(df["offensive_trigger"], errors="coerce").astype(int)

    n_pos = int(df["true_label"].sum())
    print(f"Loaded {len(df)} videos | triggering (label 1): {n_pos} | "
          f"non-triggering (label 0): {len(df) - n_pos}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prediction  (baseline analog of the LLM inference step)
# --------------------------------------------------------------------------- #

def predict_random(n: int, seed: int) -> list:
    """Fully random labels, each drawn 0/1 with p=0.5."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n).tolist()


def predict_topic_random(df: pd.DataFrame, seed: int, topic_col: str = TOPIC_COL) -> list:
    """Within each topic, sample label 1 with probability equal to that topic's
    positive rate. Topics unseen at predict time fall back to the global rate.
    Fit on the full data (optimistic); see --topic-cv for a leakage-free variant."""
    rng = np.random.default_rng(seed)
    topic_rate = df.groupby(topic_col)["true_label"].mean().to_dict()
    global_rate = float(df["true_label"].mean())
    probs = df[topic_col].map(topic_rate).fillna(global_rate).to_numpy(dtype=float)
    return (rng.random(len(df)) < probs).astype(int).tolist()


def predict_topic_random_cv(df: pd.DataFrame, n_splits: int, seed: int,
                            topic_col: str = TOPIC_COL) -> list:
    """Leakage-free variant: per-topic positive rate is estimated on the train
    fold and used to sample labels for the held-out fold (out-of-fold preds)."""
    y = df["true_label"].values
    preds = np.empty(len(df), dtype=int)
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(df, y):
        train = df.iloc[train_idx]
        topic_rate = train.groupby(topic_col)["true_label"].mean().to_dict()
        global_rate = float(train["true_label"].mean())
        probs = df.iloc[test_idx][topic_col].map(topic_rate).fillna(global_rate).to_numpy(dtype=float)
        preds[test_idx] = (rng.random(len(test_idx)) < probs).astype(int)
    return preds.tolist()


def predict_topic_majority(df: pd.DataFrame, topic_col: str = TOPIC_COL):
    """Most common label within each topic (ties -> global majority).
    Fit and evaluated on the full data: simple and standard, but optimistic
    (label leakage). Use predict_topic_majority_cv for an honest estimate."""
    global_majority = int(df["true_label"].mode().iloc[0])
    topic_majority = (
        df.groupby(topic_col)["true_label"]
        .agg(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else global_majority)
        .to_dict()
    )
    preds = df[topic_col].map(topic_majority).fillna(global_majority).astype(int)
    return preds.tolist(), topic_majority


def predict_topic_majority_cv(df: pd.DataFrame, n_splits: int, seed: int,
                              topic_col: str = TOPIC_COL) -> list:
    """Leakage-free variant: per-topic majority is computed on the train fold
    and applied to the held-out fold; returns out-of-fold predictions."""
    y = df["true_label"].values
    preds = np.empty(len(df), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(df, y):
        train = df.iloc[train_idx]
        global_majority = int(train["true_label"].mode().iloc[0])
        tm = (
            train.groupby(topic_col)["true_label"]
            .agg(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else global_majority)
            .to_dict()
        )
        fold = df.iloc[test_idx][topic_col].map(tm).fillna(global_majority).astype(int)
        preds[test_idx] = fold.values
    return preds.tolist()


# --------------------------------------------------------------------------- #
# Scoring & output
# --------------------------------------------------------------------------- #

TARGET_NAMES = ["non-triggering (No)", "triggering (Si)"]


def compute_metrics(y_true, y_pred) -> dict:
    """Per-class + macro/weighted precision/recall/f1 (sklearn)."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    _, _, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="weighted", zero_division=0
    )
    return {
        "prec_neg": p[0], "rec_neg": r[0], "f1_neg": f[0],
        "prec_pos": p[1], "rec_pos": r[1], "f1_pos": f[1],
        "macro_prec": pm, "macro_rec": rm, "macro_f1": fm,
        "weighted_f1": fw,
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    }


def write_outputs(out_dir: str, baseline_key: str, tag: str,
                  df: pd.DataFrame, preds: list, header_extra: list = None):
    """Write results/<baseline>/<tag>.csv (all input columns + prediction)
    and .txt (classification_report + macro recap), mirroring the LLM pipeline."""
    bdir = os.path.join(out_dir, baseline_key)
    os.makedirs(bdir, exist_ok=True)
    csv_path = os.path.join(bdir, f"{tag}.csv")
    txt_path = os.path.join(bdir, f"{tag}.txt")

    out_df = df.drop(columns=["true_label"], errors="ignore").copy()
    out_df["pred_label"] = preds
    out_df.to_csv(csv_path, index=False)

    y_true = df["true_label"].tolist()
    n_total = len(preds)
    n_pos = int(df["true_label"].sum())

    lines = [
        f"Baseline: {baseline_key}",
        f"Run:      {tag}",
        f"Label:    offensive_trigger (taken directly from the dataset)",
        f"Videos evaluated:   {n_total}",
        f"Label distribution: triggering (1): {n_pos} | non-triggering (0): {n_total - n_pos}",
    ]
    if header_extra:
        lines.extend(header_extra)
    lines.append("")

    report = classification_report(
        y_true, preds, labels=[0, 1],
        target_names=TARGET_NAMES, digits=4, zero_division=0,
    )
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], average="macro", zero_division=0,
    )
    lines.append(report)
    lines.append("")
    lines.append(f"Macro-averaged -> precision: {p:.4f} | recall: {r:.4f} | f1: {f1:.4f}")

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  -> {csv_path}")
    print(f"  -> {txt_path}")


def write_aggregate(out_dir: str, baseline_key: str, run_metrics: list):
    """Mean +/- std across the random runs."""
    bdir = os.path.join(out_dir, baseline_key)
    os.makedirs(bdir, exist_ok=True)
    txt_path = os.path.join(bdir, "aggregate.txt")

    keys = run_metrics[0].keys()
    lines = [
        f"Baseline: {baseline_key}",
        f"Aggregate over {len(run_metrics)} random runs (mean +/- std)",
        "",
    ]
    for k in keys:
        vals = [m[k] for m in run_metrics]
        lines.append(f"  {k:12s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  -> {txt_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(
        description="Non-learned baselines for offensiveness-triggering prediction."
    )
    ap.add_argument("--input", default="video_dataset.csv",
                    help="Per-video dataset CSV (must contain offensive_trigger; "
                         "video_topic is needed by the topic baselines).")
    ap.add_argument("--out-dir", default="results_baselines",
                    help="Root output directory (default: results_baselines).")
    ap.add_argument("--baselines", nargs="+", choices=list(REGISTRY), default=list(REGISTRY),
                    help="Which baselines to run (default: all in the registry).")
    ap.add_argument("--type", choices=["all", "gold", "silver"], default="all",
                    help="Restrict ground truth to gold/silver videos (default: all).")
    ap.add_argument("--topic-cv", type=int, default=0,
                    help="If >1, evaluate the topic baselines with K-fold CV (leakage-free) "
                         "instead of fitting on the full data (default: 0 = full data).")
    return ap.parse_args()


def main():
    args = parse_args()
    need_topic = any(REGISTRY[b].needs_topic for b in args.baselines)
    df = load_dataset(args.input, type_filter=args.type, need_topic=need_topic)
    if df.empty:
        print("No videos to evaluate — aborting.")
        return
    y = df["true_label"].tolist()

    for baseline_key in args.baselines:
        spec = REGISTRY[baseline_key]
        print(f"\n{'#'*60}\nBaseline: {baseline_key}\n{'#'*60}")

        if spec.kind == "random":
            run_metrics = []
            for i in range(spec.n_runs):
                print(f"\n[{baseline_key}] run {i}")
                preds = predict_random(len(df), seed=SEED + i)
                write_outputs(args.out_dir, baseline_key, f"run_{i}", df, preds,
                              header_extra=[f"Mode: fully random, p(1)=0.5 | seed: {SEED + i}"])
                run_metrics.append(compute_metrics(y, preds))
            print(f"\n[{baseline_key}] aggregate")
            write_aggregate(args.out_dir, baseline_key, run_metrics)

        elif spec.kind == "topic_random":
            run_metrics = []
            for i in range(spec.n_runs):
                print(f"\n[{baseline_key}] run {i}")
                if args.topic_cv and args.topic_cv > 1:
                    preds = predict_topic_random_cv(df, n_splits=args.topic_cv, seed=SEED + i)
                    mode = f"{args.topic_cv}-fold CV (out-of-fold)"
                else:
                    preds = predict_topic_random(df, seed=SEED + i)
                    mode = "full-data (optimistic; see --topic-cv)"
                write_outputs(args.out_dir, baseline_key, f"run_{i}", df, preds,
                              header_extra=[f"Mode: per-topic random sampling, {mode} | seed: {SEED + i}"])
                run_metrics.append(compute_metrics(y, preds))
            print(f"\n[{baseline_key}] aggregate")
            write_aggregate(args.out_dir, baseline_key, run_metrics)

        elif spec.kind == "topic_majority":
            if args.topic_cv and args.topic_cv > 1:
                print(f"\n[{baseline_key}] {args.topic_cv}-fold CV (leakage-free)")
                preds = predict_topic_majority_cv(df, n_splits=args.topic_cv, seed=SEED)
                write_outputs(args.out_dir, baseline_key, "predictions", df, preds,
                              header_extra=[f"Mode: {args.topic_cv}-fold stratified CV (out-of-fold preds)"])
            else:
                print(f"\n[{baseline_key}] full-data majority per topic")
                preds, topic_majority = predict_topic_majority(df)
                tm_lines = ["Mode: full-data (optimistic; see --topic-cv for leakage-free)",
                            "Per-topic majority label:"]
                tm_lines += [f"  {t} -> {lab}" for t, lab in topic_majority.items()]
                write_outputs(args.out_dir, baseline_key, "predictions", df, preds,
                              header_extra=tm_lines)

        elif spec.kind == "constant":
            label_name = "1 (triggering)" if spec.const_value == 1 else "0 (non-triggering)"
            print(f"\n[{baseline_key}] assign {label_name} to every video")
            preds = [spec.const_value] * len(df)
            write_outputs(args.out_dir, baseline_key, "predictions", df, preds,
                          header_extra=[f"Mode: constant prediction (label {spec.const_value} for all)"])

    print(f"\nDone. Results under '{args.out_dir}/'")


if __name__ == "__main__":
    main()