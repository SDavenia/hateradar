####
# Build a single per-video dataset from gold + silver annotations.
#
# For every video (matched by <id> within a newspaper) it joins:
#   - annotated_metadata/<newspaper>/<id>_{gold,silver}.json  -> title, description, topic
#   - annotated_comments/<newspaper>/<id>_{gold,silver}.csv   -> comment counts
#
# Output: one CSV row per video with columns
#   video_id, newspaper, video_text, video_description, video_topic,
#   num_comments, num_offensive_comments, offensive_score, offensive_trigger
#
# Conventions (matching the rest of the pipeline):
#   - gold wins over silver when both files exist for the same id/source.
#   - video_text         = the video title.
#   - video_description  = the video description.
#   - offensive_score    = arctan((n_off/n) + lambda*ln(n_off)) / (pi/2), with
#                          lambda = 1; 0.0 when there are no comments or no
#                          offensive comments. Denominator n = ALL comments
#                          (unparseable/empty labels stay in n, not in n_off).
#   - offensive_trigger  = 1 if offensive_score > 0.7, else 0.
#
# Comment files with no rows (including completely empty files) are still
# included as a video with num_comments = 0 -> offensive_score = 0.0.
####

import os
import json
import argparse

import numpy as np
import pandas as pd


newspapers = [
    "corriere_della_sera",
    "il_gazzettino",
    "ilmessaggero",
    "lastampa",
    "repubblica",
]

OUTPUT_COLUMNS = [
    "video_id", "newspaper", "video_text", "video_description", "video_topic",
    "num_comments", "num_offensive_comments", "offensive_score", "offensive_trigger",
]

OFFENSIVE_THRESHOLD = 0.7
LAMBDA = 1.0


# --------------------------------------------------------------------------- #
# Offensive score
# --------------------------------------------------------------------------- #

def offensive_score(n, n_off, lambda1=1.0):
    if n == 0:
        return 0.0
    if n_off == 0:
        return 0.0
    return np.arctan((n_off / n) + lambda1 * np.log(n_off)) / (np.pi / 2)


# --------------------------------------------------------------------------- #
# Filename indexing
# --------------------------------------------------------------------------- #

def parse_annotated_filename(filename: str):
    """'<id>_gold.json' -> ('<id>', 'gold'); same for _silver and .csv. None otherwise."""
    for ext in (".json", ".csv"):
        for suffix, kind in (("_gold", "gold"), ("_silver", "silver")):
            tag = suffix + ext
            if filename.endswith(tag):
                return filename[:-len(tag)], kind
    return None


def index_annotated(dir_path: str) -> dict:
    """Map id -> (path, type) for one newspaper dir; gold beats silver on conflict."""
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for fn in sorted(os.listdir(dir_path)):
        parsed = parse_annotated_filename(fn)
        if parsed is None:
            continue
        file_id, kind = parsed
        if file_id in out and out[file_id][1] == "gold":
            continue                                   # gold already claimed it
        if file_id not in out or kind == "gold":
            out[file_id] = (os.path.join(dir_path, fn), kind)
    return out


# --------------------------------------------------------------------------- #
# Per-video extraction
# --------------------------------------------------------------------------- #

def read_meta(meta_path: str):
    """Return (title, description, topic) from a metadata JSON."""
    with open(meta_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    title = (md.get("title", "") or "").strip()
    description = (md.get("description", "") or "").strip()
    topic = md.get("topic", "")
    return title, description, topic


def read_comment_stats(comm_path: str):
    """Return (num_comments, n_offensive). Files with no rows (including
    completely empty files) yield (0, 0). n_offensive counts label == 1 over
    ALL rows; unparseable/empty labels stay in the denominator, not here."""
    try:
        df = pd.read_csv(comm_path)
    except pd.errors.EmptyDataError:
        return 0, 0                                    # empty file -> 0 counts
    num_comments = len(df)
    if num_comments == 0:
        return 0, 0
    if "label" not in df.columns:
        print(f"  WARNING: no 'label' column in {comm_path} — treated as 0 offensive")
        return num_comments, 0
    labels = pd.to_numeric(df["label"], errors="coerce")     # empty/unparseable -> NaN
    n_offensive = int((labels == 1).sum())
    return num_comments, n_offensive


def build_dataset(annotated_metadata_dir: str, annotated_comments_dir: str) -> pd.DataFrame:
    rows = []
    n_no_meta = n_no_comments = 0

    for newspaper in newspapers:
        meta_idx = index_annotated(os.path.join(annotated_metadata_dir, newspaper))
        comm_idx = index_annotated(os.path.join(annotated_comments_dir, newspaper))

        all_ids = sorted(set(meta_idx) | set(comm_idx))
        if not all_ids:
            print(f"WARNING: nothing found for newspaper '{newspaper}'")
            continue

        for file_id in all_ids:
            title, description, topic = "", "", ""
            if file_id in meta_idx:
                meta_path, _ = meta_idx[file_id]
                try:
                    title, description, topic = read_meta(meta_path)
                except Exception as e:
                    print(f"  ERROR reading metadata {newspaper}/{file_id}: {e}")
            else:
                n_no_meta += 1
                print(f"  WARNING: {newspaper}/{file_id} has comments but no metadata "
                      f"— text/description/topic left empty")

            num_comments, n_offensive = 0, 0
            if file_id in comm_idx:
                comm_path, _ = comm_idx[file_id]
                try:
                    num_comments, n_offensive = read_comment_stats(comm_path)
                except Exception as e:
                    print(f"  ERROR reading comments {newspaper}/{file_id}: {e}")
            else:
                n_no_comments += 1
                print(f"  WARNING: {newspaper}/{file_id} has metadata but no comments "
                      f"— offensive_score = 0.0")

            score = offensive_score(num_comments, n_offensive, lambda1=LAMBDA)

            rows.append({
                "video_id": file_id,
                "newspaper": newspaper,
                "video_text": title,
                "video_description": description,
                "video_topic": topic,
                "num_comments": num_comments,
                "num_offensive_comments": n_offensive,
                "offensive_score": round(score, 4),
                "offensive_trigger": 1 if score > OFFENSIVE_THRESHOLD else 0,
            })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if n_no_meta or n_no_comments:
        print(f"\nMissing counterparts -> videos without metadata: {n_no_meta}, "
              f"without comments: {n_no_comments}")
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Build a per-video dataset from gold + silver annotations.")
    ap.add_argument("--use-fake-data", action="store_true",
                    help="Use the VideosComments_fake tree instead of VideosComments.")
    ap.add_argument("--out", default="video_dataset.csv",
                    help="Output CSV path (default: video_dataset.csv).")
    return ap.parse_args()


def main():
    args = parse_args()
    base = "VideosComments_fake" if args.use_fake_data else "VideosComments"
    annotated_metadata_dir = f"{base}/youtube/annotated_metadata"
    annotated_comments_dir = f"{base}/youtube/annotated_comments"

    df = build_dataset(annotated_metadata_dir, annotated_comments_dir)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*60}")
    print(f"Wrote {len(df)} videos -> {args.out}")
    if len(df):
        print(f"  offensive triggers (score > {OFFENSIVE_THRESHOLD}): "
              f"{int(df['offensive_trigger'].sum())}")
        print(f"  total comments: {int(df['num_comments'].sum())}")
        print(f"  total offensive comments: {int(df['num_offensive_comments'].sum())}")
        print(f"  mean offensive_score: {df['offensive_score'].mean():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()