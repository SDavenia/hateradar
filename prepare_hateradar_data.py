####
# Build a single per-video dataset from gold + silver annotations.
#
# For every video (matched by <id> within a newspaper) it joins:
#   - annotated_metadata/<newspaper>/<id>_{gold,silver}.json  -> text, topic
#   - annotated_comments/<newspaper>/<id>_{gold,silver}.csv   -> comment counts
#
# Output: one CSV row per video with columns
#   id, newspaper, text, topic, num_comments, percentage_offensive_comments, type
#
# Conventions (matching the rest of the pipeline):
#   - gold wins over silver when both files exist for the same id/source.
#   - text                          = title + "\n\n" + description (full, untruncated)
#   - num_comments                  = number of rows in the comments file
#   - percentage_offensive_comments = label==1 count / num_comments * 100  (0-100,
#                                     denominator = ALL comments; unparseable/empty
#                                     labels stay in the denominator, not the numerator)
#   - type                          = "gold" or "silver" from the file suffix; if the
#                                     metadata and comments suffixes disagree for a
#                                     video, it is recorded as "<meta>/<comments>"
#                                     and a warning is printed.
####

import os
import json
import argparse

import pandas as pd


newspapers = [
    "corriere_della_sera",
    "il_gazzettino",
    "ilmessaggero",
    "lastampa",
    "repubblica",
]

OUTPUT_COLUMNS = [
    "id", "newspaper", "text", "topic",
    "num_comments", "percentage_offensive_comments", "type",
]


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

def read_text_and_topic(meta_path: str):
    """Return (text, topic) from a metadata JSON. text = title + description."""
    with open(meta_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    title = (md.get("title", "") or "").strip()
    description = (md.get("description", "") or "").strip()
    topic = md.get("topic", "")
    text = "\n\n".join(part for part in (title, description) if part)
    return text, topic


def read_comment_stats(comm_path: str):
    """Return (num_comments, percentage_offensive). percentage is None if there are
    no comments; it is a 0-100 float over ALL comment rows otherwise."""
    df = pd.read_csv(comm_path)
    num_comments = len(df)
    if num_comments == 0:
        return 0, None
    if "label" not in df.columns:
        print(f"  WARNING: no 'label' column in {comm_path} — percentage left empty")
        return num_comments, None
    labels = pd.to_numeric(df["label"], errors="coerce")     # empty/unparseable -> NaN
    n_offensive = int((labels == 1).sum())
    percentage = round(n_offensive / num_comments * 100, 2)
    return num_comments, percentage


def resolve_type(meta_type, comm_type, newspaper, file_id):
    """Single gold/silver tag from the two source suffixes; flag disagreements."""
    present = [t for t in (meta_type, comm_type) if t is not None]
    if not present:
        return ""
    if len(set(present)) == 1:
        return present[0]
    print(f"  WARNING: {newspaper}/{file_id} has mismatched sources "
          f"(metadata={meta_type}, comments={comm_type}) — recorded as combined")
    return f"{meta_type}/{comm_type}"


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
            text, topic, meta_type = "", "", None
            if file_id in meta_idx:
                meta_path, meta_type = meta_idx[file_id]
                try:
                    text, topic = read_text_and_topic(meta_path)
                except Exception as e:
                    print(f"  ERROR reading metadata {newspaper}/{file_id}: {e}")
            else:
                n_no_meta += 1
                print(f"  WARNING: {newspaper}/{file_id} has comments but no metadata "
                      f"— text/topic left empty")

            num_comments, percentage = 0, None
            comm_type = None
            if file_id in comm_idx:
                comm_path, comm_type = comm_idx[file_id]
                try:
                    num_comments, percentage = read_comment_stats(comm_path)
                except Exception as e:
                    print(f"  ERROR reading comments {newspaper}/{file_id}: {e}")
            else:
                n_no_comments += 1
                print(f"  WARNING: {newspaper}/{file_id} has metadata but no comments "
                      f"— num_comments=0, percentage empty")

            rows.append({
                "id": file_id,
                "newspaper": newspaper,
                "text": text,
                "topic": topic,
                "num_comments": num_comments,
                "percentage_offensive_comments": percentage,
                "type": resolve_type(meta_type, comm_type, newspaper, file_id),
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
        print(f"  by type: {df['type'].value_counts().to_dict()}")
        print(f"  total comments: {int(df['num_comments'].sum())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()