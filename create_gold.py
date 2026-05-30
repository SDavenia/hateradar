# 1. Read all annotators files in their respective folders
# 2. Each file should be annotated three times
# 3. Do majority vote on labels 
# 4. Measure agreement between annotators (average Cohen's Kappa, Fleiss' Kappa, Krippendorff alpha)
# 5. Save to VideosComments and add a column with type=gold, indicating gold label standard

import os
import pandas as pd

# Measure annotator agreements 
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import krippendorff

use_fake_data = True # Only now for testing


input_dir = "Annotators_workload_fake" if use_fake_data else "Annotators_workload"
output_dir = "VideosComments_fake" if use_fake_data else "VideosComments"

annotators = ["A1", "A2", "A3", "A4", "A5"]

newspapers = [
    "corriere_della_sera",
    "il_gazzettino",
    "ilmessaggero",
    "lastampa",
    "repubblica"
]

############## For comments ##############
files_annotators_paths = {}
for annotator in annotators:
    for newspaper in newspapers:
        input_annotator_newspaper_dir = os.path.join(input_dir, annotator, "VideosComments/youtube/annotated_comments", newspaper)
        for filename in os.listdir(input_annotator_newspaper_dir):
            if filename.endswith(".csv"):
                input_file_path = os.path.join(input_annotator_newspaper_dir, filename)
                file_id = filename.split(".")[0]
                if file_id not in files_annotators_paths:
                    files_annotators_paths[file_id] = []
                files_annotators_paths[file_id].append((input_file_path, newspaper))  # ← store newspaper

assert all(len(paths) == 3 for paths in files_annotators_paths.values()), "Not all files have 3 annotations!"
annotator_full_labels = {ann: [] for ann in annotators}

for file_id, paths in files_annotators_paths.items():
    paths_only = [p for p, _ in paths]                         # ← unpack
    newspaper = paths[0][1]                                    # ← recover newspaper
    file_annotators = [p.split("/")[1] for p in paths_only]
    dfs = [pd.read_csv(p) for p in paths_only]

    for annotator in file_annotators:
        annotator_full_labels[annotator].extend(dfs[file_annotators.index(annotator)]["label"].tolist())
    for annotator in annotators:
        if annotator not in file_annotators:
            annotator_full_labels[annotator].extend([None] * len(dfs[0]))

    df_merged = pd.concat(dfs)
    df_majority = df_merged.groupby("comment_id")["label"].agg(lambda x: x.mode()[0]).reset_index()

    # Attach majority label back onto the full columns from the first annotator's file
    df_gold = dfs[0].drop(columns=["label"]).merge(df_majority, on="comment_id", how="left")
    df_gold["type"] = "gold"

    os.makedirs(os.path.join(output_dir, "youtube", "annotated_comments", newspaper), exist_ok=True)
    output_file_path = os.path.join(output_dir, "youtube", "annotated_comments", newspaper, f"{file_id}_gold.csv")
    df_gold.to_csv(output_file_path, index=False)

df = pd.DataFrame(annotator_full_labels)
print(df)

# Cohen's Kappa (average pairwise)
all_cohen_kappas = []
for a1, a2 in combinations(df.columns, 2):
    pair = df[[a1, a2]].dropna()
    kappa = cohen_kappa_score(pair[a1], pair[a2])
    print(f"Cohen's Kappa {a1} vs {a2}: {kappa:.3f}")
    all_cohen_kappas.append(kappa)

print(f"Average Cohen's Kappa: {sum(all_cohen_kappas)/len(all_cohen_kappas):.3f}")

# Fleiss
categories = sorted(df.stack().dropna().unique())  # infer categories from data
fleiss_matrix = np.array([
    [row.dropna().tolist().count(c) for c in categories]
    for _, row in df.iterrows()
])
fk = fleiss_kappa(fleiss_matrix)
print(f"Fleiss' Kappa: {fk:.3f}")

# krippendorff 
reliability_matrix = df.to_numpy(dtype=float).T  # shape: (n_annotators, n_items)
alpha = krippendorff.alpha(reliability_data=reliability_matrix, level_of_measurement="nominal")
print(f"Krippendorff's Alpha: {alpha:.3f}")


############## For labels ##############
############## For metadata (topic) ##############
import json
from collections import Counter

files_meta_paths = {}  # video_id -> list of (path, newspaper)
for annotator in annotators:
    for newspaper in newspapers:
        input_annotator_newspaper_dir = os.path.join(input_dir, annotator, "VideosComments/youtube/annotated_metadata", newspaper)
        if not os.path.exists(input_annotator_newspaper_dir):
            continue
        for filename in os.listdir(input_annotator_newspaper_dir):
            if filename.endswith(".json"):
                input_file_path = os.path.join(input_annotator_newspaper_dir, filename)
                video_id = filename.split(".")[0]
                if video_id not in files_meta_paths:
                    files_meta_paths[video_id] = []
                files_meta_paths[video_id].append((input_file_path, newspaper))

assert all(len(paths) == 3 for paths in files_meta_paths.values()), "Not all metadata files have 3 annotations!"

meta_annotator_full_labels = {ann: [] for ann in annotators}
three_way_disagreements = []  # video_ids where all 3 annotators disagreed

for video_id, paths in files_meta_paths.items():
    paths_only = [p for p, _ in paths]
    newspaper = paths[0][1]
    file_annotators = [p.split("/")[1] for p in paths_only]
    jsons = [json.load(open(p)) for p in paths_only]

    topics = [j["topic"] for j in jsons]

    # Track per-annotator labels (same pattern as comments)
    for annotator in file_annotators:
        meta_annotator_full_labels[annotator].append(topics[file_annotators.index(annotator)])
    for annotator in annotators:
        if annotator not in file_annotators:
            meta_annotator_full_labels[annotator].append(None)

    # Majority vote; if all 3 differ, no majority exists — use None and log it
    topic_counts = Counter(topics)
    majority_topic, majority_count = topic_counts.most_common(1)[0]

    if majority_count == 1:  # all three are different
        majority_topic = None
        three_way_disagreements.append({
            "video_id": video_id,
            "newspaper": newspaper,
            "topics": dict(zip(file_annotators, topics)),
        })

    # Build output JSON from first annotator's file as base, override topic
    output_json = dict(jsons[0])
    output_json["topic"] = majority_topic
    output_json["type"] = "gold"

    output_meta_dir = os.path.join(output_dir, "youtube", "annotated_metadata", newspaper)
    os.makedirs(output_meta_dir, exist_ok=True)
    output_file_path = os.path.join(output_meta_dir, f"{video_id}_gold.json")
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

# Agreement metrics for metadata topics
df_meta = pd.DataFrame(meta_annotator_full_labels)
print("\n--- Metadata (topic) annotator agreement ---")
print(df_meta)

all_cohen_kappas_meta = []
for a1, a2 in combinations(df_meta.columns, 2):
    pair = df_meta[[a1, a2]].dropna()
    kappa = cohen_kappa_score(pair[a1], pair[a2])
    print(f"Cohen's Kappa {a1} vs {a2}: {kappa:.3f}")
    all_cohen_kappas_meta.append(kappa)

print(f"Average Cohen's Kappa: {sum(all_cohen_kappas_meta)/len(all_cohen_kappas_meta):.3f}")

categories_meta = sorted(df_meta.stack().dropna().unique())
fleiss_matrix_meta = np.array([
    [row.dropna().tolist().count(c) for c in categories_meta]
    for _, row in df_meta.iterrows()
])
print(f"Fleiss' Kappa: {fleiss_kappa(fleiss_matrix_meta):.3f}")

reliability_matrix_meta = np.array([
    [categories_meta.index(v) if pd.notna(v) else np.nan for v in df_meta[ann]]
    for ann in annotators
])
alpha_meta = krippendorff.alpha(reliability_data=reliability_matrix_meta, level_of_measurement="nominal")
print(f"Krippendorff's Alpha: {alpha_meta:.3f}")

# Report three-way disagreements
print(f"\nThree-way disagreements (no majority): {len(three_way_disagreements)}")
for d in three_way_disagreements:
    print(f"  {d['video_id']} ({d['newspaper']}): {d['topics']}")