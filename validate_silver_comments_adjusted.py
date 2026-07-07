####
# Offensive-comment classification via zero-shot prompting.
# 1. Load gold annotated data (+ build per-comment context and prompt).
# 2. Split gold data 60/40 into dev/test (stratified on label x depth).
# 3. Score each model on dev via the logits of the Sì/No answer tokens
#    (instead of free-generation + regex parsing); optionally calibrate
#    those logits against a content-free version of the prompt; optionally
#    tune a decision threshold per model on dev; select the best model and
#    evaluate it on test using the threshold locked in from dev.
####

import os
import json
import gc
import re
import argparse

from dotenv import load_dotenv

import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
    # AutoModelForVision2Seq,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BATCH_SIZE = 2         # per-batch generation size; 8-16 is safe for 24-32B in bf16
MAX_LENGTH = 4096        # truncation ceiling for the tokenized prompt

ANNOTATION_PROMPT = """
Sei un annotatore esperto nel riconoscimento di linguaggio offensivo nei commenti online in italiano.

Il tuo compito è determinare se il COMMENTO TARGET potrebbe ragionevolmente essere considerato offensivo da qualcuno.

Devi annotare SOLO il COMMENTO TARGET, utilizzando eventuali informazioni contestuali esclusivamente per interpretarne meglio significato, tono, riferimenti e intenzione comunicativa.

DEFINIZIONE

Un commento è considerato offensivo se contiene linguaggio volgare, insultante, aggressivo, denigratorio, discriminatorio, minaccioso o gravemente irrispettoso, in modo esplicito o implicito, verso una persona o un gruppo.

Il criterio centrale è l’offensività potenziale:
non devi valutare se il commento offende te personalmente, ma se qualcuno potrebbe ragionevolmente percepirlo come offensivo.

TASK DI ANNOTAZIONE

Leggi attentamente tutto il contesto disponibile.

Poi rispondi alla seguente domanda:

"Questo COMMENTO TARGET potrebbe ragionevolmente essere considerato offensivo da qualcuno?"

Restituisci SOLO una delle seguenti etichette:

- Sì
- No

Non fornire spiegazioni o testo aggiuntivo.

ISTRUZIONI IMPORTANTI

- Annota SOLO il COMMENTO TARGET.
- Usa il titolo e la descrizione del video esclusivamente per comprendere il tema, il contesto e i riferimenti del commento.
- Usa eventuali commenti precedenti esclusivamente per comprendere tono, sarcasmo, riferimenti, bersagli o intenzioni.
- Non annotare il titolo del video, la descrizione o i commenti precedenti.
- Considera sia l’offensività esplicita sia quella implicita.
- Valuta insieme tono, intenzione, linguaggio e contesto conversazionale.
- Se un commento contiene sarcasmo, ironia, derisione o ostilità implicita che potrebbe ragionevolmente risultare offensiva, etichettalo come offensivo.
- In caso di dubbio, considera l’effetto complessivo del messaggio piuttosto che singole parole isolate.
- In caso di dubbio, preferisci "Sì" se una persona ragionevole potrebbe percepire il commento come offensivo.

ETICHETTA "Sì" SE IL COMMENTO TARGET:

- Contiene insulti, parolacce, epiteti o linguaggio denigratorio
- Attacca, umilia, deride o svaluta una persona o un gruppo
- Usa linguaggio volgare con intento aggressivo o ostile
- Esprime odio, disprezzo o forte mancanza di rispetto
- Colpisce etnia, nazionalità, genere, religione, orientamento sessuale, disabilità, appartenenza politica o caratteristiche simili
- Contiene minacce, intimidazioni o auguri di danno
- Usa sarcasmo o ironia con effetto offensivo
- Potrebbe essere percepito come offensivo da almeno una persona ragionevole

ETICHETTA "No" SE IL COMMENTO TARGET:

- Esprime disaccordo o critica in modo rispettoso
- Usa linguaggio informale o colloquiale senza aggressività
- Discute temi sensibili in modo neutro o analitico
- Riporta termini offensivi senza approvarli
- Contiene espressioni emotive leggere senza intento offensivo
- È solo vagamente scortese o ambiguo senza offensività chiara

GESTIONE DEL CONTESTO

L’input può contenere:

- VIDEO TITLE: titolo del video
- VIDEO DESCRIPTION: descrizione del video
- HEAD COMMENT: commento principale della conversazione
- PREVIOUS COMMENT: commento immediatamente precedente
- TARGET COMMENT: commento da annotare

Alcuni campi possono essere vuoti se non disponibili.

Regole:

- Usa il contesto solo per interpretare correttamente il COMMENTO TARGET.
- Un commento apparentemente neutro può diventare offensivo nel contesto della conversazione.
- Una parola apparentemente offensiva può essere neutra a seconda del contesto.
- L’etichetta finale deve riferirsi esclusivamente al COMMENTO TARGET.

FORMATO INPUT

VIDEO TITLE:
{video_title}

VIDEO DESCRIPTION:
{video_description}

HEAD COMMENT:
{head_comment}

PREVIOUS COMMENT:
{previous_comment}

TARGET COMMENT:
{target_comment}

FORMATO OUTPUT

Restituisci SOLO:

Sì

oppure

No
"""

newspapers = [
    "corriere_della_sera",
    "il_gazzettino",
    "ilmessaggero",
    "lastampa",
    "repubblica",
]


@dataclass
class ModelSpec:
    name: str
    model_class: type
    proc_class: type
    is_vlm: bool
    supports_system_role: bool      # kept as documentation; unused while we leave
                                    # the system prompt at each model's default
    max_new_tokens: int = 5         # NOTE: no longer used by the scoring path below,
                                    # which always forces exactly 1 new token so it can
                                    # read the Sì/No logits directly. Kept for reference
                                    # / in case a free-generation debug run is wanted.
    load_kwargs: dict = field(default_factory=dict)


# Explicit classes per model card (more robust than the version-dependent
# Auto* umbrellas). Add quantization etc. via a single model's load_kwargs.
REGISTRY = {
    # "gemma2": ModelSpec(
    #     "google/gemma-2-27b-it",
    #     AutoModelForCausalLM, AutoTokenizer,
    #     is_vlm=False, supports_system_role=False,   # template likely rejects system role
    # ),
    # "gemma3": ModelSpec(
    #     "google/gemma-3-27b-it",
    #     Gemma3ForConditionalGeneration, AutoProcessor,
    #     is_vlm=True, supports_system_role=True,
    # ),
    # "mistral": ModelSpec(
    #     "mistralai/Mistral-Small-24B-Instruct-2501",
    #     AutoModelForCausalLM, AutoTokenizer,
    #     is_vlm=False, supports_system_role=True,
    # ),
    "llama": ModelSpec(
        "meta-llama/Llama-3.2-1B-Instruct",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=True,
    ),

    # "qwen_vl": ModelSpec( -> DID NOT MANAGE TO ISNTALL TORCHVISION
    #     "Qwen/Qwen2.5-VL-32B-Instruct",
    #     Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    #     is_vlm=True, supports_system_role=True,
    # ),
    # "anita": ModelSpec(
    #     "m-polignano/ANITA-NEXT-24B-Magistral-2506-VISION-ITA",
    #     AutoModelForVision2Seq, AutoProcessor,
    #     is_vlm=True, supports_system_role=True,
    #     max_new_tokens=1024,                         
    # ),
}

GLOBAL_LOAD_KWARGS = dict(
    torch_dtype=torch.bfloat16,     # if transformers warns, rename to dtype=...
    device_map="auto",              # shards big models across visible GPUs
)

# TODO: Change back
GLOBAL_LOAD_KWARGS = dict(
    torch_dtype=torch.float32,     # if transformers warns, rename to dtype=...
    #device_map="auto",              # shards big models across visible GPUs
)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def add_context_columns(df_gold):
    """
    Add head_comment_text and previous_comment_text columns based on depth:
      depth 0 : both None
      depth 1 : head = depth-0 ancestor's text, previous = None
      depth 2+: head = depth-0 ancestor's text, previous = immediate parent's text
    """
    id_to_text = df_gold.set_index("comment_id")["text"].to_dict()
    id_to_parent = df_gold.set_index("comment_id")["inferred_parent_id"].to_dict()

    def get_head_text(row):
        if row["depth"] == 0:
            return None
        cid = row["comment_id"]
        seen = set()
        while cid in id_to_parent and pd.notna(id_to_parent.get(cid)):
            if cid in seen:
                break
            seen.add(cid)
            cid = id_to_parent[cid]
        return id_to_text.get(cid)

    def get_previous_text(row):
        if row["depth"] <= 1:
            return None
        return id_to_text.get(row["inferred_parent_id"])

    df_gold["head_comment_text"] = df_gold.apply(get_head_text, axis=1)
    df_gold["previous_comment_text"] = df_gold.apply(get_previous_text, axis=1)
    return df_gold


def create_prompt(row):
    return ANNOTATION_PROMPT.format(
        video_title=row["video_title"] if pd.notna(row["video_title"]) else "N/A",
        video_description=row["video_description"] if pd.notna(row["video_description"]) else "N/A",
        head_comment=row["head_comment_text"] if pd.notna(row["head_comment_text"]) else "N/A",
        previous_comment=row["previous_comment_text"] if pd.notna(row["previous_comment_text"]) else "N/A",
        target_comment=row["text"],
    )


def create_calibration_prompt(row, mode: str) -> str:
    """
    Build a (partially) content-free version of the annotation prompt, used to
    measure the model's baseline Sì/No bias for the requested calibration mode:

      "overall" : instructions only — every field is N/A.
      "video"   : keep this row's video title/description; N/A everywhere else.
      "comment" : keep video title/description + head/previous comment (if
                  available, same as the real prompt); only the target is N/A.

    In every mode the TARGET COMMENT itself is N/A — we're measuring the
    model's bias in the absence of the thing actually being judged.
    """
    video_title = row["video_title"] if pd.notna(row["video_title"]) else "N/A"
    video_description = row["video_description"] if pd.notna(row["video_description"]) else "N/A"
    head_comment = row["head_comment_text"] if pd.notna(row["head_comment_text"]) else "N/A"
    previous_comment = row["previous_comment_text"] if pd.notna(row["previous_comment_text"]) else "N/A"

    if mode == "overall":
        video_title = video_description = "N/A"
        head_comment = previous_comment = "N/A"
    elif mode == "video":
        head_comment = previous_comment = "N/A"
    elif mode == "comment":
        pass  # video title/description + head/previous kept as-is
    else:
        raise ValueError(f"Unknown calibration mode: {mode!r}")

    return ANNOTATION_PROMPT.format(
        video_title=video_title,
        video_description=video_description,
        head_comment=head_comment,
        previous_comment=previous_comment,
        target_comment="N/A",
    )


def load_gold_data(use_fake_data: bool = True) -> pd.DataFrame:
    """Load all *_gold.csv files + their metadata, attach context, build prompts."""
    base = "VideosComments_fake" if use_fake_data else "VideosComments"
    input_gold_dir = f"{base}/youtube/annotated_comments"
    input_gold_metadata_dir = f"{base}/youtube/annotated_metadata"

    all_dfs = []
    for newspaper in newspapers:
        gold_file_path = os.path.join(input_gold_dir, newspaper)
        gold_metadata_path = os.path.join(input_gold_metadata_dir, newspaper)

        for filename in os.listdir(gold_file_path):
            if not filename.endswith("_gold.csv"):
                continue

            gold_file_path_full = os.path.join(gold_file_path, filename)
            metadata_file_path_full = os.path.join(
                gold_metadata_path, filename.replace(".csv", ".json")
            )
            if not os.path.exists(metadata_file_path_full):
                raise ValueError(f"Metadata file not found for {filename}: {metadata_file_path_full}")

            with open(metadata_file_path_full, "r") as f:
                metadata = json.load(f)
            video_title = metadata.get("title", "")
            video_description = metadata.get("description", "")

            # Truncate long descriptions at the last sentence boundary before 1000 chars.
            if len(video_description) > 1000:
                trunc_point = video_description.rfind(".", 0, 1000)
                video_description = (
                    video_description[:trunc_point + 1] if trunc_point != -1
                    else video_description[:1000]
                )

            df_gold = pd.read_csv(gold_file_path_full)
            df_gold["newspaper"] = newspaper
            df_gold["video_title"] = video_title
            df_gold["video_description"] = video_description

            df_gold = add_context_columns(df_gold)
            df_gold["annotation_prompt"] = df_gold.apply(create_prompt, axis=1)
            all_dfs.append(df_gold)

    return pd.concat(all_dfs, ignore_index=True)


def make_dev_test_split(gold_df, test_size: float = 0.4, seed: int = 42):
    """
    60/40 dev/test. Stratify on label x depth so rarer deep-thread comments are
    not all dumped into one side; fall back to label-only if a stratum is too
    small to split (train_test_split needs >= 2 members per stratum).
    """
    strat = gold_df["label"].astype(str) + "_" + gold_df["depth"].astype(str)
    if strat.value_counts().min() < 2:
        print("WARNING: sparse label x depth strata — stratifying on label only")
        strat = gold_df["label"]
    return train_test_split(gold_df, test_size=test_size, random_state=seed, stratify=strat)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def build_messages(prompt: str, spec: ModelSpec) -> list:
    """
    Wrap the annotation prompt in the chat structure the model expects.
    No custom system prompt: the whole instruction lives in the user turn, and each
    model's own default system prompt (if any) is left in place by its chat template.

    Content shape is the only thing that varies:
      - text-only causal LM (tokenizer) -> plain string
      - VLM (processor)                 -> typed list of parts
    """
    if spec.is_vlm:
        content = [{"type": "text", "text": prompt}]
    else:
        content = prompt
    return [{"role": "user", "content": content}]


def load_model(spec: ModelSpec):
    """
    Instantiate model + tokenizer/processor from a ModelSpec.
    Uses the explicit class in the spec; applies bf16 + auto sharding globally
    (spec.load_kwargs can override per model); sets left padding for batched gen.
    The UNLOAD counterpart lives in run_one_model, not here.
    """
    proc = spec.proc_class.from_pretrained(spec.name)

    load_kwargs = {**GLOBAL_LOAD_KWARGS, **spec.load_kwargs}   # spec wins on conflict
    HF_TOKEN = os.getenv("HF_TOKEN")
    model = spec.model_class.from_pretrained(spec.name, token=HF_TOKEN,**load_kwargs)
    model.eval()

    # For a VLM the real tokenizer is proc.tokenizer; for a text model proc IS it.
    tok = proc.tokenizer if spec.is_vlm else proc
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, proc


# --------------------------------------------------------------------------- #
# Sì / No token-level scoring
# --------------------------------------------------------------------------- #

def get_answer_token_ids(tok) -> tuple:
    """
    Find every single-token id that decodes exactly to a bare "sì" or "no"
    answer (case-insensitive, with or without a leading space). Different
    tokenizers (Gemma2/Gemma3/Mistral) don't necessarily split "Sì" the same
    way, so instead of hardcoding one id per model we collect all variants
    that round-trip cleanly and pool them later with logsumexp.
    Raises if either set ends up empty, since that means this tokenizer needs
    a variant string added by hand.
    """
    yes_variants = ["Sì", " Sì", "sì", " sì",
                    "SI", " SI", "Si", " Si", "SÌ", " SÌ"]
    no_variants = ["No", " No", "no", " no", "NO", " NO"]

    def collect(variants):
        ids = set()
        for v in variants:
            enc = tok.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                decoded = tok.decode(enc).strip().lower()
                if decoded == v.strip().lower():
                    ids.add(enc[0])
        return ids

    yes_ids, no_ids = collect(yes_variants), collect(no_variants)
    if not yes_ids or not no_ids:
        raise ValueError(
            "Could not find single-token id(s) for the sì/no answer tokens "
            f"(yes_ids={yes_ids}, no_ids={no_ids}). Inspect this tokenizer's "
            "vocab and add the missing variant string(s) to yes_variants/no_variants."
        )
    return yes_ids, no_ids


def compute_logit_diffs(model, proc, spec: ModelSpec, prompts: list,
                         batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH,
                         test: bool = False, debug_tag: str = "") -> list:
    """
    Force exactly one generation step and read the logits for the Sì/No
    answer tokens off of it, instead of letting the model free-generate text
    and regex-parsing the result. Returns, per prompt, the raw
    (yes_logit - no_logit) log-odds of "Sì" vs "No" restricted to those two
    answer tokens (logit variants for each word are pooled with logsumexp).
    This is the quantity calibration subtracts a baseline from; turning it
    into a probability is a separate sigmoid step done by the caller.
    """
    tok = proc.tokenizer if spec.is_vlm else proc
    yes_ids, no_ids = get_answer_token_ids(tok)
    yes_idx = torch.tensor(sorted(yes_ids))
    no_idx = torch.tensor(sorted(no_ids))

    rendered = [
        proc.apply_chat_template(
            build_messages(p, spec),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    diffs = []
    debug_rows = []
    n_batches = (len(rendered) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(rendered), batch_size), 1):
        batch = rendered[start:start + batch_size]
        print(f"  batch {bi}/{n_batches}", end="\r")

        # add_special_tokens=False: the chat template already inserted BOS/specials;
        # tokenizing with the default True would add a SECOND BOS (Gemma is sensitive).
        inputs = proc(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,          # only need the logits for the very first token
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # out.scores[0]: (batch, vocab) raw logits for the first generated token.
        step_logits = out.scores[0].float().cpu()
        yes_logit = torch.logsumexp(step_logits[:, yes_idx], dim=-1)
        no_logit = torch.logsumexp(step_logits[:, no_idx], dim=-1)
        diff = yes_logit - no_logit

        diffs.extend(diff.tolist())
        if test:
            debug_rows.extend(zip(batch, yes_logit.tolist(), no_logit.tolist(), diff.tolist()))

    if test:
        suffix = f"_{debug_tag}" if debug_tag else ""
        pd.DataFrame(
            debug_rows,
            columns=["rendered_prompt", "yes_logit", "no_logit", "logit_diff"],
        ).to_csv(f"try_generations_{spec.name.replace('/', '_')}{suffix}.csv", index=False)
    print()
    return diffs


def run_inference_scores(model, proc, spec: ModelSpec, prompts: list,
                          batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH,
                          test: bool = False) -> list:
    """Uncalibrated P(offensive): sigmoid of the raw Sì/No logit difference."""
    diffs = compute_logit_diffs(model, proc, spec, prompts, batch_size, max_length, test=test)
    return torch.sigmoid(torch.tensor(diffs)).tolist()


def get_calibration_offsets(df: pd.DataFrame, mode: str, model, proc, spec: ModelSpec,
                             batch_size: int = BATCH_SIZE, test: bool = False) -> np.ndarray:
    """
    Per-row content-free logit-diff baseline for `mode`, deduplicated so
    identical calibration prompts (every row is identical under "overall";
    every comment on the same video shares one under "video") are only
    scored once instead of once per row.
    """
    calib_prompts = [create_calibration_prompt(row, mode) for _, row in df.iterrows()]
    unique_prompts = sorted(set(calib_prompts))
    unique_diffs = compute_logit_diffs(
        model, proc, spec, unique_prompts, batch_size=batch_size, test=test,
        debug_tag=f"calib_{mode}",
    )
    prompt_to_diff = dict(zip(unique_prompts, unique_diffs))
    return np.array([prompt_to_diff[p] for p in calib_prompts])


def tune_threshold(y_true: list, scores: list) -> dict:
    """
    Sweep the P(offensive) score against every candidate cut point precision_recall_curve
    reports (i.e. every point where the resulting label assignment can change) and keep
    the threshold that maximizes F1 on the offensive class. This is exhaustive over
    achievable operating points, so there's no need for a manual grid.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=1)
    # precision/recall have one extra point (predict-everyone-positive, no threshold);
    # drop it so lengths line up with `thresholds`.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1s = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    best_i = int(np.argmax(f1s))
    return {"threshold": float(thresholds[best_i]), "f1_offensive": float(f1s[best_i])}


# --------------------------------------------------------------------------- #
# Selection, evaluation
# --------------------------------------------------------------------------- #

def run_one_model(key: str, df: pd.DataFrame, batch_size: int = BATCH_SIZE, test: bool = False,
                   tune_hyperparameter: bool = True, calibration_mode: str = None) -> dict:
    """Load a model, score it (optionally calibrated), free it, pick a threshold
    (tuned on F1, or the untuned 0.5 midpoint). Self-contained so GPU memory is
    released before the next model loads (finally => freed even on error)."""
    spec = REGISTRY[key]
    model, proc = load_model(spec)
    print(f"Running with model: {key} ({spec.name})")

    prompts = df["annotation_prompt"].tolist()
    labels = df["label"].tolist()

    try:
        diffs = np.array(compute_logit_diffs(model, proc, spec, prompts, batch_size=batch_size, test=test))

        calibration_offsets = None
        if calibration_mode is not None:
            calibration_offsets = get_calibration_offsets(
                df, calibration_mode, model, proc, spec, batch_size=batch_size, test=test
            )
            diffs = diffs - calibration_offsets

        scores = torch.sigmoid(torch.tensor(diffs)).tolist()
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

    if tune_hyperparameter:
        threshold = tune_threshold(labels, scores)["threshold"]
    else:
        threshold = 0.5   # untuned midpoint: yes_logit == no_logit (post-calibration, if any)

    preds = [int(s >= threshold) for s in scores]
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    report = classification_report(labels, preds, target_names=["non-offensive", "offensive"], digits=4)
    print(
        f"\n--- {key} | dev ({'tuned' if tune_hyperparameter else 'fixed'} "
        f"threshold={threshold:.4f}) ---\n{report}\nF1 (offensive): {f1:.4f}"
    )

    if test:
        debug_df = pd.DataFrame({"prompt": prompts, "label": labels, "score": scores, "pred": preds})
        if calibration_offsets is not None:
            debug_df["calibration_offset"] = calibration_offsets
        debug_df.to_csv(f"try_{key}.csv", index=False)

    return {
        "key": key,
        "scores": scores,
        "threshold": threshold,
        "f1_offensive": f1,
        "report": report,
    }


def select_and_evaluate(dev_df: pd.DataFrame, test_df: pd.DataFrame, model_keys: list,
                         batch_size: int = BATCH_SIZE, test: bool = False,
                         tune_hyperparameter: bool = True, calibration_mode: str = None) -> dict:
    """Score every model on dev (with the requested calibration/tuning settings), pick
    the best F1(offensive), then evaluate that one model on test using ITS dev threshold
    (never retuned) — test-set calibration offsets are recomputed fresh from test_df,
    since calibration is a property of each row's own content, not a fitted parameter."""
    dev_results = [
        run_one_model(k, dev_df, batch_size, test=test,
                      tune_hyperparameter=tune_hyperparameter, calibration_mode=calibration_mode)
        for k in model_keys
    ]

    print(f"\n{'='*60}\nPer-model thresholds (dev):")
    for r in dev_results:
        print(f"  {r['key']:10s}  threshold={r['threshold']:.4f}  F1={r['f1_offensive']:.4f}")

    best = max(dev_results, key=lambda m: m["f1_offensive"])
    print(f"{'='*60}\nBest on dev: {best['key']}  (F1={best['f1_offensive']:.4f}, threshold={best['threshold']:.4f})\n{'='*60}")

    # Final, single evaluation on the untouched test split, using the threshold
    # locked in from dev (no re-tuning against test labels).
    spec = REGISTRY[best["key"]]
    model, proc = load_model(spec)
    test_prompts, test_labels = test_df["annotation_prompt"].tolist(), test_df["label"].tolist()
    try:
        test_diffs = np.array(compute_logit_diffs(model, proc, spec, test_prompts, batch_size=batch_size, test=test))
        if calibration_mode is not None:
            test_offsets = get_calibration_offsets(
                test_df, calibration_mode, model, proc, spec, batch_size=batch_size, test=test
            )
            test_diffs = test_diffs - test_offsets
        test_scores = torch.sigmoid(torch.tensor(test_diffs)).tolist()
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

    test_preds = [int(s >= best["threshold"]) for s in test_scores]
    test_report = classification_report(test_labels, test_preds, target_names=["non-offensive", "offensive"], digits=4)
    test_f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
    print(f"\n--- {best['key']} | test (threshold={best['threshold']:.4f}) ---\n{test_report}\nF1 (offensive): {test_f1:.4f}")

    summary = {
        "best_model": best["key"],
        "best_threshold": best["threshold"],
        "tune_hyperparameter": tune_hyperparameter,
        "calibration_mode": calibration_mode,
        "dev_f1_offensive": best["f1_offensive"],
        "test_f1_offensive": test_f1,
        "dev_report": best["report"],
        "test_report": test_report,
        "per_model_dev": {
            r["key"]: {"threshold": r["threshold"], "f1_offensive": r["f1_offensive"]}
            for r in dev_results
        },
    }
    output_summary_path = f"results_summary_comments{'_test' if test else ''}{f'_calib_{calibration_mode}' if calibration_mode else ''}{f'_tune_hyp' if tune_hyperparameter else ''}.json"
    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {output_summary_path}")
    return summary


# --------------------------------------------------------------------------- #
# CLI / Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot offensive-comment classification")
    parser.add_argument(
        "--test", action="store_true",
        help="Sample a small subset (6 rows) and dump per-batch debug CSVs "
             "(try_<model>.csv, try_generations_<model>*.csv).",
    )
    parser.add_argument(
        "--tune_hyperparameter", action="store_true",
        help="Tune the Sì/No decision threshold per model on dev by sweeping "
             "precision_recall_curve for the best F1. If omitted, use the "
             "untuned midpoint threshold of 0.5.",
    )

    calib_group = parser.add_mutually_exclusive_group()
    calib_group.add_argument(
        "--overall_calibration", action="store_true",
        help="Calibrate Sì/No logits against a fully content-free prompt "
             "(instructions only, N/A in every field).",
    )
    calib_group.add_argument(
        "--video_specific_calibration", action="store_true",
        help="Calibrate against a prompt with the video title/description "
             "filled in and N/A in every comment location.",
    )
    calib_group.add_argument(
        "--comment_specific_calibration", action="store_true",
        help="Calibrate against a prompt with the video title/description and "
             "head/previous comment (if available) filled in, N/A only for "
             "the target comment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv()
    gold_df = load_gold_data(use_fake_data=False)   # flip to False for the real run

    if args.test:
        gold_df = gold_df.sample(6, random_state=42).reset_index(drop=True)

    calibration_mode = None
    if args.overall_calibration:
        calibration_mode = "overall"
    elif args.video_specific_calibration:
        calibration_mode = "video"
    elif args.comment_specific_calibration:
        calibration_mode = "comment"

    dev_df, test_df = make_dev_test_split(gold_df)
    print("Correctly split into dev and test set")
    select_and_evaluate(
        dev_df, test_df, model_keys=list(REGISTRY),
        batch_size=BATCH_SIZE, test=args.test,
        tune_hyperparameter=args.tune_hyperparameter,
        calibration_mode=calibration_mode,
    )


if __name__ == "__main__":
    main()