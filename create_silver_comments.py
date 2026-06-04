####
# Offensive-comment SILVER annotation via zero-shot prompting.
#
# Unlike the selection script, nothing is split or compared here. A single,
# already-chosen "best" model (passed with --model) is loaded once and used to
# label every comment file that does NOT already have a human (gold) annotation.
#
# For each raw comment file  comments/<newspaper>/<id>.csv:
#   - if annotated_comments/<newspaper>/<id>_gold.csv exists -> skip (human-labelled)
#   - else run the model and write annotated_comments/<newspaper>/<id>_silver.csv
#
# The silver file mirrors the raw comment file's columns + a `label` column
# (1 = offensive, 0 = not, empty = unparseable model output).
####

import os
import json
import gc
import re
import argparse

from dotenv import load_dotenv

import torch
import pandas as pd
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForVision2Seq,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BATCH_SIZE = 8          # per-batch generation size; 8-16 is safe for 24-32B in bf16
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
    max_new_tokens: int = 5         # thinking models (ANITA) need far more room
    load_kwargs: dict = field(default_factory=dict)


# Explicit classes per model card (more robust than the version-dependent
# Auto* umbrellas). Add quantization etc. via a single model's load_kwargs.
REGISTRY = {
    "mistral": ModelSpec(
        "mistralai/Mistral-Small-24B-Instruct-2501",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=True,
    ),
    "gemma2": ModelSpec(
        "google/gemma-2-27b-it",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=False,   # template likely rejects system role
    ),
    "gemma3": ModelSpec(
        "google/gemma-3-27b-it",
        Gemma3ForConditionalGeneration, AutoProcessor,
        is_vlm=True, supports_system_role=True,
    ),
    # "qwen_vl": ModelSpec( -> DID NOT MANAGE TO INSTALL TORCHVISION
    #     "Qwen/Qwen2.5-VL-32B-Instruct",
    #     Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    #     is_vlm=True, supports_system_role=True,
    # ),
    # "anita": ModelSpec(
    #     "m-polignano/ANITA-NEXT-24B-Magistral-2506-VISION-ITA",
    #     AutoModelForVision2Seq, AutoProcessor,
    #     is_vlm=True, supports_system_role=True,
    #     max_new_tokens=1024,                         # thinking model: room to reason
    # ),
}

GLOBAL_LOAD_KWARGS = dict(
    torch_dtype=torch.bfloat16,     # if transformers warns, rename to dtype=...
    device_map="auto",              # shards big models across visible GPUs
)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def add_context_columns(df):
    """
    Add head_comment_text and previous_comment_text columns based on depth:
      depth 0 : both None
      depth 1 : head = depth-0 ancestor's text, previous = None
      depth 2+: head = depth-0 ancestor's text, previous = immediate parent's text
    """
    id_to_text = df.set_index("comment_id")["text"].to_dict()
    id_to_parent = df.set_index("comment_id")["inferred_parent_id"].to_dict()

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

    df["head_comment_text"] = df.apply(get_head_text, axis=1)
    df["previous_comment_text"] = df.apply(get_previous_text, axis=1)
    return df


def create_prompt(row):
    return ANNOTATION_PROMPT.format(
        video_title=row["video_title"] if pd.notna(row["video_title"]) else "N/A",
        video_description=row["video_description"] if pd.notna(row["video_description"]) else "N/A",
        head_comment=row["head_comment_text"] if pd.notna(row["head_comment_text"]) else "N/A",
        previous_comment=row["previous_comment_text"] if pd.notna(row["previous_comment_text"]) else "N/A",
        target_comment=row["text"],
    )


def load_metadata(metadata_file_path_full):
    """Return (title, description) for a video, with the description truncated at
    the last sentence boundary before 1000 chars. Missing metadata is non-fatal
    for silver labelling: we just fall back to empty strings."""
    if not os.path.exists(metadata_file_path_full):
        print(f"  WARNING: metadata not found ({metadata_file_path_full}) — using empty title/description")
        return "", ""

    with open(metadata_file_path_full, "r") as f:
        metadata = json.load(f)
    video_title = metadata.get("title", "")
    video_description = metadata.get("description", "")

    if len(video_description) > 1000:
        trunc_point = video_description.rfind(".", 0, 1000)
        video_description = (
            video_description[:trunc_point + 1] if trunc_point != -1
            else video_description[:1000]
        )
    return video_title, video_description


def prepare_df(df, video_title, video_description, newspaper):
    """Attach metadata columns + per-comment context + the rendered prompt."""
    df["newspaper"] = newspaper
    df["video_title"] = video_title
    df["video_description"] = video_description
    df = add_context_columns(df)
    df["annotation_prompt"] = df.apply(create_prompt, axis=1)
    return df


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
    """
    proc = spec.proc_class.from_pretrained(spec.name)

    load_kwargs = {**GLOBAL_LOAD_KWARGS, **spec.load_kwargs}   # spec wins on conflict
    HF_TOKEN = os.getenv("HF_TOKEN")
    model = spec.model_class.from_pretrained(spec.name, token=HF_TOKEN, **load_kwargs)
    model.eval()

    # For a VLM the real tokenizer is proc.tokenizer; for a text model proc IS it.
    tok = proc.tokenizer if spec.is_vlm else proc
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, proc


# Accented "sì" / "no" are unambiguous answer tokens. Bare "si" is also the
# Italian pronoun ("si può considerare..."), so it is only a weak fallback.
_YES = re.compile(r"\bsì\b")
_NO = re.compile(r"\bno\b")


def parse_label(raw: str):
    """1 = offensive, 0 = not, None = unparseable. Scans for the LAST answer token
    so a reasoning model's final verdict wins over tokens inside its reasoning."""
    text = raw.strip().lower()
    if not text:
        return None

    yes_hits = list(_YES.finditer(text))
    no_hits = list(_NO.finditer(text))
    last_yes = yes_hits[-1].start() if yes_hits else -1
    last_no = no_hits[-1].start() if no_hits else -1

    if last_yes != -1 or last_no != -1:
        return 1 if last_yes > last_no else 0

    # Weak fallback: unaccented "si" only if it is essentially the whole output.
    if re.fullmatch(r"si[.!]?", text):
        return 1
    return None


def run_inference(model, proc, spec: ModelSpec, prompts: list,
                  batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH) -> list:
    """Render -> batch-tokenize -> generate -> decode new tokens -> parse."""
    tok = proc.tokenizer if spec.is_vlm else proc

    # Render to formatted STRINGS first, then batch-tokenize with uniform left padding.
    rendered = [
        proc.apply_chat_template(
            build_messages(p, spec),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    preds = []
    n_batches = (len(rendered) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(rendered), batch_size), 1):
        batch = rendered[start:start + batch_size]
        print(f"    batch {bi}/{n_batches}", end="\r")

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
                max_new_tokens=spec.max_new_tokens,
                do_sample=False,                       # greedy => reproducible
                pad_token_id=tok.pad_token_id,
            )

        # Left padding makes prompt length uniform, so one slice recovers the
        # generated tokens for every row in the batch.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        preds.extend(parse_label(d) for d in decoded)

    print()
    return preds


# --------------------------------------------------------------------------- #
# Silver annotation
# --------------------------------------------------------------------------- #

# def annotate_silver(model, proc, spec: ModelSpec,
#                     comments_dir: str, metadata_dir: str, annotated_dir: str,
#                     batch_size: int = BATCH_SIZE, overwrite: bool = False):
#     """
#     Walk every raw comment file. Skip the ones that already have a gold file;
#     label the rest with the loaded model and write <id>_silver.csv next to gold.
#     """
#     n_skipped_gold = n_skipped_silver = n_written = n_errors = 0

#     for newspaper in newspapers:
#         comments_np = os.path.join(comments_dir, newspaper)
#         metadata_np = os.path.join(metadata_dir, newspaper)
#         annotated_np = os.path.join(annotated_dir, newspaper)

#         if not os.path.isdir(comments_np):
#             print(f"WARNING: no comments dir for '{newspaper}' ({comments_np}) — skipping")
#             continue
#         os.makedirs(annotated_np, exist_ok=True)

#         for filename in sorted(os.listdir(comments_np)):
#             # Only raw comment files: plain <id>.csv, never *_gold/_silver.csv.
#             if not filename.endswith(".csv"):
#                 continue
#             if filename.endswith("_gold.csv") or filename.endswith("_silver.csv"):
#                 continue

#             file_id = filename[:-4]                      # strip ".csv"
#             gold_path = os.path.join(annotated_np, f"{file_id}_gold.csv")
#             silver_path = os.path.join(annotated_np, f"{file_id}_silver.csv")

#             if os.path.exists(gold_path):
#                 n_skipped_gold += 1
#                 continue                                 # human-annotated already
#             if os.path.exists(silver_path) and not overwrite:
#                 n_skipped_silver += 1
#                 continue                                 # resume: already done

#             print(f"\n[{newspaper}/{file_id}] annotating...")
#             try:
#                 df = pd.read_csv(os.path.join(comments_np, filename))
#                 if df.empty:
#                     df["label"] = []
#                     df.to_csv(silver_path, index=False)
#                     continue

#                 # Columns to persist = raw schema + the new `label` (no leftover label col).
#                 out_cols = [c for c in df.columns.tolist() if c != "label"] + ["label"]

#                 video_title, video_description = load_metadata(
#                     os.path.join(metadata_np, f"{file_id}.json")
#                 )
#                 df = prepare_df(df, video_title, video_description, newspaper)

#                 preds = run_inference(
#                     model, proc, spec, df["annotation_prompt"].tolist(),
#                     batch_size=batch_size,
#                 )
#                 df["label"] = preds

#                 n_none = sum(p is None for p in preds)
#                 if n_none:
#                     print(f"  {n_none}/{len(preds)} unparseable — written as empty label")

#                 df[out_cols].to_csv(silver_path, index=False)
#                 print(f"  -> {silver_path}  ({len(df)} comments)")
#                 n_written += 1
#             except Exception as e:
#                 n_errors += 1
#                 print(f"  ERROR on {newspaper}/{file_id}: {e}")

#     print(f"\n{'='*60}")
#     print(f"Done. silver written: {n_written} | "
#           f"skipped (gold): {n_skipped_gold} | "
#           f"skipped (existing silver): {n_skipped_silver} | "
#           f"errors: {n_errors}")
#     print(f"{'='*60}")


def annotate_silver(model, proc, spec: ModelSpec,
                    comments_dir: str, metadata_dir: str, annotated_dir: str,
                    batch_size: int = BATCH_SIZE, overwrite: bool = False):
    """
    Pool every comment across every file into ONE batched inference run, so that
    a 1-comment file no longer wastes a whole batch. Predictions are scattered
    back to each file afterwards via the [start:end] slice it occupies.
    """
    n_skipped_gold = n_skipped_silver = n_skipped_empty = n_written = n_errors = 0

    # ---- Pass 1: collect work ------------------------------------------------ #
    jobs = []          # one entry per file we will actually annotate
    all_prompts = []   # flat list of every prompt across every file

    for newspaper in newspapers:
        comments_np = os.path.join(comments_dir, newspaper)
        metadata_np = os.path.join(metadata_dir, newspaper)
        annotated_np = os.path.join(annotated_dir, newspaper)

        if not os.path.isdir(comments_np):
            print(f"WARNING: no comments dir for '{newspaper}' ({comments_np}) — skipping")
            continue
        os.makedirs(annotated_np, exist_ok=True)

        for filename in sorted(os.listdir(comments_np)):
            if not filename.endswith(".csv"):
                continue
            if filename.endswith("_gold.csv") or filename.endswith("_silver.csv"):
                continue

            file_id = filename[:-4]
            gold_path = os.path.join(annotated_np, f"{file_id}_gold.csv")
            silver_path = os.path.join(annotated_np, f"{file_id}_silver.csv")

            if os.path.exists(gold_path):
                n_skipped_gold += 1
                continue
            if os.path.exists(silver_path) and not overwrite:
                n_skipped_silver += 1
                continue

            comment_path = os.path.join(comments_np, filename)
            try:
                df = pd.read_csv(comment_path)
            except pd.errors.EmptyDataError:
                # Truly empty input (0 bytes, no header): no schema to mirror,
                # so we emit a silver file with just the `label` column.
                print(f"[{newspaper}/{file_id}] empty file (no header/rows) — writing empty silver")
                pd.DataFrame(columns=["label"]).to_csv(silver_path, index=False)
                n_skipped_empty += 1
                continue

            if df.empty:
                # Header present but no comment rows: mirror the schema + label.
                print(f"[{newspaper}/{file_id}] no comments — writing empty silver")
                out_cols = [c for c in df.columns.tolist() if c != "label"] + ["label"]
                df["label"] = pd.Series(dtype=object)
                df[out_cols].to_csv(silver_path, index=False)
                n_skipped_empty += 1
                continue
            
            # Per-file prep is wrapped so one bad file doesn't abort collection.
            try:
                out_cols = [c for c in df.columns.tolist() if c != "label"] + ["label"]
                video_title, video_description = load_metadata(
                    os.path.join(metadata_np, f"{file_id}.json")
                )
                df = prepare_df(df, video_title, video_description, newspaper)
                prompts = df["annotation_prompt"].tolist()
            except Exception as e:
                n_errors += 1
                print(f"  ERROR preparing {newspaper}/{file_id}: {e}")
                continue

            start = len(all_prompts)
            all_prompts.extend(prompts)
            jobs.append({
                "newspaper": newspaper,
                "file_id": file_id,
                "df": df,
                "out_cols": out_cols,
                "silver_path": silver_path,
                "start": start,
                "end": start + len(prompts),
            })

    if not all_prompts:
        print("Nothing to annotate.")
        print(f"skipped (gold): {n_skipped_gold} | "
              f"skipped (existing silver): {n_skipped_silver} | "
              f"skipped (empty): {n_skipped_empty} | errors: {n_errors}")
        return

    # ---- Pass 2: ONE batched inference over the whole pool ------------------- #
    print(f"\nAnnotating {len(all_prompts)} comments across {len(jobs)} files "
          f"in batches of {batch_size}...")
    all_preds = run_inference(model, proc, spec, all_prompts, batch_size=batch_size)

    # ---- Pass 3: scatter predictions back + write --------------------------- #
    for job in jobs:
        df = job["df"]
        preds = all_preds[job["start"]:job["end"]]
        df["label"] = preds

        n_none = sum(p is None for p in preds)
        if n_none:
            print(f"  [{job['newspaper']}/{job['file_id']}] "
                  f"{n_none}/{len(preds)} unparseable — written as empty label")

        try:
            df[job["out_cols"]].to_csv(job["silver_path"], index=False)
            print(f"  -> {job['silver_path']}  ({len(df)} comments)")
            n_written += 1
        except Exception as e:
            n_errors += 1
            print(f"  ERROR writing {job['newspaper']}/{job['file_id']}: {e}")

    print(f"\n{'='*60}")
    print(f"Done. silver written: {n_written} | "
          f"skipped (gold): {n_skipped_gold} | "
          f"skipped (existing silver): {n_skipped_silver} | "
          f"skipped (empty): {n_skipped_empty} | "
          f"errors: {n_errors}")
    print(f"{'='*60}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Silver-annotate offensive comments with the best model.")
    ap.add_argument("--model", required=True, choices=list(REGISTRY),
                    help="Which (best) model to use for silver annotation.")
    ap.add_argument("--use-fake-data", action="store_true",
                    help="Use the VideosComments_fake tree instead of VideosComments.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-annotate even if a *_silver.csv already exists.")
    return ap.parse_args()


def main():
    load_dotenv()  # for Hugging Face API keys, if needed
    args = parse_args()

    base = "VideosComments_fake" if args.use_fake_data else "VideosComments"
    comments_dir = f"{base}/youtube/anonymized_comments"
    metadata_dir = f"{base}/youtube/metadata"
    annotated_dir = f"{base}/youtube/annotated_comments"

    spec = REGISTRY[args.model]
    print(f"Loading best model: {args.model} ({spec.name})")
    model, proc = load_model(spec)
    try:
        annotate_silver(
            model, proc, spec,
            comments_dir=comments_dir,
            metadata_dir=metadata_dir,
            annotated_dir=annotated_dir,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()