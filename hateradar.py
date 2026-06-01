####
# Offensiveness-TRIGGERING prediction from video metadata (zero-shot).
#
# Input : the per-video dataset built earlier (video_dataset.csv) with columns
#         id, newspaper, text, topic, num_comments, percentage_offensive_comments, type
#
# Ground truth: a video is "offensiveness triggering" (label 1) iff
#         percentage_offensive_comments > tau            (tau on the 0-100 scale)
#
# Task  : given ONLY the video title + description, each model/prompt predicts
#         whether the video will trigger offensive comments (Sì / No).
#
# Output: results/<model>/prompt_<id>.csv   -> per-video predictions
#         results/<model>/prompt_<id>.txt   -> precision/recall/F1 per class + macro
#
# The model-loading / inference machinery is reused from the silver-annotation script.
####

import os
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
from sklearn.metrics import classification_report, precision_recall_fscore_support


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BATCH_SIZE = 32          # per-batch generation size; 8-16 is safe for 24-32B in bf16
MAX_LENGTH = 4096        # truncation ceiling for the tokenized prompt


# Five distinct Italian prompts for the SAME task: predict, from title +
# description only, whether the video will attract offensive comments.
# All of them force a single-token "Sì"/"No" answer so parse_label + the small
# max_new_tokens budget keep working unchanged.
PROMPTS = [
    # 0 — minimal / direct
    """Sei un sistema automatico di moderazione di una piattaforma di notizie online.

Dato il TITOLO e la DESCRIZIONE di un video di notizie, il tuo compito è prevedere se la sezione commenti del video conterrà linguaggio offensivo (insulti, aggressività, odio, discriminazione, ecc.).

TITOLO DEL VIDEO:
{video_title}

DESCRIZIONE DEL VIDEO:
{video_description}

Il video genererà commenti offensivi?
Rispondi ESCLUSIVAMENTE con una sola parola: "Sì" oppure "No".
""",

    # 1 — definition-rich
    """Sei un esperto di moderazione di contenuti online in italiano.

DEFINIZIONE: un commento è offensivo se contiene linguaggio volgare, insultante, aggressivo, denigratorio, discriminatorio o minaccioso verso una persona o un gruppo.

COMPITO: leggendo SOLO il titolo e la descrizione di un video di notizie, prevedi se il video attirerà un numero rilevante di commenti offensivi da parte degli utenti.

TITOLO:
{video_title}

DESCRIZIONE:
{video_description}

Considera quanto l'argomento è delicato, divisivo o emotivamente carico.

Rispondi solo con "Sì" (attirerà commenti offensivi) oppure "No" (non li attirerà). Nessun'altra parola.
""",

    # 2 — analyst persona, polarization focus
    """Agisci come un analista di dinamiche social specializzato nella polarizzazione del dibattito pubblico italiano.

Ti vengono forniti il titolo e la descrizione di un video di notizie. Valuta se l'argomento e il modo in cui è presentato sono tali da scatenare reazioni ostili o offensive nei commenti (es. temi politici controversi, immigrazione, religione, cronaca nera, attriti sociali).

TITOLO: {video_title}
DESCRIZIONE: {video_description}

Questo video tenderà a scatenare commenti offensivi?
Output: una sola etichetta, "Sì" oppure "No".
""",

    # 3 — guided reasoning, final answer only
    """Sei un moderatore esperto. Devi prevedere se un video di notizie provocherà commenti offensivi, basandoti solo su titolo e descrizione.

Valuta mentalmente questi fattori, SENZA scriverli:
- L'argomento è divisivo o conflittuale?
- Tocca gruppi spesso bersaglio di odio (etnia, religione, genere, orientamento, politica)?
- Il tono è provocatorio o sensazionalistico?
- Suscita forti emozioni (rabbia, indignazione)?

TITOLO:
{video_title}

DESCRIZIONE:
{video_description}

Dopo aver considerato questi fattori, fornisci SOLO la risposta finale: "Sì" se è probabile che generi commenti offensivi, "No" altrimenti. Non scrivere il ragionamento.
""",

    # 4 — audience-reaction imagination
    """Immagina la sezione commenti di questo video di notizie su una piattaforma italiana.

TITOLO DEL VIDEO:
{video_title}

DESCRIZIONE DEL VIDEO:
{video_description}

In base al titolo e alla descrizione, pensi che tra i commenti compariranno messaggi offensivi, aggressivi o irrispettosi verso persone o gruppi?

Rispondi soltanto "Sì" oppure "No".
""",
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

def split_title_description(text):
    """The dataset's `text` column is title + '\\n\\n' + description. Recover the
    two parts (split on the first blank line). Falls back to N/A when missing."""
    if not isinstance(text, str) or not text.strip():
        return "N/A", "N/A"
    parts = text.split("\n\n", 1)
    title = parts[0].strip() or "N/A"
    description = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "N/A"
    return title, description


def load_dataset(input_path: str, tau: float, type_filter: str = "all"):
    """Read the per-video CSV, optionally filter by gold/silver, derive the binary
    `true_label` (percentage_offensive_comments > tau), drop videos with no
    percentage (no comments -> undefined ground truth)."""
    df = pd.read_csv(input_path)

    if type_filter != "all":
        before = len(df)
        df = df[df["type"].astype(str) == type_filter].copy()
        print(f"Type filter '{type_filter}': {len(df)}/{before} videos kept")

    pct = pd.to_numeric(df["percentage_offensive_comments"], errors="coerce")
    n_dropped = int(pct.isna().sum())
    if n_dropped:
        print(f"Dropping {n_dropped} videos with no percentage (no comments) — undefined label")
    df = df[pct.notna()].copy()
    pct = pct[pct.notna()]

    df["true_label"] = (pct > tau).astype(int)
    df["__title"], df["__description"] = zip(*df["text"].map(split_title_description))

    n_pos = int(df["true_label"].sum())
    print(f"Loaded {len(df)} videos | triggering (label 1): {n_pos} | "
          f"non-triggering (label 0): {len(df) - n_pos} | tau={tau}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Inference  (reused from the silver-annotation script)
# --------------------------------------------------------------------------- #

def build_messages(prompt: str, spec: ModelSpec) -> list:
    """Wrap the prompt in the chat structure the model expects (string for a
    text LM, typed parts for a VLM). No custom system prompt."""
    if spec.is_vlm:
        content = [{"type": "text", "text": prompt}]
    else:
        content = prompt
    return [{"role": "user", "content": content}]


def load_model(spec: ModelSpec):
    """Instantiate model + tokenizer/processor; bf16 + auto sharding; left padding."""
    proc = spec.proc_class.from_pretrained(spec.name)

    load_kwargs = {**GLOBAL_LOAD_KWARGS, **spec.load_kwargs}   # spec wins on conflict
    HF_TOKEN = os.getenv("HF_TOKEN")
    model = spec.model_class.from_pretrained(spec.name, token=HF_TOKEN, **load_kwargs)
    model.eval()

    tok = proc.tokenizer if spec.is_vlm else proc
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, proc


# Accented "sì" / "no" are unambiguous answer tokens. Bare "si" is also the
# Italian pronoun, so it is only a weak fallback.
_YES = re.compile(r"\bsì\b")
_NO = re.compile(r"\bno\b")


def parse_label(raw: str):
    """1 = triggering, 0 = not, None = unparseable. Scans for the LAST answer
    token so a reasoning model's final verdict wins over earlier tokens."""
    text = raw.strip().lower()
    if not text:
        return None

    yes_hits = list(_YES.finditer(text))
    no_hits = list(_NO.finditer(text))
    last_yes = yes_hits[-1].start() if yes_hits else -1
    last_no = no_hits[-1].start() if no_hits else -1

    if last_yes != -1 or last_no != -1:
        return 1 if last_yes > last_no else 0

    if re.fullmatch(r"si[.!]?", text):
        return 1
    return None


def run_inference(model, proc, spec: ModelSpec, prompts: list,
                  batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH) -> list:
    """Render -> batch-tokenize -> generate -> decode new tokens -> parse."""
    tok = proc.tokenizer if spec.is_vlm else proc

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

        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        preds.extend(parse_label(d) for d in decoded)

    print()
    return preds


# --------------------------------------------------------------------------- #
# Scoring & output
# --------------------------------------------------------------------------- #

TARGET_NAMES = ["non-triggering (No)", "triggering (Si)"]


def write_outputs(out_dir: str, model_key: str, prompt_id: int,
                  df: pd.DataFrame, preds: list, tau: float):
    """Write results/<model>/prompt_<id>.csv (predictions) and .txt (metrics)."""
    model_dir = os.path.join(out_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)
    csv_path = os.path.join(model_dir, f"prompt_{prompt_id}.csv")
    txt_path = os.path.join(model_dir, f"prompt_{prompt_id}.txt")

    # --- predictions CSV (keeps unparseable rows, pred_label left empty) ---
    out_df = df[["id", "newspaper", "text", "percentage_offensive_comments", "true_label"]].copy()
    out_df["pred_label"] = preds
    out_df.to_csv(csv_path, index=False)

    # --- metrics on parseable rows only ---
    pairs = [(t, p) for t, p in zip(df["true_label"].tolist(), preds) if p is not None]
    n_total = len(preds)
    n_unparseable = n_total - len(pairs)

    lines = [
        f"Model:   {model_key}",
        f"Prompt:  {prompt_id}",
        f"tau:     {tau}  (label 1 iff percentage_offensive_comments > tau)",
        f"Videos evaluated:        {n_total}",
        f"Unparseable (excluded):  {n_unparseable}",
        "",
    ]

    if pairs:
        y_true, y_pred = zip(*pairs)
        report = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=TARGET_NAMES, digits=4, zero_division=0,
        )
        # Explicit macro recap (also present inside the report).
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average="macro", zero_division=0,
        )
        lines.append(report)
        lines.append("")
        lines.append(f"Macro-averaged -> precision: {p:.4f} | recall: {r:.4f} | f1: {f1:.4f}")
    else:
        lines.append("No parseable predictions — metrics unavailable.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  -> {csv_path}")
    print(f"  -> {txt_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Predict offensiveness-triggering videos with multiple LLMs and prompts.")
    ap.add_argument("--tau", type=float, required=True,
                    help="Threshold on percentage_offensive_comments (0-100). label 1 iff value > tau.")
    ap.add_argument("--input", default="video_dataset.csv",
                    help="Per-video dataset CSV from the previous script.")
    ap.add_argument("--out-dir", default="results",
                    help="Root output directory (default: results).")
    ap.add_argument("--models", nargs="+", choices=list(REGISTRY), default=list(REGISTRY),
                    help="Which models to run (default: all in the registry).")
    ap.add_argument("--type", choices=["all", "gold", "silver"], default="all",
                    help="Restrict ground truth to gold/silver videos (default: all).")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return ap.parse_args()


def main():
    load_dotenv()  # for Hugging Face API keys, if needed
    args = parse_args()

    df = load_dataset(args.input, tau=args.tau, type_filter=args.type)
    if df.empty:
        print("No videos to evaluate — aborting.")
        return

    # Pre-render the 5 prompt sets once (same across models).
    tds = list(zip(df["__title"].tolist(), df["__description"].tolist()))
    prompt_sets = [
        [tmpl.format(video_title=t, video_description=d) for (t, d) in tds]
        for tmpl in PROMPTS
    ]

    for model_key in args.models:
        spec = REGISTRY[model_key]
        print(f"\n{'#'*60}\nModel: {model_key} ({spec.name})\n{'#'*60}")
        model, proc = load_model(spec)
        try:
            for prompt_id, prompts in enumerate(prompt_sets):
                print(f"\n[{model_key}] prompt {prompt_id}")
                preds = run_inference(model, proc, spec, prompts, batch_size=args.batch_size)
                write_outputs(args.out_dir, model_key, prompt_id, df, preds, args.tau)
        finally:
            del model, proc
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nDone. Results under '{args.out_dir}/'")


if __name__ == "__main__":
    main()