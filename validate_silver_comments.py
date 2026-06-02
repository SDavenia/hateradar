####
# Offensive-comment classification via zero-shot prompting.
# 1. Load gold annotated data (+ build per-comment context and prompt).
# 2. Split gold data 60/40 into dev/test (stratified on label x depth).
# 3. Select best model on the 60% dev split, report final metrics on the 40% test split.
####

import os
import json
import gc
import re

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BATCH_SIZE = 8         # per-batch generation size; 8-16 is safe for 24-32B in bf16
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
    "mistral": ModelSpec(
        "mistralai/Mistral-Small-24B-Instruct-2501",
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
                  batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH,
                  test: bool = False) -> list:
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
    raw_outputs = []                       # TODO: For testing
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
                max_new_tokens=spec.max_new_tokens,
                do_sample=False,                       # greedy => reproducible
                pad_token_id=tok.pad_token_id,
            )

        # Left padding makes prompt length uniform, so one slice recovers the
        # generated tokens for every row in the batch.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        raw_outputs.extend(decoded)  # TODO: For testing
        preds.extend(parse_label(d) for d in decoded)

    if test:
        pd.DataFrame({
            "prompt": prompts,
            "raw_pred": raw_outputs,        # now full length
            "parsed_pred": preds,
        }).to_csv(f"try_generations_{spec.name.replace('/', '_')}.csv", index=False)
    print()
    return preds


# --------------------------------------------------------------------------- #
# Scoring, selection, evaluation
# --------------------------------------------------------------------------- #

def score(y_true: list, y_pred: list, label: str) -> dict:
    """Drop unparseable predictions, report, return F1 on the offensive class."""
    n_none = sum(p is None for p in y_pred)
    if n_none:
        print(f"  [{label}] {n_none}/{len(y_pred)} unparseable — excluded from metrics")

    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        print(f"  [{label}] no parseable predictions — F1 = 0")
        return {"f1_offensive": 0.0, "report": "no parseable predictions", "n_unparseable": n_none}

    yt, yp = zip(*pairs)
    report = classification_report(yt, yp, target_names=["non-offensive", "offensive"], digits=4)
    f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
    print(f"\n--- {label} ---\n{report}\nF1 (offensive): {f1:.4f}")
    return {"f1_offensive": f1, "report": report, "n_unparseable": n_none}


def run_one_model(key: str, prompts: list, labels: list, batch_size: int = BATCH_SIZE, test: bool = False) -> dict:
    """Load a model, run it, free it, score. Self-contained so GPU memory is
    released before the next model loads (finally => freed even on error)."""
    spec = REGISTRY[key]
    model, proc = load_model(spec)
    print(f"Running with model: {key} ({spec.name})")
    try:
        preds = run_inference(model, proc, spec, prompts, batch_size=batch_size, test=test)
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()
    if test:
        # Save outputs to try_modelname.csv 
        df_out = pd.DataFrame({
            "prompt": prompts,
            "label": labels,
            "pred": preds,
        })
        df_out.to_csv(f"try_{key}.csv", index=False)

    metrics = score(labels, preds, label=f"{key} | dev")
    metrics.update(key=key, preds=preds)
    return metrics


def select_and_evaluate(dev_df, test_df, model_keys, batch_size: int = BATCH_SIZE, test: bool = False) -> dict:
    """Run every model on dev, pick best F1(offensive), evaluate that one on test."""
    dev_prompts, dev_labels = dev_df["annotation_prompt"].tolist(), dev_df["label"].tolist()

    dev_results = [run_one_model(k, dev_prompts, dev_labels, batch_size, test=test) for k in model_keys]

    best = max(dev_results, key=lambda m: m["f1_offensive"])
    print(f"\n{'='*60}\nBest on dev: {best['key']}  (F1={best['f1_offensive']:.4f})\n{'='*60}")

    # Final, single evaluation on the untouched test split.
    test_prompts, test_labels = test_df["annotation_prompt"].tolist(), test_df["label"].tolist()
    spec = REGISTRY[best["key"]]
    model, proc = load_model(spec)
    try:
        test_preds = run_inference(model, proc, spec, test_prompts, batch_size=batch_size, test=test)
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()
    test_metrics = score(test_labels, test_preds, label=f"{best['key']} | test")
    
    summary = {
        "best_model": best["key"],
        "dev_f1_offensive": best["f1_offensive"],
        "test_f1_offensive": test_metrics["f1_offensive"],
        "dev_report": best["report"],
        "test_report": test_metrics["report"],
    }
    with open("results_summary_comments.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved -> results_summary_comments.json")
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    load_dotenv()  
    gold_df = load_gold_data(use_fake_data=False)   # flip to False for the real run

    # TODO: For small testing runs
    test = False
    if test:
        gold_df = gold_df.sample(50, random_state=42).reset_index(drop=True)

    dev_df, test_df = make_dev_test_split(gold_df)
    print(f"Correctly split into dev and test set")
    select_and_evaluate(dev_df, test_df, model_keys=list(REGISTRY), batch_size=BATCH_SIZE, test=test)


if __name__ == "__main__":
    main()