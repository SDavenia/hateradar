####
# Article topic-assignment via zero-shot prompting.
# 1. Load gold-annotated article metadata (title + description + gold `topic`).
# 2. Split gold data 60/40 into dev/test (stratified on topic).
# 3. Select best model on the 60% dev split, report final metrics on the 40% test split.
#
# Nothing is trained: the 60% is a model-SELECTION split, hence "dev" not "train".
# Each article is read from the annotated_metadata folder; the gold label is the
# value of the "topic" field, mapped to one of the 17 top-level taxonomy categories.
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
# Taxonomy
# --------------------------------------------------------------------------- #
# Top-level entries (keys) are the CATEGORIES the model must choose from.
# The lists are example instances/subtopics, used only to describe what each
# category covers — they are NOT valid answers on their own (a model that emits
# an instance is mapped back to its parent category by parse_topic).

TAXONOMY = {
    "arts, culture, entertainment and media": [
        "arts and entertainment", "culture", "mass media",
    ],
    "crime, law and justice": [
        "crime", "judiciary", "justice", "law", "law enforcement",
    ],
    "disaster, accident and emergency incident": [
        "accident and emergency incident", "disaster", "emergency incident",
        "emergency planning", "emergency response",
    ],
    "economy, business and finance": [
        "business information", "products and services", "economy",
        "business enterprise", "market and exchange",
    ],
    "education": [
        "parents group", "religious education", "school", "social learning",
        "teaching and learning", "curriculum",
        "educational testing and examinations", "entrance examination",
        "students", "teachers", "vocational education", "educational grading",
        "online and remote learning",
    ],
    "environment": [
        "climate change", "conservation", "environmental pollution",
        "natural resource", "nature", "sustainability",
    ],
    "health": [
        "disease and condition", "health facility", "health organisation",
        "health treatment and procedure", "government health care",
        "health insurance", "private health care", "medical profession",
        "non-human diseases", "public health",
    ],
    "human interest": [
        "accomplishment", "award and prize", "record and achievement",
        "ceremony", "people", "human mishap", "high society", "celebrity",
        "anniversary", "birthday",
    ],
    "labour": [
        "employment", "employment legislation", "labour market",
        "labour relations", "retirement", "unemployment", "unions",
    ],
    "lifestyle and leisure": [
        "leisure", "lifestyle", "wellness",
    ],
    "politics and government": [
        "election", "government", "government policy", "international relations",
        "non-governmental organisation (NGO)", "political crisis",
        "political prisoners and dissenters", "political process",
    ],
    "religion": [
        "belief systems", "interreligious dialogue", "religious conflict",
        "religious event", "religious festival and holiday", "religious ritual",
        "religious facility", "relations between religion and government",
        "religious leader", "religious text",
    ],
    "science and technology": [
        "biomedical science", "mathematics", "natural science",
        "scientific research", "scientific institution", "social sciences",
        "scientific standards", "technology and engineering",
    ],
    "society": [
        "fundamental rights", "communities", "demographics", "immigration",
        "emigration", "discrimination", "family", "demographic group",
        "social condition", "social problem", "values", "welfare",
        "diversity, equity and inclusion",
    ],
    "sport": [
        "competition discipline", "disciplinary action in sport",
        "drug use in sport", "sport event", "sport industry",
        "sport organisation", "sport venue", "sports transaction",
        "sport achievement", "sports coaching",
        "sports management and ownership", "sports officiating",
    ],
    "conflict, war and peace": [
        "act of terror", "armed conflict", "civil unrest", "coup d'etat",
        "massacre", "peace process", "post-war reconstruction", "cyber warfare",
        "war victims",
    ],
    "weather": [
        "weather forecast", "weather phenomena", "weather statistic",
        "weather warning",
    ],
}


def _format_taxonomy(taxonomy: dict) -> str:
    """Human-readable category list for the prompt (instances shown only as hints)."""
    lines = []
    for cat, instances in taxonomy.items():
        if instances:
            lines.append(f"- {cat} (es.: {', '.join(instances)})")
        else:
            lines.append(f"- {cat}")
    return "\n".join(lines)


TAXONOMY_BLOCK = _format_taxonomy(TAXONOMY)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BATCH_SIZE = 32          # per-batch generation size; 8-16 is safe for 24-32B in bf16
MAX_LENGTH = 4096       # truncation ceiling for the tokenized prompt

ANNOTATION_PROMPT = """
Sei un annotatore esperto nella classificazione tematica di articoli di notizie in italiano.

Il tuo compito è assegnare all'ARTICOLO UNA SOLA categoria tematica, scegliendola esclusivamente dalla lista di categorie qui sotto.

CATEGORIE DISPONIBILI

Scegli una sola tra le seguenti categorie. Tra parentesi sono riportati alcuni esempi di sottotemi che appartengono a quella categoria: servono solo a chiarire cosa copre la categoria e NON sono opzioni di risposta valide.

{taxonomy}

TASK DI ANNOTAZIONE

Leggi attentamente il titolo e la descrizione dell'articolo, individua il tema principale e assegna la categoria più pertinente tra quelle elencate sopra.

ISTRUZIONI IMPORTANTI

- Considera l'articolo nel suo complesso e identifica il suo argomento PRINCIPALE.
- Scegli una sola categoria, quella che meglio rappresenta il contenuto.
- Rispondi ESCLUSIVAMENTE con il nome ESATTO di una categoria, copiato letteralmente dalla lista (senza gli esempi tra parentesi).
- Non inventare nuove categorie e non rispondere con i sottotemi tra parentesi.
- Non fornire spiegazioni, punteggiatura extra o testo aggiuntivo.

FORMATO INPUT

ARTICLE TITLE:
{article_title}

ARTICLE DESCRIPTION:
{article_description}

FORMATO OUTPUT

Restituisci SOLO il nome di una categoria dalla lista, ad esempio:

politics and government
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
    max_new_tokens: int = 24        # room for a full multi-word category label;
                                    # thinking models (ANITA) need far more
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
    # "qwen_vl": ModelSpec( -> DID NOT MANAGE TO ISNTALL TORCHVISION
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
# Topic normalisation / lookup
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """Lower, collapse whitespace, normalise apostrophes — for robust matching."""
    s = str(s).strip().lower().replace("\u2019", "'")
    return re.sub(r"\s+", " ", s)


# Category names take precedence over instance names on any ambiguity.
_CAT_BY_NORM = {_norm(c): c for c in TAXONOMY}
_CATEGORY_KEYS = sorted(_CAT_BY_NORM, key=len, reverse=True)

_INSTANCE_LOOKUP: dict = {}
for _cat, _insts in TAXONOMY.items():
    for _inst in _insts:
        _INSTANCE_LOOKUP.setdefault(_norm(_inst), _cat)
_INSTANCE_KEYS = sorted(_INSTANCE_LOOKUP, key=len, reverse=True)


def parse_topic(raw: str):
    """Map a model output to one of the 17 categories. None if nothing matches.

    Order: exact category -> exact instance -> category substring (longest first)
    -> instance substring (longest first). Categories always win over instances
    so a clean 'society' is never shadowed by a longer instance string."""
    if not raw:
        return None
    text = _norm(raw)

    if text in _CAT_BY_NORM:
        return _CAT_BY_NORM[text]
    if text in _INSTANCE_LOOKUP:
        return _INSTANCE_LOOKUP[text]
    for key in _CATEGORY_KEYS:
        if key in text:
            return _CAT_BY_NORM[key]
    for key in _INSTANCE_KEYS:
        if key in text:
            return _INSTANCE_LOOKUP[key]
    return None


def canonical_topic(raw):
    """Normalise a GOLD topic value to its top-level category.
    Accepts a category, an instance, or (defensively) a 1-element list.
    Unknown strings are returned as-is so they surface as their own class."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
        if raw is None:
            return None
    text = _norm(raw)
    if text in _CAT_BY_NORM:
        return _CAT_BY_NORM[text]
    if text in _INSTANCE_LOOKUP:
        return _INSTANCE_LOOKUP[text]
    return str(raw)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def create_prompt(row) -> str:
    return ANNOTATION_PROMPT.format(
        taxonomy=TAXONOMY_BLOCK,
        article_title=row["video_title"] if row["video_title"] else "N/A",
        article_description=row["video_description"] if row["video_description"] else "N/A",
    )


def load_gold_data(use_fake_data: bool = True) -> pd.DataFrame:
    """Load every gold-annotated article from annotated_metadata.

    Each metadata JSON describes one article (title, description) and carries the
    gold label in its "topic" field. One row == one article."""
    base = "VideosComments_fake" if use_fake_data else "VideosComments"
    input_gold_metadata_dir = f"{base}/youtube/annotated_metadata"

    rows = []
    for newspaper in newspapers:
        metadata_dir = os.path.join(input_gold_metadata_dir, newspaper)
        if not os.path.isdir(metadata_dir):
            continue
        # Only load gold ones
        for filename in sorted(os.listdir(metadata_dir)):
            if not filename.endswith("_gold.json"):
                continue

            with open(os.path.join(metadata_dir, filename), "r", encoding="utf-8") as f:
                metadata = json.load(f)

            topic = metadata.get("topic")
            if topic is None or (isinstance(topic, str) and not topic.strip()):
                # No gold topic on this article -> not usable for selection/eval.
                print(f"WARNING: no 'topic' field in {newspaper}/{filename} — skipped")
                continue

            video_title = metadata.get("title", "") or ""
            video_description = metadata.get("description", "") or ""

            # Truncate long descriptions at the last sentence boundary before 1000 chars.
            if len(video_description) > 1000:
                trunc_point = video_description.rfind(".", 0, 1000)
                video_description = (
                    video_description[:trunc_point + 1] if trunc_point != -1
                    else video_description[:1000]
                )

            rows.append({
                "newspaper": newspaper,
                "filename": filename,
                "video_title": video_title,
                "video_description": video_description,
                "topic_raw": topic,                  # original gold value, kept for audit
                "topic": canonical_topic(topic),     # mapped to a top-level category
            })

    if not rows:
        raise ValueError(f"No annotated metadata with a 'topic' field found under {input_gold_metadata_dir}")

    df = pd.DataFrame(rows)
    df["annotation_prompt"] = df.apply(create_prompt, axis=1)
    return df


def make_dev_test_split(gold_df, test_size: float = 0.4, seed: int = 42):
    """
    60/40 dev/test, stratified on topic so rare categories are not all dumped
    into one side. With many categories a stratum may be too small to split
    (train_test_split needs >= 2 members per stratum); fall back to a plain
    random split in that case.
    """
    strat = gold_df["topic"]
    if strat.value_counts().min() < 2:
        print("WARNING: some topics have < 2 examples — splitting without stratification")
        strat = None
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
    model = spec.model_class.from_pretrained(spec.name, token=HF_TOKEN, **load_kwargs)
    model.eval()

    # For a VLM the real tokenizer is proc.tokenizer; for a text model proc IS it.
    tok = proc.tokenizer if spec.is_vlm else proc
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, proc


def run_inference(model, proc, spec: ModelSpec, prompts: list,
                  batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH,
                  test: bool = False) -> list:
    """Render -> batch-tokenize -> generate -> decode new tokens -> parse topic."""
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
        preds.extend(parse_topic(d) for d in decoded)

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
    """Drop unparseable predictions, report, return macro-F1 across categories."""
    n_none = sum(p is None for p in y_pred)
    if n_none:
        print(f"  [{label}] {n_none}/{len(y_pred)} unparseable — excluded from metrics")

    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        print(f"  [{label}] no parseable predictions — macro-F1 = 0")
        return {"macro_f1": 0.0, "report": "no parseable predictions", "n_unparseable": n_none}

    yt, yp = zip(*pairs)
    report = classification_report(yt, yp, digits=4, zero_division=0)
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    print(f"\n--- {label} ---\n{report}\nMacro-F1: {macro_f1:.4f}")
    return {"macro_f1": macro_f1, "report": report, "n_unparseable": n_none}


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
            "topic": labels,
            "pred": preds,
        })
        df_out.to_csv(f"try_{key}.csv", index=False)

    metrics = score(labels, preds, label=f"{key} | dev")
    metrics.update(key=key, preds=preds)
    return metrics


def select_and_evaluate(dev_df, test_df, model_keys, batch_size: int = BATCH_SIZE, test: bool = False) -> dict:
    """Run every model on dev, pick best macro-F1, evaluate that one on test."""
    dev_prompts, dev_labels = dev_df["annotation_prompt"].tolist(), dev_df["topic"].tolist()

    dev_results = [run_one_model(k, dev_prompts, dev_labels, batch_size, test=test) for k in model_keys]

    best = max(dev_results, key=lambda m: m["macro_f1"])
    print(f"\n{'='*60}\nBest on dev: {best['key']}  (macro-F1={best['macro_f1']:.4f})\n{'='*60}")

    # Final, single evaluation on the untouched test split.
    test_prompts, test_labels = test_df["annotation_prompt"].tolist(), test_df["topic"].tolist()
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
        "dev_macro_f1": best["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "dev_report": best["report"],
        "test_report": test_metrics["report"],
    }
    with open("results_summary_topics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved -> results_summary_topics.json")
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    load_dotenv()  # for Hugging Face API keys, if needed

    gold_df = load_gold_data(use_fake_data=True)   # flip to False for the real run
    # gold_df.to_csv("try.csv", index=False)         # keep for inspection

    # TODO: For small testing runs
    test = True
    if test:
        n = min(50, len(gold_df))
        gold_df = gold_df.sample(n, random_state=42).reset_index(drop=True)

    dev_df, test_df = make_dev_test_split(gold_df)
    print(f"Correctly split into dev and test set")
    select_and_evaluate(dev_df, test_df, model_keys=list(REGISTRY), batch_size=BATCH_SIZE, test=test)


if __name__ == "__main__":
    main()