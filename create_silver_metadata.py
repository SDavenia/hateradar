####
# Article topic SILVER annotation via zero-shot prompting.
#
# Unlike the selection script, nothing is split or compared here. A single,
# already-chosen "best" model (passed with --model) is loaded once and used to
# assign a topic to every article that does NOT already have a human (gold) one.
#
# For each raw metadata file  metadata/<newspaper>/<id>.json:
#   - if annotated_metadata/<newspaper>/<id>_gold.json exists -> skip (human-labelled)
#   - else run the model and write annotated_metadata/<newspaper>/<id>_silver.json
#
# The silver file mirrors the raw metadata's contents + a `topic` field
# (one of the 17 top-level categories, "" = unparseable model output).
#
# NOTE on batching: each metadata file is a SINGLE article, so unlike the
# comment pipeline (many comments per file) we collect all pending articles
# first and run ONE batched inference pass — labelling them one file at a time
# would mean a batch size of 1 and waste the GPU.
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
# Taxonomy
# --------------------------------------------------------------------------- #
# Top-level entries (keys) are the CATEGORIES the model must choose from.
# The lists are example instances/subtopics, used only to describe what each
# category covers — they are NOT valid answers (a model that emits an instance
# is mapped back to its parent category by parse_topic).

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

BATCH_SIZE = 8          # per-batch generation size; 8-16 is safe for 24-32B in bf16
MAX_LENGTH = 4096        # truncation ceiling for the tokenized prompt

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


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def read_metadata(path: str) -> dict:
    """Load a raw article metadata JSON as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt_fields(metadata: dict):
    """Extract (title, description) for the prompt; description truncated at the
    last sentence boundary before 1000 chars. The stored metadata is left intact
    — truncation only affects what the model sees."""
    title = metadata.get("title", "") or ""
    description = metadata.get("description", "") or ""
    if len(description) > 1000:
        trunc_point = description.rfind(".", 0, 1000)
        description = description[:trunc_point + 1] if trunc_point != -1 else description[:1000]
    return title, description


def create_prompt(title: str, description: str) -> str:
    return ANNOTATION_PROMPT.format(
        taxonomy=TAXONOMY_BLOCK,
        article_title=title if title else "N/A",
        article_description=description if description else "N/A",
    )


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


def run_inference(model, proc, spec: ModelSpec, prompts: list,
                  batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH) -> list:
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
        preds.extend(parse_topic(d) for d in decoded)

    print()
    return preds


# --------------------------------------------------------------------------- #
# Silver annotation
# --------------------------------------------------------------------------- #

def collect_pending(metadata_dir: str, annotated_dir: str, overwrite: bool):
    """
    Walk every raw metadata file and return the list of articles that still need
    a silver topic (no gold, and no existing silver unless --overwrite). Each item
    carries enough to write its silver file later.
    """
    pending = []
    n_skipped_gold = n_skipped_silver = n_errors = 0

    for newspaper in newspapers:
        metadata_np = os.path.join(metadata_dir, newspaper)
        annotated_np = os.path.join(annotated_dir, newspaper)

        if not os.path.isdir(metadata_np):
            print(f"WARNING: no metadata dir for '{newspaper}' ({metadata_np}) — skipping")
            continue
        os.makedirs(annotated_np, exist_ok=True)

        for filename in sorted(os.listdir(metadata_np)):
            # Only raw metadata files: plain <id>.json, never *_gold/_silver.json.
            if not filename.endswith(".json"):
                continue
            if filename.endswith("_gold.json") or filename.endswith("_silver.json"):
                continue

            file_id = filename[:-5]                      # strip ".json"
            gold_path = os.path.join(annotated_np, f"{file_id}_gold.json")
            silver_path = os.path.join(annotated_np, f"{file_id}_silver.json")

            if os.path.exists(gold_path):
                n_skipped_gold += 1
                continue                                 # human-annotated already
            if os.path.exists(silver_path) and not overwrite:
                n_skipped_silver += 1
                continue                                 # resume: already done

            try:
                metadata = read_metadata(os.path.join(metadata_np, filename))
            except Exception as e:
                n_errors += 1
                print(f"  ERROR reading {newspaper}/{file_id}: {e}")
                continue

            title, description = prompt_fields(metadata)
            pending.append({
                "newspaper": newspaper,
                "file_id": file_id,
                "metadata": metadata,
                "silver_path": silver_path,
                "prompt": create_prompt(title, description),
            })

    return pending, n_skipped_gold, n_skipped_silver, n_errors


def annotate_silver(model, proc, spec: ModelSpec,
                    metadata_dir: str, annotated_dir: str,
                    batch_size: int = BATCH_SIZE, overwrite: bool = False):
    """
    Gather every article still missing a topic, label them all in one batched
    pass, and write <id>_silver.json (raw metadata + predicted `topic`) next to gold.
    """
    pending, n_skipped_gold, n_skipped_silver, n_errors = collect_pending(
        metadata_dir, annotated_dir, overwrite
    )

    print(f"\n{len(pending)} articles to annotate "
          f"(skipped gold: {n_skipped_gold}, existing silver: {n_skipped_silver}, "
          f"read errors: {n_errors})")

    n_written = n_none = 0
    if pending:
        preds = run_inference(
            model, proc, spec, [p["prompt"] for p in pending], batch_size=batch_size
        )

        for item, pred in zip(pending, preds):
            topic = pred if pred is not None else ""     # "" mirrors "unparseable"
            if pred is None:
                n_none += 1
            out = dict(item["metadata"])                 # preserve raw schema...
            out["topic"] = topic                         # ...+ the predicted topic
            try:
                with open(item["silver_path"], "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                n_written += 1
            except Exception as e:
                n_errors += 1
                print(f"  ERROR writing {item['silver_path']}: {e}")

        if n_none:
            print(f"{n_none}/{len(preds)} unparseable model outputs — written with empty topic")

    print(f"\n{'='*60}")
    print(f"Done. silver written: {n_written} | "
          f"skipped (gold): {n_skipped_gold} | "
          f"skipped (existing silver): {n_skipped_silver} | "
          f"errors: {n_errors}")
    print(f"{'='*60}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Silver-annotate article topics with the best model.")
    ap.add_argument("--model", required=True, choices=list(REGISTRY),
                    help="Which (best) model to use for silver annotation.")
    ap.add_argument("--use-fake-data", action="store_true",
                    help="Use the VideosComments_fake tree instead of VideosComments.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-annotate even if a *_silver.json already exists.")
    return ap.parse_args()


def main():
    load_dotenv()  # for Hugging Face API keys, if needed
    args = parse_args()

    base = "VideosComments_fake" if args.use_fake_data else "VideosComments"
    metadata_dir = f"{base}/youtube/metadata"
    annotated_dir = f"{base}/youtube/annotated_metadata"

    spec = REGISTRY[args.model]
    print(f"Loading best model: {args.model} ({spec.name})")
    model, proc = load_model(spec)
    try:
        annotate_silver(
            model, proc, spec,
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