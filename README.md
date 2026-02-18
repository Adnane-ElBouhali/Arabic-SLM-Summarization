# Fine-tuning a Small Language Model for Arabic Summarization

This project was done in the class **Introduction to Text Mining and NLP (CSC_52082_EP)** at **École Polytechnique**.

This repository contains the code and data for a project that fine-tunes a **small decoder-only language model** for **Arabic news summarization**, and compares multiple fine-tuning strategies under **limited GPU resources** (e.g., Google Colab free-tier).

## Project overview

### Dataset
- Based on **XLSum** (BBC news article–summary pairs).
- This project uses a **filtered subset of 5,000 Arabic articles** stored as a CSV.
- The dataset was additionally **annotated with synthetic “ground-truth” summaries** generated using a larger instruction-tuned model, with some post-processing to improve quality.

### Fine-tuning methods implemented
This repo implements and compares:
- **Full fine-tuning** (update all parameters)
- **LoRA** (parameter-efficient adapters)
- **QLoRA** (4-bit quantization + LoRA)
- **Prefix-tuning** (soft prompts / virtual tokens)

### Evaluation
- Automatic metrics: **ROUGE-1/2/L** + **BERTScore** (Arabic).
- Baseline vs fine-tuned models.

---

## Repository structure

```

.
├── Data/
│   └── arabic_summaries_5000_v2.csv
└── Code/
├── create_annotated_data.ipynb
├── example.ipynb
├── normal_finetuning.py
├── qlora_finetuning.py
├── prefix_finetuning.py
├── evaluation.py
├── utils.py
└── requirements.txt

````

---

## Setup

### 1) Create an environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
````

### 2) Install dependencies

From the `Code/` folder:

```bash
pip install -r requirements.txt
```

> Note: Some configurations assume **CUDA** is available (training/evaluation moves models to `"cuda"`).

### (Optional) Weights & Biases logging

Some scripts can log to W&B if `WANDB_API_KEY` is set:

```bash
export WANDB_API_KEY="YOUR_KEY"
```

---

## Data

* CSV location: `Data/arabic_summaries_5000_v2.csv`
* Expected columns used by the code: **`text`** and **`summary`**.

---

## How it works

### Preprocessing (key detail)

This project fine-tunes a **causal LM** by concatenating:

* prompt + template + article text → then appending the **target summary**
* labels are masked (`-100`) so the loss is computed **only on the summary tokens**

Key helper functions live in `utils.py`:

* `preprocess_function(...)`
* `compute_metrics(...)` (ROUGE)
* `CustomCallback(...)` (training curves + logs)
* `log_hardware(...)`

---

## Training

The scripts expose `finetune_model(...)` functions that you can call from notebooks (recommended) or your own runner code.

### Full fine-tuning

File: `Code/normal_finetuning.py`

What it does:

* Loads a base model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
* Preprocesses train/val
* Trains with Hugging Face Trainer (FP16, gradient accumulation, ROUGE eval, optional W&B)
* Saves logs + hardware stats + metrics plots

### Prefix-tuning

File: `Code/prefix_finetuning.py`

What it does:

* Wraps the base model with PEFT Prefix-Tuning (`PrefixTuningConfig`)
* Supports number of virtual tokens and optional prefix projection
* Uses the same preprocessing & Trainer flow

### LoRA / QLoRA

File: `Code/qlora_finetuning.py`

What it does:

* `method="lora"`: standard LoRA adapters
* `method="qlora"`: loads the model in **4-bit** using BitsAndBytes config (`nf4`) + prepares for k-bit training, then applies LoRA
* Uses the same Trainer flow + ROUGE evaluation

---

## Evaluation

File: `Code/evaluation.py`

What it computes:

* Generates summaries in batches using `model.generate(...)`
* Extracts the summary portion using a delimiter (`output_split`)
* Reports:

  * **Average ROUGE-1/2/L**
  * **Average BERTScore (P/R/F1)** with `lang="ar"`

---

## Notebooks

* `Code/create_annotated_data.ipynb`: dataset annotation workflow (synthetic summaries generation pipeline).
* `Code/example.ipynb`: example end-to-end usage (loading data, training, evaluation).

---
