# OncoChat Fine-Tuned LLM

A specialized pipeline designed to fine-tune small language models (SLMs) on FDA cancer drug labels. This tool extracts structured medical information from PDFs, generates instruction-tuning datasets, and applies Parameter-Efficient Fine-Tuning (PEFT) using LoRA to create a domain-specific oncology chatbot.

---

## 🚀 Features

* **PDF Ingestion:** Reuses logic from `OncoChat-RAG.py` to extract text and detect specific FDA label sections (e.g., Indications, Dosage, Adverse Reactions).
* **Instruction Generation:** Automatically creates Q&A pairs from structured drug label sections using clinical templates.
* **Memory-Efficient Training:** Utilizes **4-bit quantization (BitsAndBytes)** and **LoRA** to fine-tune 7B parameter models (like BioMistral) on consumer-grade hardware.
* **Evaluation Suite:** Includes automated metrics for drug name recall, section relevance, and response coherence.
* **Interactive Chat:** A built-in CLI interface to query the fine-tuned model.

---

## 🛠️ Installation

Ensure you have a Python environment (3.9+) and a CUDA-capable GPU.

```bash
pip install transformers peft datasets torch accelerate pypdf bitsandbytes
```

---

---

## 🛠️ Training

Training of this model took more than 10 hours on 2 RTX 3090 (48 GB VRAM).

---


## 📂 Project Structure

* `drug_reports/`: Directory containing source FDA drug label PDFs. (https://github.com/mohbot/OncoChat-RAG/tree/main/drug_reports)
* `training_data/`: Generated JSONL files for training and testing.
* `oncochat_biomistral/`: Output directory for the fine-tuned LoRA weights.

---

## 📖 Usage

The script is controlled via a CLI with several commands.

### 1. Complete Pipeline
To run extraction, training, evaluation, and testing in one go:
```bash
python OncoChat-FineTune.py all
```

### 2. Data Preparation
Extract sections from PDFs and generate `train.jsonl` and `test.jsonl`:
```bash
python OncoChat-FineTune.py prepare --pdf-dir ./my_pdfs
```

### 3. Fine-Tuning
Start the training process (defaulting to `BioMistral-7B`):
```bash
python OncoChat-FineTune.py train --epochs 3 --batch-size 2
```

### 4. Interactive Chat
Talk to your fine-tuned model:
```bash
python OncoChat-FineTune.py chat
```

---

## ⚙️ Configuration

Key hyperparameters can be adjusted within the script constants:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `BASE_MODEL` | `BioMistral/BioMistral-7B` | The foundation model from HuggingFace. |
| `LORA_R` | `16` | Rank of the LoRA update matrices. |
| `BATCH_SIZE` | `2` | Batch size per device (keep low for 24GB VRAM). |
| `MAX_SEQ_LENGTH` | `512` | Maximum tokens per training sample. |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Used to simulate larger batch sizes. |

---

## 📊 Evaluation Metrics

The `evaluate` command tracks:
* **Drug Name Recall:** Does the model identify the correct drug in its response?
* **Section Keyword Hit:** Does the response use terminology appropriate for the section (e.g., "pharmacokinetics" for Clinical Pharmacology)?
* **No Repetition:** Measures response coherence and lack of "looping" text.

---

## ⚠️ Disclaimer
*This tool is for research purposes only. The generated responses are based on AI interpretations of drug labels and should not be used as professional medical advice.*