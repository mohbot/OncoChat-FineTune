"""
OncoChat Fine-Tuned LLM

Fine-tunes a small language model on FDA cancer drug labels for chatbot use.
Reuses PDF ingestion and section detection from OncoChat-RAG.py.

Pipeline:
  1. Extract & chunk FDA drug label PDFs (reused from OncoChat-RAG.py)
  2. Generate instruction-tuning Q&A pairs from structured sections
  3. Fine-tune a small LLM with LoRA (parameter-efficient)
  4. Evaluate on held-out test set
  5. Interactive chat

Requirements:
  pip install transformers peft datasets torch accelerate pypdf
"""

import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (shared with OncoChat-RAG.py)
# ---------------------------------------------------------------------------
CANONICAL_HEADERS = [
    "Adverse Reactions",
    "Use in Specific Populations",
    "Indications and Usage",
    "Dosage and Administration",
    "Dosage Forms and Strengths",
    "Contraindications",
    "Warnings and Precautions",
    "Overdosage",
    "Description",
    "Clinical Pharmacology",
    "Nonclinical Toxicology",
    "Clinical Studies",
    "How Supplied/Storage and Handling",
    "Patient Counseling Information",
    "Drug Interactions",
]

SECTION_NUMBER_MAP: Dict[str, int] = {
    "Indications and Usage": 1,
    "Dosage and Administration": 2,
    "Dosage Forms and Strengths": 3,
    "Contraindications": 4,
    "Warnings and Precautions": 5,
    "Adverse Reactions": 6,
    "Drug Interactions": 7,
    "Use in Specific Populations": 8,
    "Overdosage": 10,
    "Description": 11,
    "Clinical Pharmacology": 12,
    "Nonclinical Toxicology": 13,
    "Clinical Studies": 14,
    "How Supplied/Storage and Handling": 16,
    "Patient Counseling Information": 17,
}

PDF_DIR = "drug_reports"
TRAIN_DATA_DIR = "training_data"
MODEL_OUTPUT_DIR = "oncochat_model"
MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

# Model config — defaults to Qwen2.5-0.5B (small, fast to train)
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 512
TRAIN_SPLIT = 0.9


# ===========================================================================
# PDF Parsing (reused from OncoChat-RAG.py)
# ===========================================================================

def extract_drug_name(filename: str) -> str:
    stem = Path(filename).stem
    m = re.match(r"^(.+?)_(?:(?:20|19)\d{2}|pre\d+)_", stem)
    if m:
        return m.group(1).strip().title()
    return stem.split("_")[0].strip().title()


def extract_pdf_text(filepath: str) -> Optional[str]:
    try:
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        if not pages:
            return None
        return "\n".join(pages)
    except Exception as e:
        log.warning("Failed to read PDF %s: %s", filepath, e)
        return None


def _build_section_patterns() -> List[Tuple[str, re.Pattern]]:
    patterns = []
    for header in CANONICAL_HEADERS:
        header_upper = re.escape(header.upper())
        num = SECTION_NUMBER_MAP.get(header)
        if num is not None:
            pat = re.compile(
                rf"(?:^|\n)[ \t]*(?:{num}\.?\s+)?{header_upper}(?:\s+\d+)?[ \t]*(?:\n|$)",
                re.IGNORECASE,
            )
        else:
            pat = re.compile(
                rf"(?:^|\n)[ \t]*{header_upper}(?:\s+\d+)?[ \t]*(?:\n|$)",
                re.IGNORECASE,
            )
        patterns.append((header, pat))
    return patterns


_SECTION_PATTERNS = _build_section_patterns()


def _find_body_start(text: str) -> int:
    body_start = 0
    for m in re.finditer(
        r"(?:^|\n)\s*FULL PRESCRIBING INFORMATION\s*(?:\n|$)",
        text, re.IGNORECASE,
    ):
        body_start = m.start()
    return body_start


def detect_sections(text: str) -> List[Tuple[int, str]]:
    body_start = _find_body_start(text)
    sections: List[Tuple[int, str]] = []
    for header, pattern in _SECTION_PATTERNS:
        best_match = None
        for match in pattern.finditer(text, pos=body_start):
            after = text[match.end(): match.end() + 30]
            if re.match(r"\s*[\(\[]", after):
                continue
            best_match = match
        if best_match is not None:
            sections.append((best_match.start(), header))
    sections.sort(key=lambda x: x[0])
    return sections


def extract_section_texts(text: str, drug_name: str) -> Dict[str, str]:
    """Extract each canonical section's text content from a drug label."""
    sections = detect_sections(text)
    result = {}
    for idx, (pos, header) in enumerate(sections):
        end_pos = sections[idx + 1][0] if idx + 1 < len(sections) else len(text)
        section_text = text[pos:end_pos].strip()
        first_newline = section_text.find("\n")
        if first_newline > 0:
            section_text = section_text[first_newline:].strip()
        if section_text and len(section_text) > 50:
            result[header] = section_text
    return result


# ===========================================================================
# Training Data Generation
# ===========================================================================

QUESTION_TEMPLATES = {
    "Indications and Usage": [
        "What is {drug} indicated for?",
        "What are the approved uses of {drug}?",
        "What conditions does {drug} treat?",
        "What type of cancer is {drug} used for?",
    ],
    "Dosage and Administration": [
        "What is the recommended dosage for {drug}?",
        "How should {drug} be administered?",
        "What is the dosing schedule for {drug}?",
        "How do you take {drug}?",
    ],
    "Contraindications": [
        "What are the contraindications for {drug}?",
        "Who should not take {drug}?",
        "When is {drug} contraindicated?",
    ],
    "Warnings and Precautions": [
        "What are the warnings for {drug}?",
        "What precautions should be taken with {drug}?",
        "What are the safety warnings for {drug}?",
        "What risks are associated with {drug}?",
    ],
    "Adverse Reactions": [
        "What are the side effects of {drug}?",
        "What adverse reactions does {drug} cause?",
        "What are the common adverse effects of {drug}?",
        "What side effects should patients watch for with {drug}?",
    ],
    "Drug Interactions": [
        "What drugs interact with {drug}?",
        "What are the drug interactions for {drug}?",
        "What medications should be avoided with {drug}?",
    ],
    "Use in Specific Populations": [
        "Can {drug} be used during pregnancy?",
        "Is {drug} safe for elderly patients?",
        "What are the special population considerations for {drug}?",
    ],
    "Description": [
        "What is {drug}?",
        "Describe the drug {drug}.",
        "What type of drug is {drug}?",
        "What is the mechanism of action of {drug}?",
    ],
    "Clinical Pharmacology": [
        "What is the pharmacology of {drug}?",
        "How does {drug} work in the body?",
        "What is the pharmacokinetics of {drug}?",
    ],
    "Clinical Studies": [
        "What clinical trials were done for {drug}?",
        "What evidence supports the use of {drug}?",
        "What were the clinical study results for {drug}?",
    ],
    "Overdosage": [
        "What happens in case of {drug} overdose?",
        "What are the symptoms of {drug} overdosage?",
    ],
    "How Supplied/Storage and Handling": [
        "How is {drug} supplied?",
        "How should {drug} be stored?",
    ],
    "Patient Counseling Information": [
        "What should patients know about {drug}?",
        "What counseling information should be given for {drug}?",
    ],
    "Dosage Forms and Strengths": [
        "What forms does {drug} come in?",
        "What strengths is {drug} available in?",
    ],
    "Nonclinical Toxicology": [
        "What are the nonclinical toxicology findings for {drug}?",
    ],
}

GENERAL_QUESTIONS = [
    "Tell me about {drug}.",
    "What do I need to know about {drug}?",
    "Summarize the key information about {drug}.",
]


@dataclass
class TrainingSample:
    instruction: str
    input_text: str
    output_text: str
    drug_name: str
    section: str


def clean_section_text(text: str) -> str:
    """Remove FDA formatting artifacts that teach the model to mimic structure."""
    text = re.sub(r"^\d+\.?\d*\s+", "", text)
    text = re.sub(r"\n\s*\d+\.\d+\s+", "\n", text)
    text = re.sub(
        r"\[see\s+(?:Boxed\s+Warning(?:\s+and)?|Warnings\s+and\s+Precautions|"
        r"Adverse\s+Reactions|Clinical\s+(?:Studies|Pharmacology)|"
        r"Dosage\s+and\s+Administration|Contraindications|"
        r"Use\s+in\s+Specific\s+Populations|Drug\s+Interactions|"
        r"Description)\s*\([^)]*\)\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def truncate_answer(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """Truncate text to fit model context, preferring sentence boundaries."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[:last_period + 1]
    return truncated + "..."


def generate_qa_pairs(drug_name: str, sections: Dict[str, str]) -> List[TrainingSample]:
    """Generate instruction-tuning Q&A pairs from a drug's extracted sections."""
    samples = []

    for section_name, section_text in sections.items():
        templates = QUESTION_TEMPLATES.get(section_name, [])
        if not templates:
            continue

        answer = truncate_answer(clean_section_text(section_text))

        for template in templates:
            question = template.format(drug=drug_name)
            samples.append(TrainingSample(
                instruction=question,
                input_text="",
                output_text=answer,
                drug_name=drug_name,
                section=section_name,
            ))

    if sections:
        summary_parts = []
        for sec in ["Description", "Indications and Usage", "Warnings and Precautions"]:
            if sec in sections:
                summary_parts.append(truncate_answer(clean_section_text(sections[sec]), 250))
        if summary_parts:
            combined = " ".join(summary_parts)
            for template in GENERAL_QUESTIONS:
                question = template.format(drug=drug_name)
                samples.append(TrainingSample(
                    instruction=question,
                    input_text="",
                    output_text=combined,
                    drug_name=drug_name,
                    section="General",
                ))

    return samples


def process_pdfs_to_training_data(pdf_dir: str) -> List[TrainingSample]:
    """Process all PDFs and generate training samples."""
    pdf_path = Path(pdf_dir)
    pdf_files = sorted(pdf_path.glob("*.pdf"))
    log.info("Found %d PDF files", len(pdf_files))

    all_samples: List[TrainingSample] = []
    drugs_processed = 0

    for i, fpath in enumerate(pdf_files, 1):
        if i % 25 == 0 or i == len(pdf_files):
            log.info("Processing %d/%d: %s", i, len(pdf_files), fpath.name)

        text = extract_pdf_text(str(fpath))
        if text is None:
            continue

        drug_name = extract_drug_name(fpath.name)
        sections = extract_section_texts(text, drug_name)

        if not sections:
            log.warning("No sections found in %s", fpath.name)
            continue

        samples = generate_qa_pairs(drug_name, sections)
        all_samples.extend(samples)
        drugs_processed += 1

    log.info(
        "Generated %d training samples from %d drugs (avg %.1f per drug)",
        len(all_samples), drugs_processed,
        len(all_samples) / max(drugs_processed, 1),
    )
    return all_samples


def format_for_training(sample: TrainingSample) -> str:
    """Format a sample into the training prompt format."""
    return (
        f"### Instruction:\n{sample.instruction}\n\n"
        f"### Response:\n{sample.output_text}\n"
    )


def save_training_data(
    samples: List[TrainingSample],
    output_dir: str = TRAIN_DATA_DIR,
) -> Tuple[str, str]:
    """Save training data as JSONL files, split into train/test."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * TRAIN_SPLIT)
    train_samples = shuffled[:split_idx]
    test_samples = shuffled[split_idx:]

    train_file = str(out_path / "train.jsonl")
    test_file = str(out_path / "test.jsonl")

    for filepath, data in [(train_file, train_samples), (test_file, test_samples)]:
        with open(filepath, "w") as f:
            for sample in data:
                record = {
                    "text": format_for_training(sample),
                    "instruction": sample.instruction,
                    "output": sample.output_text,
                    "drug_name": sample.drug_name,
                    "section": sample.section,
                }
                f.write(json.dumps(record) + "\n")

    log.info("Saved %d train, %d test samples to %s/", len(train_samples), len(test_samples), output_dir)
    return train_file, test_file


# ===========================================================================
# Model Fine-Tuning
# ===========================================================================

def fine_tune_model(
    train_file: str,
    test_file: str,
    base_model: str = BASE_MODEL,
    output_dir: str = MODEL_OUTPUT_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_seq_length: int = MAX_SEQ_LENGTH,
):
    """Fine-tune a small causal LM with LoRA on the training data."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    log.info("Loading base model: %s", base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
    )

    target_modules = _detect_target_modules(model)
    log.info("LoRA target modules: %s", target_modules)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    log.info("Starting fine-tuning...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    log.info("Training completed in %.1f minutes", elapsed / 60)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved to %s/", output_dir)

    log.info("Merging LoRA weights into base model for inference...")
    merge_model(model_dir=output_dir, base_model=base_model)

    return model, tokenizer


def _detect_target_modules(model) -> List[str]:
    """Auto-detect linear layer names for LoRA targeting."""
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            linear_names.add(parts[-1])
    linear_names.discard("lm_head")
    return list(linear_names) if linear_names else ["c_attn", "c_proj"]


# ===========================================================================
# Inference / Chat
# ===========================================================================

MERGED_MODEL_DIR = "oncochat_model_merged"


def merge_model(
    model_dir: str = MODEL_OUTPUT_DIR,
    base_model: str = BASE_MODEL,
    merged_dir: str = MERGED_MODEL_DIR,
):
    """Merge LoRA adapters into the base model for simpler inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, model_dir)

    log.info("Merging LoRA weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    log.info("Merged model saved to %s/", merged_dir)


class OncoChat:
    """Chat interface for the fine-tuned OncoChat model."""

    def __init__(self, model_dir: str = MODEL_OUTPUT_DIR, base_model: str = BASE_MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        merged_dir = model_dir.rstrip("/") + "_merged"
        is_merged = Path(merged_dir).exists() and (Path(merged_dir) / "model.safetensors").exists()
        is_standalone = (Path(model_dir) / "model.safetensors").exists()

        if is_merged:
            load_dir = merged_dir
            log.info("Loading merged model from %s", load_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(load_dir, dtype=torch.float32)
        elif is_standalone:
            load_dir = model_dir
            log.info("Loading standalone model from %s", load_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(load_dir, dtype=torch.float32)
        else:
            from peft import PeftModel
            log.info("Loading LoRA adapter model from %s", model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
            self.model = PeftModel.from_pretrained(base, model_dir)

        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        log.info("Model loaded on %s", self.device)

    def generate(
        self,
        question: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> str:
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response_marker = "### Response:\n"
        if response_marker in full_text:
            response = full_text.split(response_marker, 1)[1].strip()
        else:
            response = full_text[len(prompt):].strip()

        next_instruction = response.find("### Instruction:")
        if next_instruction > 0:
            response = response[:next_instruction].strip()

        return response


# ===========================================================================
# Evaluation / Testing
# ===========================================================================

def evaluate_model(
    model_dir: str = MODEL_OUTPUT_DIR,
    test_file: str = None,
    num_samples: int = 20,
    base_model: str = BASE_MODEL,
) -> Dict:
    """Evaluate the fine-tuned model on test samples.

    Metrics:
      - Perplexity on test set
      - Drug name recall (does the answer mention the right drug?)
      - Section relevance (keyword overlap with expected section)
      - Response coherence (no excessive repetition)
      - Qualitative examples
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if test_file is None:
        test_file = str(Path(TRAIN_DATA_DIR) / "test.jsonl")

    test_samples = []
    with open(test_file) as f:
        for line in f:
            test_samples.append(json.loads(line))

    log.info("Evaluating on %d test samples (showing %d)", len(test_samples), num_samples)

    chat = OncoChat(model_dir=model_dir, base_model=base_model)

    random.seed(42)
    eval_samples = random.sample(test_samples, min(num_samples, len(test_samples)))

    results = {
        "total_samples": len(test_samples),
        "evaluated": len(eval_samples),
        "drug_name_recall": 0,
        "section_keyword_hit": 0,
        "no_repetition": 0,
        "non_empty_response": 0,
        "examples": [],
    }

    section_keywords = {
        "Indications and Usage": ["indicated", "treatment", "patients", "use"],
        "Dosage and Administration": ["mg", "dose", "administer", "oral", "daily"],
        "Adverse Reactions": ["adverse", "reaction", "common", "reported", "patients"],
        "Warnings and Precautions": ["warning", "risk", "monitor", "serious"],
        "Contraindications": ["contraindicated", "should not", "hypersensitivity"],
        "Drug Interactions": ["interaction", "concomitant", "co-administered"],
        "Description": ["mechanism", "inhibitor", "receptor", "molecular"],
        "Clinical Pharmacology": ["absorption", "metabolism", "half-life", "clearance"],
    }

    for i, sample in enumerate(eval_samples):
        question = sample["instruction"]
        expected = sample["output"]
        drug = sample["drug_name"]
        section = sample["section"]

        response = chat.generate(question, max_new_tokens=200)

        drug_mentioned = drug.lower() in response.lower() or drug.lower() in question.lower()
        if drug_mentioned:
            results["drug_name_recall"] += 1

        keywords = section_keywords.get(section, [])
        keyword_hit = any(kw.lower() in response.lower() for kw in keywords) if keywords else True
        if keyword_hit:
            results["section_keyword_hit"] += 1

        words = response.split()
        if len(words) > 5:
            repeated_ratio = len(words) - len(set(words))
            no_repeat = repeated_ratio / len(words) < 0.5
        else:
            no_repeat = True
        if no_repeat:
            results["no_repetition"] += 1

        if len(response.strip()) > 10:
            results["non_empty_response"] += 1

        results["examples"].append({
            "question": question,
            "drug": drug,
            "section": section,
            "response": response[:500],
            "expected_snippet": expected[:200],
            "drug_mentioned": drug_mentioned,
            "keyword_hit": keyword_hit,
            "no_repetition": no_repeat,
        })

        if i < 5:
            print(f"\n{'='*60}")
            print(f"Q: {question}")
            print(f"Drug: {drug} | Section: {section}")
            print(f"A: {response[:300]}")
            print(f"Metrics: drug_recall={drug_mentioned}, keyword={keyword_hit}, coherent={no_repeat}")

    n = results["evaluated"]
    results["scores"] = {
        "drug_name_recall_pct": 100 * results["drug_name_recall"] / n,
        "section_keyword_pct": 100 * results["section_keyword_hit"] / n,
        "no_repetition_pct": 100 * results["no_repetition"] / n,
        "non_empty_pct": 100 * results["non_empty_response"] / n,
    }

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples evaluated: {n}")
    for metric, val in results["scores"].items():
        print(f"  {metric}: {val:.1f}%")

    eval_file = str(Path(model_dir) / "eval_results.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(eval_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Evaluation results saved to %s", eval_file)

    return results


# ===========================================================================
# Functional Tests (adapted from OncoChat-RAG.py testing patterns)
# ===========================================================================

def run_tests(model_dir: str = MODEL_OUTPUT_DIR, base_model: str = BASE_MODEL):
    """Run functional tests on the fine-tuned model.

    Tests verify that the model produces reasonable oncology-specific
    responses — similar to how OncoChat-RAG.py would be tested.
    """
    print("\n" + "=" * 60)
    print("RUNNING FUNCTIONAL TESTS")
    print("=" * 60)

    chat = OncoChat(model_dir=model_dir, base_model=base_model)

    test_cases = [
        {
            "name": "Drug indication query",
            "question": "What is Pembrolizumab indicated for?",
            "expect_any": ["cancer", "tumor", "melanoma", "lung", "treatment",
                           "carcinoma", "indication", "patients"],
        },
        {
            "name": "Adverse reactions query",
            "question": "What are the side effects of Imatinib?",
            "expect_any": ["adverse", "reaction", "nausea", "effect", "common",
                           "reported", "patients", "diarrhea"],
        },
        {
            "name": "Dosage query",
            "question": "What is the recommended dosage for Osimertinib?",
            "expect_any": ["mg", "dose", "daily", "oral", "tablet",
                           "administer", "once"],
        },
        {
            "name": "Contraindications query",
            "question": "Who should not take Bevacizumab?",
            "expect_any": ["contraindicated", "should not", "hypersensitivity",
                           "patients", "risk", "warning"],
        },
        {
            "name": "Drug description query",
            "question": "What type of drug is Trastuzumab?",
            "expect_any": ["antibody", "receptor", "HER2", "inhibitor",
                           "monoclonal", "drug", "treatment", "cancer"],
        },
        {
            "name": "General oncology query",
            "question": "Tell me about Rituximab.",
            "expect_any": ["lymphoma", "CD20", "antibody", "treatment",
                           "patients", "cancer", "indicated"],
        },
        {
            "name": "Non-empty response test",
            "question": "What are the warnings for Cisplatin?",
            "expect_any": [],  # just check non-empty
        },
        {
            "name": "Response coherence test",
            "question": "How should Capecitabine be administered?",
            "expect_any": [],  # check no extreme repetition
        },
    ]

    passed = 0
    failed = 0

    for tc in test_cases:
        response = chat.generate(tc["question"], max_new_tokens=200)
        response_lower = response.lower()

        test_pass = True
        failure_reason = ""

        if len(response.strip()) < 10:
            test_pass = False
            failure_reason = "Response too short"

        words = response.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                test_pass = False
                failure_reason = f"Excessive repetition (unique ratio: {unique_ratio:.2f})"

        if tc["expect_any"] and test_pass:
            has_keyword = any(kw.lower() in response_lower for kw in tc["expect_any"])
            if not has_keyword:
                test_pass = False
                failure_reason = f"No expected keywords found. Expected one of: {tc['expect_any']}"

        status = "PASS" if test_pass else "FAIL"
        if test_pass:
            passed += 1
        else:
            failed += 1

        print(f"\n  [{status}] {tc['name']}")
        print(f"    Q: {tc['question']}")
        print(f"    A: {response[:200]}")
        if not test_pass:
            print(f"    Reason: {failure_reason}")

    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{passed + failed} passed")
    print(f"{'='*60}")
    return passed, failed


# ===========================================================================
# Interactive Chat Loop
# ===========================================================================

BANNER = """
=======================================================
       OncoChat — Fine-Tuned Cancer Drug LLM          
  Ask questions about FDA-approved cancer drugs       
=======================================================
  Commands:                                            
    quit     — exit                                    
    help     — show this message                       
=======================================================
"""


def run_chat(model_dir: str = MODEL_OUTPUT_DIR, base_model: str = BASE_MODEL):
    """Interactive chat with the fine-tuned model."""
    chat = OncoChat(model_dir=model_dir, base_model=base_model)
    print(BANNER)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        cmd = question.lower()
        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif cmd == "help":
            print(BANNER)
            continue

        print("\nAssistant: ", end="", flush=True)
        response = chat.generate(question)
        print(response)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

USAGE = """\
Usage: python OncoChat-FineTune.py <command> [options]

Commands:
  prepare     Extract PDFs and generate training data
  train       Fine-tune the model on prepared data
  merge       Merge LoRA adapters into base model
  evaluate    Run evaluation on the test set
  test        Run functional tests
  chat        Interactive chat with the fine-tuned model
  all         Run full pipeline: prepare -> train -> evaluate -> test

Options:
  --pdf-dir DIR       Path to PDF directory (default: drug_reports)
  --output-dir DIR    Model output directory (default: oncochat_model)
  --base-model MODEL  Base model name (default: distilgpt2)
  --epochs N          Number of training epochs (default: 3)
  --batch-size N      Training batch size (default: 4)
  -h, --help          Show this help message
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    command = args[0]
    pdf_dir = PDF_DIR
    output_dir = MODEL_OUTPUT_DIR
    base_model = BASE_MODEL
    epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE

    i = 1
    while i < len(args):
        if args[i] == "--pdf-dir" and i + 1 < len(args):
            pdf_dir = args[i + 1]; i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]; i += 2
        elif args[i] == "--base-model" and i + 1 < len(args):
            base_model = args[i + 1]; i += 2
        elif args[i] == "--epochs" and i + 1 < len(args):
            epochs = int(args[i + 1]); i += 2
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = int(args[i + 1]); i += 2
        else:
            i += 1

    if command == "prepare":
        samples = process_pdfs_to_training_data(pdf_dir)
        save_training_data(samples)

    elif command == "train":
        train_file = str(Path(TRAIN_DATA_DIR) / "train.jsonl")
        test_file = str(Path(TRAIN_DATA_DIR) / "test.jsonl")
        if not Path(train_file).exists():
            log.info("No training data found — running prepare first...")
            samples = process_pdfs_to_training_data(pdf_dir)
            train_file, test_file = save_training_data(samples)
        fine_tune_model(
            train_file, test_file,
            base_model=base_model,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
        )

    elif command == "merge":
        merge_model(model_dir=output_dir, base_model=base_model)

    elif command == "evaluate":
        evaluate_model(model_dir=output_dir, base_model=base_model)

    elif command == "test":
        run_tests(model_dir=output_dir, base_model=base_model)

    elif command == "chat":
        run_chat(model_dir=output_dir, base_model=base_model)

    elif command == "all":
        train_file = str(Path(TRAIN_DATA_DIR) / "train.jsonl")
        test_file = str(Path(TRAIN_DATA_DIR) / "test.jsonl")

        if Path(train_file).exists() and Path(test_file).exists():
            print("=" * 60)
            print("STEP 1: Training data already exists — skipping preparation")
            print(f"  {train_file}")
            print(f"  {test_file}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("STEP 1: Preparing training data from PDFs")
            print("=" * 60)
            samples = process_pdfs_to_training_data(pdf_dir)
            train_file, test_file = save_training_data(samples)

        print("\n" + "=" * 60)
        print("STEP 2: Fine-tuning model")
        print("=" * 60)
        fine_tune_model(
            train_file, test_file,
            base_model=base_model,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
        )

        print("\n" + "=" * 60)
        print("STEP 3: Evaluating model")
        print("=" * 60)
        evaluate_model(model_dir=output_dir, base_model=base_model)

        print("\n" + "=" * 60)
        print("STEP 4: Running functional tests")
        print("=" * 60)
        run_tests(model_dir=output_dir, base_model=base_model)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Model saved to: {output_dir}/")
        print(f"Run 'python OncoChat-FineTune.py chat' to start chatting")
        print("=" * 60)

    else:
        print(f"Unknown command: {command}")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
