#!/usr/bin/env python3
"""AWQ Quantization and Evaluation Script.

This script quantizes language models using AWQ techniques and evaluates
their perplexity on the WikiText dataset, including relative increases
compared to the base model.

Usage:
    python script.py --model-id /path/to/model
"""

import argparse
import logging
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
DEFAULT_MODEL_ID = "/home/rahul/llm-compressor/my_model"
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 512

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantize and evaluate language models using AWQ techniques"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Path to the model directory (default: %(default)s)",
    )
    return parser.parse_args()


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples: int = NUM_CALIBRATION_SAMPLES,
    max_seq_len: int = MAX_SEQUENCE_LENGTH,
    split: str = "train",
    text_column: str = "text",
) -> Dataset:
    """Prepare calibration dataset for quantization."""
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)
        dataset = dataset.shuffle(seed=42)
    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0], (list, tuple)) and isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Data must be a Hugging Face dataset name, list of texts, "
                "or list of tokenized sequences."
            )
    else:
        raise NotImplementedError(
            "Data must be a Hugging Face dataset name, list of texts, "
            "or list of tokenized sequences."
        )

    samples = []
    for n_run, data_item in enumerate(dataset):
        if n_run >= n_samples:
            break

        if isinstance(data_item, list):
            line_encoded = data_item
        else:
            line = data_item[text_column].strip()
            line_encoded = tokenizer.encode(line)

        if len(line_encoded) > max_seq_len or not line_encoded:
            continue

        sample = torch.tensor([line_encoded])
        samples.append(sample)

    if not samples:
        raise ValueError("No valid samples found for calibration")

    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logger.debug("Split into %d blocks", n_split)

    return Dataset.from_list([
        {"input_ids": cat_samples[:, i * max_seq_len: (i + 1) * max_seq_len].reshape(-1)}
        for i in range(n_split)
    ])


def run_llmc_awq(model_id: Path, tokenizer) -> AutoModelForCausalLM:
    """Quantize model using LLMC-AWQ."""
    output_dir = f"{model_id}-llmc-awq-{NUM_CALIBRATION_SAMPLES}"
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.transformers import oneshot

    recipe = [
        AWQModifier(bits=4, scheme="AWQ", targets="Linear", ignore=["lm_head"])
    ]

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )

    oneshot(
        model=model,
        dataset=get_calib_dataset(tokenizer=tokenizer),
        recipe=recipe,
        save_compressed=True,
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    logger.info("LLMC-AWQ model saved to %s", output_dir)
    return model


def run_auto_awq(model_id: Path) -> AutoModelForCausalLM:
    """Quantize model using Auto-AWQ."""
    output_dir = f"{model_id}-auto-awq-{NUM_CALIBRATION_SAMPLES}-quant-only"
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_dir)

    logger.info("Auto-AWQ model saved to %s", output_dir)
    return AutoAWQForCausalLM.from_pretrained(output_dir, device_map="cuda:0").model


def evaluate_perplexity(model: AutoModelForCausalLM, tokenizer, model_name: str) -> float:
    """Evaluate model perplexity on the WikiText-2 test set."""
    def _perplexity(nlls: List[torch.Tensor], n_samples: int, seqlen: int) -> torch.Tensor:
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data_ids = tokenized.input_ids.to(model.device)

    seqlen = 2048
    model.eval()
    n_samples = data_ids.numel() // seqlen
    nlls = []

    with tqdm(range(n_samples), desc=f"Evaluating {model_name} Perplexity") as progress_bar:
        for i in progress_bar:
            start_idx = i * seqlen
            end_idx = (i + 1) * seqlen
            batch = data_ids[:, start_idx:end_idx].to(model.device)

            with torch.no_grad():
                logits = model(batch).logits

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            nlls.append(loss.float() * seqlen)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"{model_name} Perplexity: {curr_ppl:.3f}")

    return _perplexity(nlls, n_samples, seqlen).item()


def main():
    """Main execution function."""
    args = parse_args()
    model_id = args.model_id
    logger.info("Using model ID: %s", model_id)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info("Starting model quantization and evaluation")

    # Evaluate base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    base_ppl = evaluate_perplexity(base_model, tokenizer, "Base Model")
    logger.info("Base Model Perplexity: %.3f", base_ppl)

    # Quantize and evaluate LLMC-AWQ model
    llmc_model = run_llmc_awq(model_id, tokenizer)
    llmc_ppl = evaluate_perplexity(llmc_model, tokenizer, "LLMC-AWQ")
    logger.info("LLMC-AWQ Perplexity: %.3f", llmc_ppl)

    # Quantize and evaluate Auto-AWQ model
    auto_model = run_auto_awq(model_id)
    auto_ppl = evaluate_perplexity(auto_model, tokenizer, "Auto-AWQ")
    logger.info("Auto-AWQ Perplexity: %.3f", auto_ppl)

    # Calculate relative increases
    llmc_relative = ((llmc_ppl - base_ppl) / base_ppl) * 100
    auto_relative = ((auto_ppl - base_ppl) / base_ppl) * 100

    # Log detailed results
    logger.info("\nDetailed Perplexity Results:")
    logger.info("-" * 40)
    logger.info(f"{'Model':<12}{'Perplexity':>12}{'Relative Increase':>20}")
    logger.info("-" * 40)
    logger.info(f"{'Base Model':<12}{base_ppl:>12.3f}{'':>20}")
    logger.info(f"{'LLMC-AWQ':<12}{llmc_ppl:>12.3f}{llmc_relative:>19.2f}%")
    logger.info(f"{'Auto-AWQ':<12}{auto_ppl:>12.3f}{auto_relative:>19.2f}%")
    logger.info("-" * 40)

    # Also print the summary table to the console
    summary = (
        "\nPerplexity Summary:\n"
        "-----------------------------\n"
        f"{'Model':<12}{'Perplexity':>12}{'Relative Increase':>20}\n"
        "-----------------------------\n"
        f"{'Base Model':<12}{base_ppl:>12.3f}{'':>20}\n"
        f"{'LLMC-AWQ':<12}{llmc_ppl:>12.3f}{llmc_relative:>19.2f}%\n"
        f"{'Auto-AWQ':<12}{auto_ppl:>12.3f}{auto_relative:>19.2f}%\n"
        "-----------------------------"
    )
    print(summary)


if __name__ == "__main__":
    main()
