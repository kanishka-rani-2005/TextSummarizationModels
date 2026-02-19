import os
os.environ["HF_HOME"] = "E:/hf_cache"

import argparse
import pandas as pd
import numpy as np
import torch
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from .topsis import calculate_topsis


def evaluate_model(model_name, texts, references, device):
    print(f"\nEvaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.to(device)
    model.eval()

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rougeL_scores = []

    start_time = time.time()

    with torch.no_grad():
        for text, ref in zip(texts, references):
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            summary_ids = model.generate(**inputs, max_length=100)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            scores = scorer.score(ref, summary)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    end_time = time.time()

    avg_r1 = np.mean(rouge1_scores)
    avg_rL = np.mean(rougeL_scores)
    inference_time = (end_time - start_time) * 1000
    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)

    return avg_r1, avg_rL, inference_time, model_size


def main():
    parser = argparse.ArgumentParser(
        description="Summarization Model Comparator using TOPSIS"
    )

    parser.add_argument("--models", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="results.csv")

    args = parser.parse_args()

    models = args.models.split(',')

    df = pd.read_csv(args.dataset)

    if "text" not in df.columns or "reference" not in df.columns:
        raise ValueError("CSV must contain text and reference columns")

    texts = df["text"].tolist()
    references = df["reference"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []

    for model in models:
        r1, rL, t, size = evaluate_model(model, texts, references, device)
        results.append([model, r1, rL, t, size])

    result_df = pd.DataFrame(results, columns=[
        "Model", "ROUGE-1", "ROUGE-L", "Time_ms", "Size_MB"
    ])

    weights = [0.35, 0.35, 0.15, 0.15]
    impacts = ['+', '+', '-', '-']

    matrix = result_df[["ROUGE-1", "ROUGE-L", "Time_ms", "Size_MB"]].values

    result_df["TOPSIS_Score"] = calculate_topsis(matrix, weights, impacts)
    result_df["Rank"] = result_df["TOPSIS_Score"].rank(ascending=False)

    result_df = result_df.sort_values("Rank")

    result_df.to_csv(args.output, index=False)

    print("\nFinal Ranking:\n")
    print(result_df)


if __name__ == "__main__":
    main()
