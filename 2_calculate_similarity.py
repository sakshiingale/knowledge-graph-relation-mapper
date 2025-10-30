#!/usr/bin/env python3
"""
2_calculate_similarity.py

- Loads sections.csv and rules.csv
- Computes SBERT embeddings for each section and rule (model: all-MiniLM-L6-v2)
- Computes cosine similarity matrix
- Saves:
    - embeddings for sections and rules (.npz)
    - full pairs above threshold to CSV: section_rule_similarity.csv
    - full matrix to a feather/pickle (optional)
- Uses threshold 0.6 by default (matches your request).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ---- CONFIG ----
INPUT_DIR = Path("output_data")
SECTIONS_CSV = INPUT_DIR / "sections.csv"
RULES_CSV = INPUT_DIR / "rules.csv"

OUTPUT_DIR = INPUT_DIR
SECTIONS_EMB = OUTPUT_DIR / "sections_embeddings.npz"
RULES_EMB = OUTPUT_DIR / "rules_embeddings.npz"
SIMILARITY_CSV = OUTPUT_DIR / "section_rule_similarity.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.6  # threshold for saving edges (you requested 0.6)

def load_csv(path):
    df = pd.read_csv(path, encoding="utf8")
    return df

def compute_embeddings(model, texts, batch_size=64):
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def main():
    if not SECTIONS_CSV.exists() or not RULES_CSV.exists():
        raise FileNotFoundError("Run 1_extract_text.py first. Input CSVs not found in output_data/")

    print("Loading CSVs...")
    sections = load_csv(SECTIONS_CSV)
    rules = load_csv(RULES_CSV)
    print(f"Sections: {len(sections)}, Rules: {len(rules)}")

    # Prepare the texts (use title + first 1000 chars for stability)
    sections_texts = (sections["title"].fillna("") + ". " + sections["text"].fillna("")).tolist()
    rules_texts = (rules["title"].fillna("") + ". " + rules["text"].fillna("")).tolist()

    print("Loading SBERT model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Computing embeddings for sections...")
    sec_emb = compute_embeddings(model, sections_texts)
    print("Computing embeddings for rules...")
    rule_emb = compute_embeddings(model, rules_texts)

    # Save embeddings
    print("Saving embeddings...")
    np.savez_compressed(SECTIONS_EMB, embeddings=sec_emb)
    np.savez_compressed(RULES_EMB, embeddings=rule_emb)
    print("Embeddings saved to:", SECTIONS_EMB, RULES_EMB)

    # Compute cosine similarity matrix (since embeddings normalized, cosine = dot)
    print("Computing similarity matrix (this may take a moment)...")
    sim_matrix = np.matmul(sec_emb, rule_emb.T)

    # Find all pairs above threshold
    print(f"Filtering pairs with similarity >= {THRESHOLD} ...")
    rows = []
    n_sec, n_rule = sim_matrix.shape
    for i in range(n_sec):
        # get top matches per section (optional)
        for j in range(n_rule):
            score = float(sim_matrix[i, j])
            if score >= THRESHOLD:
                rows.append({
                    "section_index": i,
                    "section_id": sections.loc[i, "section_id"],
                    "section_title": sections.loc[i, "title"],
                    "rule_index": j,
                    "rule_id": rules.loc[j, "rule_id"],
                    "rule_title": rules.loc[j, "title"],
                    "similarity": round(score, 4),
                    "section_snippet": (sections.loc[i, "text"][:300].replace("\n", " ").strip()),
                    "rule_snippet": (rules.loc[j, "text"][:300].replace("\n", " ").strip())
                })

    sim_df = pd.DataFrame(rows)
    sim_df = sim_df.sort_values(["section_id", "similarity"], ascending=[True, False])
    sim_df.to_csv(SIMILARITY_CSV, index=False, encoding="utf8")
    print("Saved similarity (pairs >= threshold) to:", SIMILARITY_CSV)
    print(f"Total edges saved: {len(sim_df)}")

    # Additionally, save a full matrix (pickle) for fast future loads if desired
    FULL_MATRIX_PKL = OUTPUT_DIR / "similarity_matrix.npy"
    print("Saving full similarity matrix to:", FULL_MATRIX_PKL)
    np.save(FULL_MATRIX_PKL, sim_matrix)

    print("Done.")

if __name__ == "__main__":
    main()
