#!/usr/bin/env python3
"""
1_extract_text.py

- Extracts sections (Act) and rules (Rules) from provided PDFs.
- Produces CSVs:
    - sections.csv  -> columns: section_id, title, text
    - rules.csv     -> columns: rule_id, title, text

Assumptions/heuristics:
- Both PDFs contain top-level numbered headings like:
    "1. Short title and commencement" or "3. STANDARDS FOR..." etc.
- We split text at lines that start with a top-level number followed by a dot and a space.
- This heuristic is robust for many legal PDFs but not perfect for extremely odd formats.
"""

import re
import csv
import os
import sys
from pathlib import Path
import pdfplumber
from tqdm import tqdm

# ---- CONFIG ----
ACT_PDF = r"C:\Users\Sakshi\Downloads\The Environment (Protection) Act, 1986.pdf"
RULES_PDF = r"C:\Users\Sakshi\Downloads\Environment (Protection) Rules, 1986.pdf"

OUTPUT_DIR = Path("output_data")
OUTPUT_DIR.mkdir(exist_ok=True)

SECTIONS_CSV = OUTPUT_DIR / "sections.csv"
RULES_CSV = OUTPUT_DIR / "rules.csv"

# regex to find top-level numbered headings:
# matches things like "\n1. Title" or start-of-file "1. Title"
HEADING_RE = re.compile(r'(?:\n|^)\s*(\d{1,3})\.\s+([A-Z0-9A-Za-z\-\(\),:\'\"\s]+?)(?=\n)', flags=re.MULTILINE)

def extract_text_from_pdf(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # get text preserving line breaks
            t = page.extract_text()
            if t:
                text.append(t + "\n")
    return "".join(text)

def split_into_numbered_blocks(text, heading_re=HEADING_RE):
    """
    Find heading matches; split text into list of (number, title, body_text)
    Using the heading regex, which finds numbered headings.
    If no headings found, fallback: return entire text as one block with id '0'.
    """
    matches = list(heading_re.finditer(text))
    blocks = []
    if not matches:
        return [("0", "FULL_TEXT", text.strip())]

    for i, m in enumerate(matches):
        num = m.group(1).strip()
        title = m.group(2).strip()
        start = m.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)
        body = text[start:end].strip()
        # combine title and body for clarity
        full_body = (title + "\n\n" + body).strip()
        blocks.append( (num, title, full_body) )
    return blocks

def write_csv(path, rows, headers):
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

def main():
    print("Extracting Act PDF...")
    act_text = extract_text_from_pdf(ACT_PDF)
    print("Splitting Act into sections...")
    act_sections = split_into_numbered_blocks(act_text)
    print(f"Found {len(act_sections)} sections (heuristic).")

    sections_rows = []
    for num, title, body in act_sections:
        sid = f"Section {num}"
        sections_rows.append( (sid, title.strip(), body.strip()) )

    print("Writing sections CSV:", SECTIONS_CSV)
    write_csv(SECTIONS_CSV, sections_rows, ["section_id", "title", "text"])

    print("\nExtracting Rules PDF...")
    rules_text = extract_text_from_pdf(RULES_PDF)
    print("Splitting Rules into rules...")
    rules_blocks = split_into_numbered_blocks(rules_text)
    print(f"Found {len(rules_blocks)} rules (heuristic).")

    rules_rows = []
    for num, title, body in rules_blocks:
        rid = f"Rule {num}"
        rules_rows.append( (rid, title.strip(), body.strip()) )

    print("Writing rules CSV:", RULES_CSV)
    write_csv(RULES_CSV, rules_rows, ["rule_id", "title", "text"])

    print("\nDone. Output files saved in:", OUTPUT_DIR.resolve())
    print("Check the CSVs and if you need different splitting heuristics tell me and I will adapt.")

if __name__ == "__main__":
    main()
