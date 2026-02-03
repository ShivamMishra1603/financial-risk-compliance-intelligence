import os
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/sections.jsonl"

# Regex for finding items
ITEM_1A_PATTERN = re.compile(r"Item\s+1A\.?\s+Risk\s+Factors", re.IGNORECASE)
ITEM_1B_PATTERN = re.compile(r"Item\s+1B\.?\s+Unresolved", re.IGNORECASE)
ITEM_7_PATTERN = re.compile(r"Item\s+7\.?\s+Management", re.IGNORECASE)
ITEM_7A_PATTERN = re.compile(r"Item\s+7A\.?\s+Quantitative", re.IGNORECASE)

def clean_text(text):
    """Normalize whitespace and remove junk."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_filing(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "lxml")
    text = soup.get_text(" ", strip=True) # Simple text extraction first
    
    # Ideally, we loop through text to find indices of headers.
    # This is a naive extraction for the resume project (production needs robust HTML parsing)
    
    # Find start/end based on regex strings in the extracted text
    matches_1a = list(ITEM_1A_PATTERN.finditer(text))
    matches_1b = list(ITEM_1B_PATTERN.finditer(text))
    matches_7 = list(ITEM_7_PATTERN.finditer(text))
    matches_7a = list(ITEM_7A_PATTERN.finditer(text))
    
    sections = []
    
    # Extract Risk Factors (Item 1A -> Item 1B)
    # We take the LAST match of 1A before the LAST match of 1B (usually Table of Contents is first)
    if matches_1a and matches_1b:
        start = matches_1a[-1].end()
        end = matches_1b[-1].start()
        if start < end:
            sections.append({
                "section": "Item 1A",
                "text": clean_text(text[start:end])
            })
            
    # Extract MD&A (Item 7 -> Item 7A)
    if matches_7 and matches_7a:
        start = matches_7[-1].end()
        end = matches_7a[-1].start()
        if start < end:
            sections.append({
                "section": "Item 7",
                "text": clean_text(text[start:end])
            })
            
    return sections

def process_all():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_records = []
    
    # Walk through the sec-edgar-downloader structure: /data/raw/sec-edgar-filings/{Ticker}/10-K/{Accession}/full-submission.txt
    
    total_files = 0
    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith(".txt") or file.endswith(".html"):
                total_files += 1
                
    print(f"Found {total_files} filings to process.")
    
    processed_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(RAW_DIR):
            for file in files:
                if file.endswith(".txt") or file.endswith(".html"):
                    path = os.path.join(root, file)
                    
                    # Ticker matches directory structure
                    parts = path.split(os.sep)
                    try:
                        ticker = parts[-4] # Adjust based on actual structure
                    except:
                        ticker = "UNKNOWN"
                        
                    try:
                        sections = parse_filing(path)
                        for sec in sections:
                            record = {
                                "ticker": ticker,
                                "section": sec["section"],
                                "text": sec["text"],
                                "source_path": path
                            }
                            out_f.write(json.dumps(record) + "\n")
                        processed_count += 1
                        if processed_count % 10 == 0:
                            print(f"Processed {processed_count} files...")
                    except Exception as e:
                        print(f"Error parsing {path}: {e}")

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all()
