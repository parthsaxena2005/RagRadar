import sys
import re
from pathlib import Path
from typing import List


current_filepath = Path(__file__).resolve()

project_root = current_filepath.parents[3] 
if str(project_root) not in sys.path:
    sys.path.insert(0,str(project_root))

from demo_target_app.app.services.ingest import chunk_text_by_tokens

def clean_sec_text(raw_text : str) ->str:
    clean_text = re.sub(r'<[^>]+>', ' ', raw_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    print(clean_text.strip())
    return clean_sec_text.strip()

# with open(r'E:\Project\ragradar\demo_target_app\app\data\sec-edgar-filings\AMZN\8-K\0001018724-26-000012\full-submission.txt', 'r') as f:
#     data = clean_sec_text(f.readlines)
def pipeline_process_and_chunk()->List[dict]:
    print("Starting Chunking")
    script_dir = Path(__file__).resolve().parent
    base_data_path = script_dir.parent / "Data" / "sec-edgar-filings"

    all_processed_chunks = []

    for ticker_path in base_data_path.iterdir():
        if not ticker_path.is_dir():
            continue

        ticker = ticker_path.name
        form_path = ticker_path / "8-K"

        if not form_path.exists():
            continue

        print(f"Slicing chunks for Ticker: {ticker}")

        for submission_dir in form_path.iterdir():
            file_path = submission_dir / "full-submission.txt"
            if not file_path.exists():
                continue

            with open(file_path, 'r', encoding='utf-8', errors="ignore") as f:
                raw_content = f.read()
            
            metadata = {
                'company':ticker,
                'quarter':"Q3_2024",
                "section": "Current_Report"
            }

            chunks= chunk_text_by_tokens(raw_content,metadata,chunk_overlap=64,chunk_size=512)
            all_processed_chunks.extend(chunks)
    print(f"Completed \nTotal Corporate Token Chunks Processed: {len(all_processed_chunks)}")    
    if all_processed_chunks:
        print(all_processed_chunks[0]["content"][:300]+"...")
    return all_processed_chunks

if __name__ =="__main__":
    pipeline_process_and_chunk()