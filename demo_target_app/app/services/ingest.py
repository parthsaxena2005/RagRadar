from pathlib import Path
import os
import tiktoken
from typing import List

def load_data():
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'sebi_algo_2025.txt'
    return data_path.read_text(encoding='utf-8')


def process(data: str):
    lines = data.splitlines()
    refno = None
    datee = None
    subject = None
    actualdata = []
    for line in lines:
        line = line.lower()
        if refno is None and  'ref no' in line:
            _ , refno = line.split(' ')
        elif datee is None and 'date:' in line:
            _ , datee = line.split(' ')
        elif subject is None and 'subject:' in line:
            _ , subject = line.split(' ')
        else:
            actualdata.append(line)

    return refno , datee, subject , actualdata
   #print(lines)

def parse_regulatory_file(file_path: str):
    metadatadict = {"ref_no": None, "date": None, "subject": None}
    body_lines = []
    processing_clauses = False

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target document not found at path: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()
        # print(content)
    for line in content:
        cleaned_line = line.strip()
        lower_line = cleaned_line.lower()


        if not cleaned_line:
            continue # skip empty lines

        if cleaned_line.startswith('1.'):
            processing_clauses = True
        if processing_clauses:
            body_lines.append(cleaned_line)
        else:
            if 'ref no:' in lower_line or 'ref_no:' in lower_line:
                metadatadict["ref_no"] = cleaned_line.split(":", 1)[1].strip()
            elif "date:" in lower_line:
                metadatadict["date"] = cleaned_line.split(":", 1)[1].strip()
            
            elif "subject:" in lower_line:
                metadatadict["subject"] = cleaned_line.split(":", 1)[1].strip()
        
    clean_body_text = "\n".join(body_lines)
    return metadatadict, clean_body_text


def chunk_text_with_metadata(body_text : str,   metadatadict: dict, chunk_size  = 500, chunk_overlap = 15)-> list[dict]:
    chunks = []
    start_idx =0
    text_length = len(body_text)

    metadata_stamp = (
        f"[Source: {metadatadict['ref_no']} "
        f"Date: {metadatadict['date']}"
        f"Subject: {metadatadict['subject']}]\n"
    )
    while start_idx < text_length:

        end_idx = start_idx+chunk_size
        raw_chunk_slice = body_text[start_idx:end_idx]
        injected_chunk_text = metadata_stamp + raw_chunk_slice

        chunks.append({"content": injected_chunk_text, "metadata":metadata_stamp})

        start_idx += chunk_size-chunk_overlap

        if text_length   - start_idx <= chunk_overlap:
            break
    return chunks


def chunk_text_by_tokens(body_text : str, metadatadict: dict, chunk_size: int = 250, chunk_overlap: int = 15) -> List[dict]:
    encoding = tiktoken.get_encoding('cl100k_base')
    chunks_list = []
    body_text_token = encoding.encode(body_text) #list of numbers

    metadata_stamp = (
        f"[Company: {metadatadict['company']} "
        f"Quarter: {metadatadict['quarter']} "
        f"Section: {metadatadict['section']}]\n"
    )

    begin = 0

    while begin < len(body_text_token):
        chunk_embedding_decoded = encoding.decode(body_text_token[begin:begin+chunk_size])
        # text_encoding_list.append(bod)
        chunk_embedding_decoded = metadata_stamp + chunk_embedding_decoded
        
        chunks_list.append(
            {
                "content":chunk_embedding_decoded,
                "metadata": metadatadict
            }
            )
        begin += chunk_size-chunk_overlap

        if  begin >= len(body_text_token):
            break
    return chunks_list