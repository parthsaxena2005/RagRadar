import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo_target_app.app.services.ingest import chunk_text_by_tokens
from demo_target_app.app.services.manual_retriever import top_k_simmilar_rankbm25
from demo_target_app.app.services.manual_retriever import retrieve_hybrid
from demo_target_app.app.services.manual_retriever import retrieve_dense
from demo_target_app.app.Scripts.chunk_transcripts import clean_sec_text


mock_metadata = {"company": "TSLA", "quarter": "Q3_2024", "section": "CAPEX"}
mock_text = """
Tesla's capital expenditures for gigafactory expansion hit record highs. 
We spent 2.5 billion dollars on manufacturing capacity, specifically targeting 
the Austin Texas and Berlin facilities for tooling and production lines.
"""

chunks = chunk_text_by_tokens(mock_text, mock_metadata, chunk_size=50, chunk_overlap=5)

query_string = "How much dollars did Tesla spend on Gigafactory expansion?"

# results = top_k_simmilar_rankbm25(query_string,1, chunks)
results = retrieve_hybrid(query_string,1, chunks)
print("Test query: ", query_string)

if results:
    print("results highest match: ")
    print(results[0]["content"])
else:
    print("no results retrieved")
