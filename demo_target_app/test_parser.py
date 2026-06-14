from app.services.ingest import parse_regulatory_file
from app.services.ingest import chunk_text_with_metadata
from app.services.ingest import chunk_text_by_tokens
meta, body = parse_regulatory_file(r"E:\Project\ragradar\demo_target_app\app\data\sebi_algo_2025.txt")
print("--- EXTRACTED METADATA ---")
# print(meta)
print("\n--- CLEANED BODY TEXT ---")
# print(body[:300] + "...") # Show just the start of the body

# chunks = chunk_text_with_metadata(body, meta)
# print("chunks extracted: ", len(chunks))
# print(chunks[:2])

chunks = chunk_text_by_tokens(body, meta)
print("chunks extracted: ", len(chunks))
print(chunks[:2])