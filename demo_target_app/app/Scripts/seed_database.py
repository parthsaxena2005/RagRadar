import os
from pathlib import Path

hf_cache_dir = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
if (hf_cache_dir / "models--nomic-ai--nomic-embed-text-v1.5").exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct



from chunk_transcripts import pipeline_process_and_chunk


def seed_vector_database():

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    COLLECTION_NAME="sec_transcripts"

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        
        client.create_collection(collection_name=COLLECTION_NAME,
                                vectors_config=(VectorParams(size=768 , distance=Distance.DOT))
                                )
        print("Collection created successfully.")
    else:
        print(f"collection {COLLECTION_NAME} already exists")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    all_processed_chunks = pipeline_process_and_chunk()
    totalchunks = len(all_processed_chunks)
    print(f"Extractesd {totalchunks} chunks for vector transformation")
    
    
    batch_size = 256
    
    for start in range(0, len(all_processed_chunks),batch_size):
        end = min(start+batch_size, totalchunks)

        batch_chunks  = all_processed_chunks[start:end]

        batch_ids = [start+local_idx for local_idx in range(len(batch_chunks))]

        existing_points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch_ids,
            with_vectors=False,
            with_payload=False
        )

        if(len(existing_points) == len(batch_chunks)):
            print("skipping points [{start}-{end}] : Already indexed in DB")
            continue

        batch_contents = [chunk["content"] for chunk in batch_chunks]
        batch_vectors = model.encode(batch_contents, convert_to_numpy=True)
    

        point_struct_list = []

        for local_idx , chunk in enumerate(batch_chunks):
            global_id = start+local_idx
            metadata = chunk.get("metadata",{})

            point = PointStruct(
                id = global_id,
                vector=batch_vectors[local_idx].tolist(),
                payload={
                    "content": chunk["content"],
                    "company": metadata.get("company", "UNKNOWN"),
                    "quarter": metadata.get("quarter", "UNKNOWN"),
                    "section": metadata.get("section", "UNKNOWN"),
                }
            )
            point_struct_list.append(point)

    
        # PointStructlist = [PointStruct(id=i+start, vector=x, 
        #                                payload={"content":y["content"],
        #                                         "company":y["metadata"]["company"],
        #                                         "quarter":y["metadata"]["quarter"],
        #                                         "section":y["metadata"]["section"],
        #                                          })
        #                      for i,(x,y) in 
        #                     enumerate(zip(vectorized_chunks[start:start+batch_size], all_processed_chunks[start:start+batch_size]),
        #                     start=start) ]
        operation_info = client.upsert(
            collection_name=COLLECTION_NAME, wait=False,
            points = point_struct_list,
        )
        # print(operation_info)
        print(f"Streamed points: [{end}/{totalchunks}] ({int((end/totalchunks)*100)}%) -> Status: {operation_info.status}")
    print("\nDistributed vector seeding completed successfully.")

if __name__ == "__main__":
    seed_vector_database()