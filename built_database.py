
import pandas as pd
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

print("Loading biology dataset...")
df = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
texts = df['passage'].tolist()

print(f"Loaded {len(texts)} passages. Using lighter model...")


embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")


print("Processing in small batches (to prevent overheating)...")
docs = [Document(page_content=text) for text in texts]

batch_size = 500  # Smaller batches

for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1}/{(len(docs)//batch_size)+1}...")
    
    if i == 0:
        
        vector_db = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory="./bio_db"
        )
    else:
        
        vector_db.add_documents(batch)
    
    
    time.sleep(10) 
    print("Batch completed. Cooling down...")

vector_db.persist()
print("✅ Database built and saved to './bio_db' folder!")

print(f"✅ Added {len(docs)} biology passages to the database!")
