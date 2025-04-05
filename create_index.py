import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from datetime import datetime

# Load API Key
load_dotenv()

# Check if OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OpenAI API key not found in .env file.")
    exit(1)

# Use OpenAI embeddings
Settings.embed_model = OpenAIEmbedding()

# Load Winterville events data
try:
    data = pd.read_excel("winterville_events.xlsx", sheet_name="Sheet1")
except FileNotFoundError:
    print("❌ Error: 'winterville_events.xlsx' file not found.")
    exit(1)
except Exception as e:
    print(f"❌ Error reading Excel file: {e}")
    exit(1)

# Create documents
documents = []
for _, row in data.iterrows():
    text = f"""Event: {row['Event']}
Time: {row['Time']}
Location: {row['Location']}
Price: {row['Price']}"""
    metadata = {
        "Category": row["Category"],
        "Date": row["Date"].strftime("%Y-%m-%d") if not isinstance(row['Date'], str) else row['Date'],
        "Price": row["Price"],
        "Location": row["Location"],
    }
    documents.append(Document(text=text, metadata=metadata))

# Create and persist the index
os.makedirs("./winterville_events_index", exist_ok=True)
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./winterville_events_index")

print("✅ Index successfully created and saved!")
