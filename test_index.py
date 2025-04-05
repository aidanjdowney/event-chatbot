from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode, MetadataFilters, MetadataFilter
from llama_index.core.response import Response
from dotenv import load_dotenv

load_dotenv()

Settings.llm = OpenAI(model="gpt-4")

storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# Create metadata filters
filters = MetadataFilters(filters=[
    MetadataFilter(key="Category", value="Music"),
    MetadataFilter(key="Date", value="2025-03-15"),
])

# Create a query bundle with the filters
query_bundle = QueryBundle(
    query_str="What music events are happening on 3/15/25?",
    custom_embedding_strs=["What music events are happening on 3/15/25?"],
    metadata_filters=filters,
)

nodes = query_engine.retrieve(query_bundle) #Use the query bundle here.

filtered_nodes = []
for node in nodes:
    print(f"Node metadata: {node.metadata}") #Print the metadata
    if node.metadata.get("Category") == "Music" and node.metadata.get("Date") == "2025-03-15":
        filtered_nodes.append(node)

response_text = "\n".join([node.get_content() for node in filtered_nodes])

response = Response(response=response_text)

print(response)
