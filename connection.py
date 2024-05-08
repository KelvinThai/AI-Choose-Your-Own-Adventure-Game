import os
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.environ.get("ASTRA_DB_KEYSPACE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASTRA_DB_ID = "06ee2aec-4814-4c1a-86b9-46c9ddcefffd"

# Specify the embeddings model, database, and collection to use. If the collection does not exist, it is created automatically.
embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    embedding=embedding,
    namespace=ASTRA_DB_KEYSPACE,
    collection_name="test",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

auth_provider = PlainTextAuthProvider(ASTRA_DB_API_ENDPOINT,ASTRA_DB_APPLICATION_TOKEN )
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()


message_history = CassandraChatMessageHistory(
    session_id="anything",
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    ttl_seconds=3600
)

message_history.clear()

cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)
#Load a small dataset of philosophical quotes with the Python dataset module.

philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
print("An example entry:")
print(philo_dataset[16])

#Process metadata and convert to LangChain documents.

docs = []
for entry in philo_dataset:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        # Add metadata tags to the metadata dictionary
        for tag in entry["tags"].split(";"):
            metadata[tag] = "y"
    # Add a LangChain document with the quote and metadata tags
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)

#Compute embeddings for each document and store in the database.
inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")

#Show quotes that are similar to a specific quote.
results = vstore.similarity_search("Our life is what we make of it", k=3)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")