import chromadb

client = chromadb.EphemeralClient()

print(client.list_collections())
client = chromadb.PersistentClient(path="/home/terry/code/learning/Agent/image_similarity_main/test/chroma_db/chroma_persistent_db")

print(client.list_collections())