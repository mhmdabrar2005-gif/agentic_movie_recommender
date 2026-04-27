import chromadb

def check_db():
    # Connect to the local Chroma database
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = client.get_collection(name="movies")
    except Exception as e:
        print("Collection 'movies' not found in ChromaDB.")
        return
        
    # Get the total count
    count = collection.count()
    print(f"Total items in ChromaDB: {count}\n")
    
    # Fetch all items to analyze metadata
    results = collection.get()
    metadatas = results.get("metadatas", [])
    
    movies_count = 0
    tv_count = 0
    
    print("--- TV Shows found in DB ---")
    for meta in metadatas:
        # Check the 'type' metadata we added in ingest.py
        item_type = meta.get("type", "movie")
        
        if item_type == "tv":
            tv_count += 1
            print(f"[TV] {meta.get('title')}")
        elif item_type == "movie":
            movies_count += 1
            
    print("\n--- Breakdown ---")
    print(f"Movies: {movies_count}")
    print(f"TV Shows: {tv_count}")

if __name__ == "__main__":
    check_db()
