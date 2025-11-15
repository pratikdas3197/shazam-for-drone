import numpy as np
from pymilvus import MilvusClient
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
DB_FILE = "milvus_drone_db.db"
COLLECTION_NAME = "drone_audio_db"
# ---------------------

def fetch_and_plot():
    """
    Connects to the Milvus DB, fetches all data, and plots it using t-SNE.
    """
    
    # 1. Connect to Milvus
    print(f"Connecting to Milvus DB: {DB_FILE}...")
    try:
        client = MilvusClient(uri=DB_FILE)
        if not client.has_collection(COLLECTION_NAME):
            print(f"Error: Collection '{COLLECTION_NAME}' not found in {DB_FILE}.")
            print("Please run the main indexing script first.")
            sys.exit(1)
        print("✓ Connected.")
        
        print(f"Loading collection '{COLLECTION_NAME}'...")
        client.load_collection(COLLECTION_NAME)
        print("✓ Collection loaded.")
    except Exception as e:
        print(f"Error connecting or loading collection: {e}")
        sys.exit(1)

    # 2. Fetch all data from the collection
    print("Fetching all data from collection...")
    try:
        # Get the total entity count first
        count_res = client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=["count(*)"]
        )
        entity_count = count_res[0]["count(*)"]
        if entity_count == 0:
            print("Error: No data found in the collection.")
            sys.exit(1)
            
        print(f"Found {entity_count} total entities. Fetching all...")

        # Use the count as the 'limit'
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=["drone_name", "embedding"],
            limit=entity_count
        )
        
        print(f"✓ Fetched {len(results)} entities.")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    # 3. Process data for plotting
    print("Processing data for t-SNE...")
    try:
        embeddings = np.array([item['embedding'] for item in results])
        labels = [item['drone_name'] for item in results]
    except KeyError:
        print("Error: Data is missing 'embedding' or 'drone_name' fields.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

    # 4. Run t-SNE
    print(f"Running t-SNE on {len(embeddings)} vectors... (this may take a minute)")
    tsne = TSNE(
        n_components=2,
        perplexity=15.0, 
        max_iter=1000,  # <-- FIX APPLIED HERE
        init='pca',
        learning_rate='auto',
        random_state=42
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    print("✓ t-SNE complete.")

    # 5. Plot the results
    print("Generating plot...")
    
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}

    fig, ax = plt.subplots(figsize=(16, 12))
    
    for label, color in color_map.items():
        indices = [i for i, l in enumerate(labels) if l == label]
        label_points = embeddings_2d[indices]
        
        ax.scatter(
            label_points[:, 0], 
            label_points[:, 1], 
            c=[color], 
            label=label, 
            alpha=0.8, 
            s=50 
        )

    ax.set_title('t-SNE Visualization of Drone Audio Embeddings', fontsize=18)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(title="Drone Models", loc="best", bbox_to_anchor=(1.05, 1), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    print("✓ Plot generated. Displaying now...")
    plt.show()

if __name__ == "__main__":
    fetch_and_plot()