import numpy as np
from pymilvus import MilvusClient
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

# --- Configuration ---
DB_FILE = "milvus_drone_db.db"
COLLECTION_NAME = "drone_audio_db"
# ---------------------

def fetch_and_plot_3d_averaged():
    """
    Connects to the Milvus DB, fetches all data, plots the *average*
    embedding for each drone in a 3D t-SNE.
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

    # 3. Process data: Group and average embeddings
    print("Grouping and averaging embeddings by drone name...")
    
    # Use defaultdict to automatically create lists for new keys
    grouped_embeddings = defaultdict(list)
    
    try:
        # Group all embeddings by their drone_name
        for item in results:
            grouped_embeddings[item['drone_name']].append(item['embedding'])
            
        averaged_embeddings_list = []
        final_labels_list = []
        
        # Calculate the average vector for each group
        for label, embeddings_list in grouped_embeddings.items():
            embeddings_array = np.array(embeddings_list)
            # Calculate mean along axis 0 (the mean of all vectors)
            mean_vector = np.mean(embeddings_array, axis=0)
            
            averaged_embeddings_list.append(mean_vector)
            final_labels_list.append(label)
            
        # This is now our new dataset for t-SNE
        embeddings = np.array(averaged_embeddings_list)
        labels = final_labels_list
        
        print(f"✓ Averaged data into {len(labels)} unique drone vectors.")
        
    except KeyError:
        print("Error: Data is missing 'embedding' or 'drone_name' fields.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

    # 4. Run t-SNE
    print(f"Running t-SNE for 3 dimensions on {len(embeddings)} vectors...")
    
    # --- IMPORTANT ---
    # Perplexity MUST be less than the number of samples (k).
    # Since k is small (e.g., 10-12), we must use a small value.
    k_samples = len(embeddings)
    perplexity_value = max(1.0, min(5.0, k_samples - 2.0)) # A safe value
    print(f"Using perplexity: {perplexity_value} (k={k_samples})")
    
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity_value,
        max_iter=1000, 
        init='pca',
        learning_rate='auto',
        random_state=42
    )
    embeddings_3d = tsne.fit_transform(embeddings)
    print("✓ t-SNE complete.")

    # 5. Plot the results
    print("Generating 3D plot...")
    
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(projection='3d')

    # This loop now plots one point per label
    for label, color in color_map.items():
        # Find the index for this label (will be just one)
        indices = [i for i, l in enumerate(labels) if l == label]
        label_points = embeddings_3d[indices]
        
        ax.scatter(
            label_points[:, 0], # X axis
            label_points[:, 1], # Y axis
            label_points[:, 2], # Z axis
            c=[color], 
            label=label, 
            alpha=1.0, 
            s=150  # Make points larger
        )

    ax.set_title('3D t-SNE Visualization of *Averaged* Drone Embeddings', fontsize=18)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_zlabel('t-SNE Component 3', fontsize=12) 
    
    ax.legend(title="Drone Models", loc="best", bbox_to_anchor=(1.1, 1), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    print("✓ Plot generated. Displaying now... (You can click and drag to rotate)")
    plt.show()

if __name__ == "__main__":
    fetch_and_plot_3d_averaged()