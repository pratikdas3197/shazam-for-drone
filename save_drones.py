"""
Advanced Audio Matching for Similar Drone Sounds
Multiple techniques to handle noise and fluctuations

MODIFIED: Changed HNSW index to AUTOINDEX for Milvus Lite compatibility.
"""

import numpy as np
import librosa
from panns_inference import AudioTagging
import torch
import os
from typing import List, Tuple, Dict
import glob
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================
# NEW IMPORTS FOR MILVUS
# ============================================
from pymilvus import MilvusClient, DataType

# (All existing classes: AudioPreprocessor, MultiScaleEmbedder, 
#  AudioFeatureExtractor, AdvancedMatcher, EmbeddingOptimizer, 
#  AudioFingerprinter ... remain unchanged from the previous code)
# ...

# ============================================
# METHOD 1: Audio Preprocessing
# ============================================
class AudioPreprocessor:
    """Clean and normalize audio before embedding"""
    
    @staticmethod
    def denoise_audio(audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove background noise using spectral gating"""
        # Compute spectrogram
        D = librosa.stft(audio)
        magnitude, phase = librosa.magphase(D)
        
        # Estimate noise from first 0.5 seconds
        noise_profile = np.median(magnitude[:, :int(0.5 * sr / 512)], axis=1, keepdims=True)
        
        # Spectral gating
        mask = magnitude > (noise_profile * 1.5)
        magnitude_cleaned = magnitude * mask
        
        # Reconstruct audio
        D_cleaned = magnitude_cleaned * phase
        audio_cleaned = librosa.istft(D_cleaned)
        
        return audio_cleaned
    
    @staticmethod
    def normalize_loudness(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to consistent loudness"""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.1
            audio = audio * (target_rms / rms)
        return audio
    
    @staticmethod
    def remove_silence(audio: np.ndarray, sr: int, threshold_db: int = 30) -> np.ndarray:
        """Trim silence from beginning and end"""
        intervals = librosa.effects.split(audio, top_db=threshold_db)
        if len(intervals) > 0:
            start = intervals[0][0]
            end = intervals[-1][1]
            audio = audio[start:end]
        return audio
    
    @staticmethod
    def apply_bandpass_filter(audio: np.ndarray, sr: int, 
                             lowcut: float = 100, highcut: float = 8000) -> np.ndarray:
        """Apply bandpass filter to focus on drone frequency range"""
        nyquist = sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    @staticmethod
    def preprocess_full(audio_path: str) -> Tuple[np.ndarray, int]:
        """Complete preprocessing pipeline"""
        # Load
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        
        # Apply all preprocessing
        audio = AudioPreprocessor.remove_silence(audio, sr)
        audio = AudioPreprocessor.apply_bandpass_filter(audio, sr, lowcut=150, highcut=6000)
        audio = AudioPreprocessor.denoise_audio(audio, sr)
        audio = AudioPreprocessor.normalize_loudness(audio)
        
        return audio, sr


# ============================================
# METHOD 2: Multi-Scale Embeddings
# ============================================
class MultiScaleEmbedder:
    """Generate embeddings at multiple time scales"""
    
    def __init__(self):
        self.model = AudioTagging(
            checkpoint_path=None,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def embed_with_segments(self, audio: np.ndarray, segment_duration: float = 2.0,
                           sr: int = 32000) -> np.ndarray:
        """
        Split audio into segments and average embeddings
        More robust to temporal variations
        """
        segment_samples = int(segment_duration * sr)
        embeddings_list = []
        
        # Process overlapping segments (50% overlap)
        hop_samples = segment_samples // 2
        
        for start in range(0, len(audio) - segment_samples, hop_samples):
            segment = audio[start:start + segment_samples]
            segment = segment[np.newaxis, :]
            
            _, embedding = self.model.inference(segment)
            embeddings_list.append(embedding[0])
        
        if len(embeddings_list) == 0:
            # Audio too short, use full audio
            audio_batch = audio[np.newaxis, :]
            _, embedding = self.model.inference(audio_batch)
            return embedding[0]
        
        # Aggregate: use both mean and std for richer representation
        embeddings_array = np.array(embeddings_list)
        mean_embedding = np.mean(embeddings_array, axis=0)
        std_embedding = np.std(embeddings_array, axis=0)
        
        # Concatenate mean and std
        combined = np.concatenate([mean_embedding, std_embedding * 0.5])
        return combined
    
    def embed_multiscale(self, audio: np.ndarray, sr: int = 32000) -> np.ndarray:
        """Generate embeddings at multiple scales and combine"""
        scales = [1.0, 2.0, 4.0]  # seconds
        embeddings = []
        
        for scale in scales:
            emb = self.embed_with_segments(audio, segment_duration=scale, sr=sr)
            embeddings.append(emb)
        
        # Concatenate all scales
        return np.concatenate(embeddings)


# ============================================
# METHOD 3: Feature Engineering
# ============================================
class AudioFeatureExtractor:
    """Extract robust acoustic features"""
    
    @staticmethod
    def extract_spectral_features(audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral features invariant to noise"""
        features = []
        
        # Mel-frequency cepstral coefficients (robust to noise)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features.append(np.mean(mfccs, axis=1))
        features.append(np.std(mfccs, axis=1))
        
        # Delta MFCCs (temporal dynamics)
        delta_mfccs = librosa.feature.delta(mfccs)
        features.append(np.mean(delta_mfccs, axis=1))
        
        # Spectral contrast (robust to amplitude variations)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.append(np.mean(contrast, axis=1))
        
        # Chroma features (pitch content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(np.mean(chroma, axis=1))
        
        # Zero crossing rate (temporal texture)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append([np.mean(zcr), np.std(zcr)])
        
        return np.concatenate(features)
    
    @staticmethod
    def extract_temporal_features(audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract temporal pattern features"""
        # Onset strength (rhythmic pattern)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_features = [
            np.mean(onset_env),
            np.std(onset_env),
            np.max(onset_env)
        ]
        
        # Tempogram (periodicity)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tempo_features = [
            np.mean(tempogram),
            np.std(tempogram)
        ]
        
        return np.array(onset_features + tempo_features)


# ============================================
# METHOD 4: Advanced Similarity Metrics
# ============================================
class AdvancedMatcher:
    """Multiple similarity metrics for robust matching"""
    
    @staticmethod
    def weighted_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray, 
                                  weights: np.ndarray = None) -> float:
        """Cosine similarity with feature weighting"""
        if weights is None:
            weights = np.ones_like(emb1)
        
        emb1_weighted = emb1 * weights
        emb2_weighted = emb2 * weights
        
        norm1 = np.linalg.norm(emb1_weighted)
        norm2 = np.linalg.norm(emb1_weighted)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1_weighted, emb2_weighted) / (norm1 * norm2)
    
    @staticmethod
    def pearson_correlation(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Pearson correlation (invariant to linear transformations)"""
        return np.corrcoef(emb1, emb2)[0, 1]
    
    @staticmethod
    def normalized_cross_correlation(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Normalized cross-correlation"""
        emb1_normalized = (emb1 - np.mean(emb1)) / (np.std(emb1) + 1e-8)
        emb2_normalized = (emb2 - np.mean(emb2)) / (np.std(emb2) + 1e-8)
        
        return np.mean(emb1_normalized * emb2_normalized)
    
    @staticmethod
    def manhattan_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """L1 distance (more robust to outliers than L2)"""
        return np.sum(np.abs(emb1 - emb2))
    
    @staticmethod
    def earth_movers_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Wasserstein distance approximation"""
        # Normalize to probability distributions
        emb1_pos = np.abs(emb1)
        emb2_pos = np.abs(emb2)
        
        emb1_norm = emb1_pos / (np.sum(emb1_pos) + 1e-8)
        emb2_norm = emb2_pos / (np.sum(emb2_pos) + 1e-8)
        
        # Cumulative distributions
        cdf1 = np.cumsum(emb1_norm)
        cdf2 = np.cumsum(emb2_norm)
        
        return np.sum(np.abs(cdf1 - cdf2))
    
    @staticmethod
    def ensemble_similarity(emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """Combine multiple similarity metrics"""
        metrics = {
            'cosine': AdvancedMatcher.weighted_cosine_similarity(emb1, emb2),
            'pearson': AdvancedMatcher.pearson_correlation(emb1, emb2),
            'ncc': AdvancedMatcher.normalized_cross_correlation(emb1, emb2),
            'manhattan': AdvancedMatcher.manhattan_distance(emb1, emb2),
            'emd': AdvancedMatcher.earth_movers_distance(emb1, emb2)
        }
        
        # Ensemble score (weighted combination)
        # High values are better for cosine, pearson, ncc
        # Low values are better for manhattan, emd
        
        # Normalize to 0-1 range
        similarity_score = (
            metrics['cosine'] * 0.35 +
            metrics['pearson'] * 0.25 +
            metrics['ncc'] * 0.20 -
            min(metrics['manhattan'] / 100, 1) * 0.10 -
            min(metrics['emd'] / 2, 1) * 0.10
        )
        
        metrics['ensemble'] = max(0, min(1, similarity_score))
        return metrics


# ============================================
# METHOD 5: Dimensionality Reduction & Clustering
# ============================================
class EmbeddingOptimizer:
    """Reduce noise and improve discriminability"""
    
    def __init__(self, n_components: int = 512):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA and transform embeddings"""
        # Standardize
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # PCA
        embeddings_reduced = self.pca.fit_transform(embeddings_scaled)
        self.is_fitted = True
        
        return embeddings_reduced
    
    def transform(self, embedding: np.ndarray) -> np.ndarray:
        """Transform new embedding"""
        if not self.is_fitted:
            return embedding
        
        embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
        embedding_reduced = self.pca.transform(embedding_scaled)
        return embedding_reduced[0]


# ============================================
# METHOD 6: Audio Fingerprinting
# ============================================
class AudioFingerprinter:
    """Create robust audio fingerprints"""
    
    @staticmethod
    def create_fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Create chromaprint-like fingerprint
        Robust to noise and pitch variations
        """
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
        
        # Binarize based on local maxima
        fingerprint = np.zeros_like(chroma, dtype=int)
        
        for i in range(chroma.shape[1]):
            if i > 0 and i < chroma.shape[1] - 1:
                # Find local peaks
                for j in range(chroma.shape[0]):
                    if chroma[j, i] > chroma[j, i-1] and chroma[j, i] > chroma[j, i+1]:
                        fingerprint[j, i] = 1
        
        # Flatten to 1D fingerprint
        return fingerprint.flatten()
    
    @staticmethod
    def hamming_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Hamming distance between fingerprints"""
        min_len = min(len(fp1), len(fp2))
        return np.sum(fp1[:min_len] != fp2[:min_len]) / min_len


# ============================================
# INTEGRATED PIPELINE
# ============================================
class RobustAudioMatcher:
    """Complete robust matching pipeline"""
    
    def __init__(self, use_preprocessing: bool = True):
        self.use_preprocessing = use_preprocessing
        self.embedder = MultiScaleEmbedder()
        self.feature_extractor = AudioFeatureExtractor()
    
    def process_audio(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract all features from audio"""
        print(f"  Processing: {os.path.basename(audio_path)}")
        
        # Load and preprocess
        if self.use_preprocessing:
            audio, sr = AudioPreprocessor.preprocess_full(audio_path)
        else:
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        
        features = {}
        
        # Deep embeddings (PANNs multi-scale)
        features['deep_embedding'] = self.embedder.embed_multiscale(audio, sr)
        
        # Acoustic features
        features['spectral'] = self.feature_extractor.extract_spectral_features(audio, sr)
        features['temporal'] = self.feature_extractor.extract_temporal_features(audio, sr)
        
        # Fingerprint
        features['fingerprint'] = AudioFingerprinter.create_fingerprint(audio, sr)
        
        # Combined feature vector
        features['combined'] = np.concatenate([
            features['deep_embedding'],
            features['spectral'],
            features['temporal']
        ])
        print(f"    ✓ Combined vector dim: {features['combined'].shape}")
        
        return features


# ============================================
# MILVUS VECTOR DATABASE MANAGER (FIXED)
# ============================================
class MilvusManager:
    """
    Handles connection and operations for Milvus DB using the new MilvusClient.
    This will automatically use MilvusLite by connecting to a local file.
    """
    
    def __init__(self, db_file: str, collection_name: str):
        self.collection_name = collection_name
        self.db_file = db_file
        print(f"Initializing Milvus Lite with database file: {self.db_file}")
        self.client = MilvusClient(uri=self.db_file)
        print("✓ MilvusClient initialized.")

    def setup_collection(self, vector_dim: int):
        """Create the collection and index if they don't exist"""
        if self.client.has_collection(self.collection_name):
            print(f"Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)

        # Define schema
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="drone_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
        
        # === FIX 1: Use AUTOINDEX instead of HNSW ===
        index_params = self.client.prepare_index_params(
            field_name="embedding",
            metric_type="IP", # Inner Product (Cosine Similarity for normalized vectors)
            index_type="AUTOINDEX" # Milvus Lite supports AUTOINDEX, FLAT, IVF_FLAT
            # No `params` needed for AUTOINDEX
        )
        
        # Create collection
        print(f"Creating collection: {self.collection_name}")
        self.client.create_collection(
            self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print("✓ Collection and Index created (AUTOINDEX, IP).")

    def insert_data(self, data: List[Dict]):
        """Insert a batch of data into the collection"""
        try:
            print(f"Inserting batch of {len(data)} entities...")
            mr = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            self.flush() # Ensure data is saved
            print(f"✓ Inserted {mr['insert_count']} entities.")
            return mr
        except Exception as e:
            print(f"✗ Error inserting data: {e}")
            return None

    def load_collection(self):
        """Load the collection into memory for searching"""
        print("Loading collection into memory...")
        self.client.load_collection(self.collection_name)
        print("✓ Collection loaded.")

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        """Search the collection for the closest vectors"""
        
        # === FIX 2: Remove HNSW-specific search params ===
        search_params = {
            "metric_type": "IP"
            # No "params" key needed, as we're not using HNSW
        }
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            search_params=search_params,
            limit=top_k,
            output_fields=["file_path", "drone_name"]
        )
        return results[0]

    def flush(self):
        """Flush inserted data to disk"""
        self.client.flush(self.collection_name)
        
    def drop(self):
        """Drop the collection"""
        print(f"Dropping collection: {self.collection_name}")
        self.client.drop_collection(self.collection_name)
        print("✓ Collection dropped.")
        
    def count_entities(self) -> int:
        """Count entities in the collection"""
        res = self.client.query(self.collection_name, filter="", output_fields=["count(*)"])
        return res[0]["count(*)"]


# ============================================
# HELPER FUNCTIONS
# ============================================

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector for IP (Cosine) search in Milvus"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return (v / norm).astype(np.float32)

def index_train_audio(matcher: RobustAudioMatcher, db: MilvusManager, train_path: str):
    """
    TASK 1: Find, process, and index all training audio into Milvus.
    """
    print("\n" + "="*70)
    print("STARTING: TASK 1 - INDEXING TRAINING AUDIO")
    print("="*70)
    
    train_files = glob.glob(os.path.join(train_path, "drone_*", "*.wav"))
    if not train_files:
        print(f"✗ No training files found at: {train_path}")
        return
        
    print(f"Found {len(train_files)} training files.")
    
    all_data = []
    for i, file_path in enumerate(train_files):
        print(f"\n--- Processing file {i+1}/{len(train_files)} ---")
        try:
            features = matcher.process_audio(file_path)
            embedding_normalized = normalize_vector(features['combined'])
            drone_name = os.path.basename(os.path.dirname(file_path))
            
            all_data.append({
                "file_path": file_path,
                "drone_name": drone_name,
                "embedding": embedding_normalized
            })
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")

    if all_data:
        db.insert_data(all_data)
        print(f"Total entities in collection: {db.count_entities()}")
    
    db.load_collection()

def query_test_audio(matcher: RobustAudioMatcher, db: MilvusManager, test_path: str, top_k: int = 5):
    """
    TASK 2: Find, process, and query all test audio against Milvus.
    """
    print("\n" + "="*70)
    print("STARTING: TASK 2 - QUERYING TEST AUDIO")
    print("="*70)
    
    test_files = glob.glob(os.path.join(test_path, "*.wav"))
    if not test_files:
        print(f"✗ No test files found at: {test_path}")
        return
        
    print(f"Found {len(test_files)} test files to query.")
    
    for file_path in test_files:
        print("\n" + "-"*70)
        print(f"QUERYING FOR: {os.path.basename(file_path)}")
        print("-"*(15 + len(os.path.basename(file_path))))
        
        try:
            features = matcher.process_audio(file_path)
            embedding_normalized = normalize_vector(features['combined'])
            
            results = db.search(embedding_normalized, top_k=top_k)
            
            print("\nClosest matches from training data:")
            if not results:
                print("  No matches found.")
                continue
                
            for rank, hit in enumerate(results, 1):
                print(f"  {rank}. File:    {hit['entity']['file_path']}")
                print(f"     Drone:   {hit['entity']['drone_name']}")
                print(f"     Score:   {hit['distance']:.4f} (Cosine Similarity)")
                print("     ---")
        
        except Exception as e:
            print(f"✗ Error querying {file_path}: {e}")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    
    # --- Configuration ---
    BASE_AUDIO_PATH = "drone_audio"
    TRAIN_PATH = os.path.join(BASE_AUDIO_PATH, "train")
    TEST_PATH = os.path.join(BASE_AUDIO_PATH, "test")
    
    DB_FILE = "milvus_drone_db.db"
    COLLECTION_NAME = "drone_audio_db"
    
    VECTOR_DIMENSION = 12374 
    TOP_K_RESULTS = 5
    
    db_manager = None
    try:
        # 1. Setup Milvus
        db_manager = MilvusManager(db_file=DB_FILE, collection_name=COLLECTION_NAME)
        
        # 2. Setup the collection
        db_manager.setup_collection(vector_dim=VECTOR_DIMENSION)
        
        # 3. Init audio processor
        matcher = RobustAudioMatcher(use_preprocessing=True)
        
        # 4. (Task 1) Index all training audio
        index_train_audio(matcher, db_manager, TRAIN_PATH)
        
        # 5. (Task 2) Query with all test audio
        query_test_audio(matcher, db_manager, TEST_PATH, top_k=TOP_K_RESULTS)
        
        print("\n" + "="*70)
        print("✓ FULL PIPELINE COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\nAn error occurred during the main execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if db_manager:
            # Uncomment the next line to delete the collection after running
            # db_manager.drop()
            print(f"\nCleanup: Collection '{COLLECTION_NAME}' retained in '{DB_FILE}'.")