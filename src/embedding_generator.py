"""
Embedding generation module.

This module handles generating embeddings for bug reports and source code files
using pre-trained models (CodeBERT, CodeT5), and caching them to avoid recomputation.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from codebert_model import CodeBERTModel


class EmbeddingGenerator:
    """
    A class to generate and cache embeddings for bug reports and code files.
    
    Attributes:
        model: The embedding model (CodeBERT or CodeT5)
        cache_dir (Path): Directory to store cached embeddings
        bug_embeddings (Dict): Dictionary mapping bug_id to embedding
        code_embeddings (Dict): Dictionary mapping filepath to embedding
    """
    
    def __init__(self, model, cache_dir: str = "data/processed/embeddings"):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model: Model object with encode_text() and encode_code() methods
            cache_dir: Directory to store cached embeddings
        """
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.bug_embeddings: Dict[int, np.ndarray] = {}
        self.code_embeddings: Dict[str, np.ndarray] = {}
    
    def generate_bug_embeddings(self, bug_reports_df, force_regenerate: bool = False) -> Dict[int, np.ndarray]:
        """
        Generate embeddings for all bug reports.
        
        Args:
            bug_reports_df: DataFrame containing bug reports
            force_regenerate: If True, regenerate embeddings even if cached
            
        Returns:
            Dictionary mapping bug_id to embedding
        """
        cache_file = self.cache_dir / "bug_embeddings.pkl"
        
        # Try to load from cache
        if not force_regenerate and cache_file.exists():
            print(f"[INFO] Loading cached bug embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.bug_embeddings = pickle.load(f)
            print(f"[OK] Loaded {len(self.bug_embeddings)} cached bug embeddings")
            return self.bug_embeddings
        
        # Generate embeddings
        print(f"[INFO] Generating embeddings for {len(bug_reports_df)} bug reports...")
        self.bug_embeddings = {}
        
        for idx, row in tqdm(bug_reports_df.iterrows(), desc="Encoding bug reports"):
            bug_id = row['bug_id']
            bug_report = row['bug_report']
            
            # Generate embedding
            embedding = self.model.encode_text(bug_report)
            
            # Store (embedding is shape [1, 768], we store [768])
            self.bug_embeddings[bug_id] = embedding[0]
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.bug_embeddings, f)
        
        print(f"[OK] Generated and cached {len(self.bug_embeddings)} bug embeddings")
        return self.bug_embeddings
    
    def generate_code_embeddings(self, files_data: List[Dict], force_regenerate: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all code files.
        
        Args:
            files_data: List of dictionaries containing file information
            force_regenerate: If True, regenerate embeddings even if cached
            
        Returns:
            Dictionary mapping filepath to embedding
        """
        cache_file = self.cache_dir / "code_embeddings.pkl"
        
        # Try to load from cache
        if not force_regenerate and cache_file.exists():
            print(f"[INFO] Loading cached code embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.code_embeddings = pickle.load(f)
            print(f"[OK] Loaded {len(self.code_embeddings)} cached code embeddings")
            return self.code_embeddings
        
        # Generate embeddings
        print(f"[INFO] Generating embeddings for {len(files_data)} code files...")
        self.code_embeddings = {}
        
        for file_info in tqdm(files_data, desc="Encoding code files"):
            filepath = file_info['filepath']
            source_code = file_info['source_code']
            
            # Generate embedding
            embedding = self.model.encode_code(source_code)
            
            # Store (embedding is shape [1, 768], we store [768])
            self.code_embeddings[filepath] = embedding[0]
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.code_embeddings, f)
        
        print(f"[OK] Generated and cached {len(self.code_embeddings)} code embeddings")
        return self.code_embeddings
    
    def get_bug_embedding(self, bug_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a specific bug report.
        
        Args:
            bug_id: ID of the bug report
            
        Returns:
            Embedding vector or None if not found
        """
        return self.bug_embeddings.get(bug_id)
    
    def get_code_embedding(self, filepath: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific code file.
        
        Args:
            filepath: Path of the code file
            
        Returns:
            Embedding vector or None if not found
        """
        return self.code_embeddings.get(filepath)
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the embeddings.
        
        Returns:
            Dictionary containing statistics
        """
        stats = {
            'num_bug_embeddings': len(self.bug_embeddings),
            'num_code_embeddings': len(self.code_embeddings),
            'embedding_dimension': 768,  # CodeBERT-base hidden size
            'cache_dir': str(self.cache_dir)
        }
        
        if self.bug_embeddings:
            bug_emb_array = np.array(list(self.bug_embeddings.values()))
            stats['bug_embedding_stats'] = {
                'mean': float(bug_emb_array.mean()),
                'std': float(bug_emb_array.std()),
                'min': float(bug_emb_array.min()),
                'max': float(bug_emb_array.max())
            }
        
        if self.code_embeddings:
            code_emb_array = np.array(list(self.code_embeddings.values()))
            stats['code_embedding_stats'] = {
                'mean': float(code_emb_array.mean()),
                'std': float(code_emb_array.std()),
                'min': float(code_emb_array.min()),
                'max': float(code_emb_array.max())
            }
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self.bug_embeddings = {}
        self.code_embeddings = {}
        
        # Remove cache files
        bug_cache = self.cache_dir / "bug_embeddings.pkl"
        code_cache = self.cache_dir / "code_embeddings.pkl"
        
        if bug_cache.exists():
            bug_cache.unlink()
        if code_cache.exists():
            code_cache.unlink()
        
        print("[OK] Cache cleared")


def main():
    """Main function to test embedding generation."""
    print("=" * 80)
    print("Phase 5: Embedding Generation - Testing")
    print("=" * 80)
    
    # Import required modules
    from dataset import BugDataset, create_sample_dataset
    from extractor import RepositoryExtractor, create_test_repository
    
    # Initialize CodeBERT model
    print("\n1. Loading CodeBERT model...")
    codebert = CodeBERTModel()
    codebert.load_model()
    
    # Initialize embedding generator
    print("\n2. Initializing embedding generator...")
    embedding_gen = EmbeddingGenerator(codebert)
    
    # Create and load dataset
    print("\n3. Loading bug reports...")
    create_sample_dataset()
    dataset = BugDataset("data/raw/bug_reports.csv")
    bug_reports_df = dataset.load()
    print(f"   Loaded {len(bug_reports_df)} bug reports")
    
    # Create test repository and extract files
    print("\n4. Extracting code files...")
    repo_path = create_test_repository()
    extractor = RepositoryExtractor(repo_path)
    files_data = extractor.extract_python_files()
    print(f"   Extracted {len(files_data)} Python files")
    
    # Generate bug embeddings
    print("\n5. Generating bug report embeddings...")
    bug_embeddings = embedding_gen.generate_bug_embeddings(bug_reports_df)
    print(f"   Generated {len(bug_embeddings)} bug embeddings")
    
    # Generate code embeddings
    print("\n6. Generating code file embeddings...")
    code_embeddings = embedding_gen.generate_code_embeddings(files_data)
    print(f"   Generated {len(code_embeddings)} code embeddings")
    
    # Display statistics
    print("\n7. Embedding statistics:")
    stats = embedding_gen.get_statistics()
    print(f"   Bug embeddings: {stats['num_bug_embeddings']}")
    print(f"   Code embeddings: {stats['num_code_embeddings']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    
    if 'bug_embedding_stats' in stats:
        print(f"\n   Bug embedding stats:")
        for key, value in stats['bug_embedding_stats'].items():
            print(f"     {key}: {value:.4f}")
    
    if 'code_embedding_stats' in stats:
        print(f"\n   Code embedding stats:")
        for key, value in stats['code_embedding_stats'].items():
            print(f"     {key}: {value:.4f}")
    
    # Test retrieving specific embeddings
    print("\n8. Testing embedding retrieval...")
    bug_emb = embedding_gen.get_bug_embedding(1)
    if bug_emb is not None:
        print(f"   Bug 1 embedding shape: {bug_emb.shape}")
        print(f"   First 5 values: {bug_emb[:5]}")
    
    code_emb = embedding_gen.get_code_embedding('main.py')
    if code_emb is not None:
        print(f"   main.py embedding shape: {code_emb.shape}")
        print(f"   First 5 values: {code_emb[:5]}")
    
    # Test cache loading (regenerate should be fast from cache)
    print("\n9. Testing cache loading (should be instant)...")
    embedding_gen2 = EmbeddingGenerator(codebert)
    bug_embeddings_cached = embedding_gen2.generate_bug_embeddings(bug_reports_df)
    print(f"   Loaded {len(bug_embeddings_cached)} bug embeddings from cache")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 5 Complete! Embedding generation is ready.")
    print("=" * 80)
    print("\nKey Points:")
    print("  - Embeddings are cached to avoid recomputation")
    print("  - Each embedding is 768-dimensional (CodeBERT-base)")
    print("  - Bug reports and code files are encoded separately")
    print("  - Cache is stored in data/processed/embeddings/")


if __name__ == "__main__":
    main()