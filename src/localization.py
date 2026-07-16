"""
Bug localization pipeline module.

This module combines all components into a unified pipeline for bug localization.
"""

from typing import Dict, List, Optional
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from codebert_model import CodeBERTModel
from embedding_generator import EmbeddingGenerator
from similarity import SimilarityEngine, display_ranked_results
from dataset import BugDataset
from extractor import RepositoryExtractor


class BugLocalizationPipeline:
    """
    A complete bug localization pipeline.
    
    This class integrates all components:
    - Dataset loading
    - Repository processing
    - Embedding generation
    - Similarity computation
    
    Attributes:
        model: The embedding model (CodeBERT or CodeT5)
        dataset (BugDataset): Dataset handler
        extractor (RepositoryExtractor): Repository file extractor
        embedding_gen (EmbeddingGenerator): Embedding generator
        similarity_engine (SimilarityEngine): Similarity computation engine
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base",
                 repo_path: str = "data/raw/test_repo",
                 dataset_path: str = "data/raw/bug_reports.csv"):
        """
        Initialize the bug localization pipeline.
        
        Args:
            model_name: Name of the pre-trained model
            repo_path: Path to the repository to analyze
            dataset_path: Path to the bug reports dataset
        """
        self.model_name = model_name
        self.repo_path = repo_path
        self.dataset_path = dataset_path
        
        self.model = None
        self.dataset = None
        self.extractor = None
        self.embedding_gen = None
        self.similarity_engine = None
    
    def setup(self, force_regenerate_embeddings: bool = False) -> None:
        """
        Set up the pipeline by loading all components.
        
        Args:
            force_regenerate_embeddings: If True, regenerate embeddings even if cached
        """
        print("=" * 80)
        print("Setting up Bug Localization Pipeline")
        print("=" * 80)
        
        # Load model
        print("\n1. Loading model...")
        self.model = CodeBERTModel(model_name=self.model_name)
        self.model.load_model()
        
        # Load dataset
        print("\n2. Loading bug reports...")
        self.dataset = BugDataset(self.dataset_path)
        bug_reports_df = self.dataset.load()
        
        # Extract repository files
        print("\n3. Extracting repository files...")
        self.extractor = RepositoryExtractor(self.repo_path)
        files_data = self.extractor.extract_python_files()
        
        # Generate embeddings
        print("\n4. Generating embeddings...")
        self.embedding_gen = EmbeddingGenerator(self.model)
        bug_embeddings = self.embedding_gen.generate_bug_embeddings(
            bug_reports_df, 
            force_regenerate=force_regenerate_embeddings
        )
        code_embeddings = self.embedding_gen.generate_code_embeddings(
            files_data,
            force_regenerate=force_regenerate_embeddings
        )
        
        # Initialize similarity engine
        print("\n5. Initializing similarity engine...")
        self.similarity_engine = SimilarityEngine(bug_embeddings, code_embeddings)
        
        print("\n" + "=" * 80)
        print("[OK] Pipeline setup complete!")
        print("=" * 80)
    
    def localize_bug(self, bug_id: int, top_k: int = 5) -> List[Dict]:
        """
        Localize a bug by finding the most likely files.
        
        Args:
            bug_id: ID of the bug report
            top_k: Number of top files to return
            
        Returns:
            List of ranked file dictionaries with:
            - rank: Rank position
            - filepath: Path of the file
            - similarity: Cosine similarity score
            - confidence: Percentage confidence
        """
        if self.similarity_engine is None:
            raise ValueError("Pipeline not set up. Call setup() first.")
        
        # Get bug report text
        bug_report, expected_file = self.dataset.get_bug_report(bug_id)
        
        # Rank files
        ranked_files = self.similarity_engine.rank_files(bug_id, top_k=top_k)
        
        # Display results
        display_ranked_results(bug_id, ranked_files, bug_report)
        
        # Print expected file location
        print(f"\nExpected buggy file: {expected_file}")
        
        # Check if expected file is in top-k
        top_files = [r['filepath'] for r in ranked_files]
        if expected_file in top_files:
            rank = top_files.index(expected_file) + 1
            print(f"[OK] Expected file found at rank #{rank}")
        else:
            print(f"[INFO] Expected file not in top-{top_k}")
        
        return ranked_files
    
    def localize_all_bugs(self, top_k: int = 5) -> Dict[int, List[Dict]]:
        """
        Localize all bugs in the dataset.
        
        Args:
            top_k: Number of top files to return for each bug
            
        Returns:
            Dictionary mapping bug_id to list of ranked files
        """
        if self.similarity_engine is None:
            raise ValueError("Pipeline not set up. Call setup() first.")
        
        print("\n" + "=" * 80)
        print(f"Localizing All Bugs (Top-{top_k})")
        print("=" * 80)
        
        all_results = {}
        
        for bug_id in self.dataset.df['bug_id']:
            print(f"\n--- Bug {bug_id} ---")
            ranked_files = self.localize_bug(bug_id, top_k=top_k)
            all_results[bug_id] = ranked_files
        
        return all_results
    
    def get_pipeline_statistics(self) -> Dict:
        """
        Get statistics about the pipeline.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        if self.similarity_engine is None:
            raise ValueError("Pipeline not set up. Call setup() first.")
        
        stats = {
            'model_name': self.model_name,
            'repo_path': self.repo_path,
            'dataset_path': self.dataset_path,
            'num_bug_reports': len(self.dataset.df),
            'num_code_files': len(self.extractor.files_data),
            'embedding_dimension': 768
        }
        
        # Add embedding statistics
        embedding_stats = self.embedding_gen.get_statistics()
        stats.update(embedding_stats)
        
        return stats


def main():
    """Main function to test the bug localization pipeline."""
    print("=" * 80)
    print("Phase 7: Bug Localization Pipeline - Testing")
    print("=" * 80)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = BugLocalizationPipeline()
    
    # Setup pipeline
    print("\n2. Setting up pipeline...")
    pipeline.setup(force_regenerate_embeddings=False)
    
    # Display pipeline statistics
    print("\n3. Pipeline statistics:")
    stats = pipeline.get_pipeline_statistics()
    for key, value in stats.items():
        if key not in ['bug_embedding_stats', 'code_embedding_stats']:
            print(f"   {key}: {value}")
    
    # Test single bug localization
    print("\n4. Testing single bug localization...")
    results = pipeline.localize_bug(bug_id=1, top_k=3)
    
    # Test with different bug
    print("\n5. Testing with another bug...")
    results = pipeline.localize_bug(bug_id=2, top_k=3)
    
    # Test localizing all bugs
    print("\n6. Testing batch localization...")
    all_results = pipeline.localize_all_bugs(top_k=3)
    
    # Summary
    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 7 Complete! Bug localization pipeline is ready.")
    print("=" * 80)
    print("\nPipeline Components:")
    print("  1. CodeBERT Model - Generates embeddings")
    print("  2. Dataset - Loads bug reports")
    print("  3. Extractor - Reads repository files")
    print("  4. Embedding Generator - Creates and caches embeddings")
    print("  5. Similarity Engine - Ranks files by similarity")
    print("\nUsage:")
    print("  pipeline = BugLocalizationPipeline()")
    print("  pipeline.setup()")
    print("  results = pipeline.localize_bug(bug_id=1, top_k=5)")


if __name__ == "__main__":
    main()