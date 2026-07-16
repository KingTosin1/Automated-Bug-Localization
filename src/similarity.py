"""
Similarity computation module.

This module implements cosine similarity computation and ranking functionality
to match bug reports with source code files based on their embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    """
    A class to compute similarity between bug reports and code files.
    
    Attributes:
        bug_embeddings (Dict): Dictionary mapping bug_id to embedding
        code_embeddings (Dict): Dictionary mapping filepath to embedding
    """
    
    def __init__(self, bug_embeddings: Dict[int, np.ndarray], 
                 code_embeddings: Dict[str, np.ndarray]):
        """
        Initialize the SimilarityEngine.
        
        Args:
            bug_embeddings: Dictionary mapping bug_id to embedding vector
            code_embeddings: Dictionary mapping filepath to embedding vector
        """
        self.bug_embeddings = bug_embeddings
        self.code_embeddings = code_embeddings
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, 
                                  embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (between -1 and 1, typically 0 to 1 for similar items)
        """
        # Reshape to 2D arrays if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        return float(similarity)
    
    def rank_files(self, bug_id: int, top_k: int = 5) -> List[Dict]:
        """
        Rank code files by similarity to a bug report.
        
        Args:
            bug_id: ID of the bug report
            top_k: Number of top files to return
            
        Returns:
            List of dictionaries containing ranked file information:
            - rank: Rank position
            - filepath: Path of the file
            - similarity: Cosine similarity score
            - confidence: Percentage confidence (similarity * 100)
        """
        if bug_id not in self.bug_embeddings:
            raise ValueError(f"Bug ID {bug_id} not found in embeddings")
        
        bug_embedding = self.bug_embeddings[bug_id]
        
        # Compute similarity with all code files
        similarities = []
        for filepath, code_embedding in self.code_embeddings.items():
            similarity = self.compute_cosine_similarity(bug_embedding, code_embedding)
            similarities.append({
                'filepath': filepath,
                'similarity': similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top-k results
        top_results = similarities[:top_k]
        
        # Add rank and confidence
        ranked_results = []
        for idx, result in enumerate(top_results, 1):
            ranked_results.append({
                'rank': idx,
                'filepath': result['filepath'],
                'similarity': result['similarity'],
                'confidence': result['similarity'] * 100
            })
        
        return ranked_results
    
    def rank_all_bugs(self, top_k: int = 5) -> Dict[int, List[Dict]]:
        """
        Rank files for all bug reports.
        
        Args:
            top_k: Number of top files to return for each bug
            
        Returns:
            Dictionary mapping bug_id to list of ranked files
        """
        results = {}
        
        for bug_id in self.bug_embeddings.keys():
            ranked_files = self.rank_files(bug_id, top_k=top_k)
            results[bug_id] = ranked_files
        
        return results
    
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Compute similarity matrix between all bug reports and code files.
        
        Returns:
            2D numpy array where rows are bug reports and columns are code files
        """
        # Stack all embeddings
        bug_matrix = np.array(list(self.bug_embeddings.values()))
        code_matrix = np.array(list(self.code_embeddings.values()))
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(bug_matrix, code_matrix)
        
        return similarity_matrix
    
    def get_most_similar_pair(self) -> Tuple[int, str, float]:
        """
        Find the most similar bug-code pair.
        
        Returns:
            Tuple of (bug_id, filepath, similarity_score)
        """
        similarity_matrix = self.get_similarity_matrix()
        
        # Find maximum similarity
        max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        bug_id = list(self.bug_embeddings.keys())[max_idx[0]]
        filepath = list(self.code_embeddings.keys())[max_idx[1]]
        max_similarity = similarity_matrix[max_idx]
        
        return bug_id, filepath, float(max_similarity)


def display_ranked_results(bug_id: int, ranked_files: List[Dict], 
                          bug_report: str = "") -> None:
    """
    Display ranked results in a formatted way.
    
    Args:
        bug_id: ID of the bug report
        ranked_files: List of ranked file dictionaries
        bug_report: Optional bug report text to display
    """
    print(f"\n{'='*80}")
    print(f"Bug Localization Results for Bug ID: {bug_id}")
    print(f"{'='*80}")
    
    if bug_report:
        print(f"\nBug Report: {bug_report[:200]}...")
    
    print(f"\nTop {len(ranked_files)} Most Likely Files:")
    print(f"{'-'*80}")
    
    for result in ranked_files:
        print(f"\nRank #{result['rank']}: {result['filepath']}")
        print(f"  Similarity Score: {result['similarity']:.4f}")
        print(f"  Confidence:       {result['confidence']:.2f}%")
    
    print(f"\n{'='*80}")


def main():
    """Main function to test similarity engine."""
    print("=" * 80)
    print("Phase 6: Similarity Engine - Testing")
    print("=" * 80)
    
    # Import required modules
    from codebert_model import CodeBERTModel
    from embedding_generator import EmbeddingGenerator
    from dataset import BugDataset, create_sample_dataset
    from extractor import RepositoryExtractor, create_test_repository
    
    # Initialize model and generate embeddings
    print("\n1. Loading model and generating embeddings...")
    codebert = CodeBERTModel()
    codebert.load_model()
    
    embedding_gen = EmbeddingGenerator(codebert)
    
    # Load data
    create_sample_dataset()
    dataset = BugDataset("data/raw/bug_reports.csv")
    bug_reports_df = dataset.load()
    
    repo_path = create_test_repository()
    extractor = RepositoryExtractor(repo_path)
    files_data = extractor.extract_python_files()
    
    # Generate embeddings
    bug_embeddings = embedding_gen.generate_bug_embeddings(bug_reports_df)
    code_embeddings = embedding_gen.generate_code_embeddings(files_data)
    
    # Initialize similarity engine
    print("\n2. Initializing similarity engine...")
    similarity_engine = SimilarityEngine(bug_embeddings, code_embeddings)
    print("[OK] Similarity engine initialized")
    
    # Test single bug localization
    print("\n3. Testing single bug localization...")
    bug_id = 1
    bug_report, expected_file = dataset.get_bug_report(bug_id)
    ranked_files = similarity_engine.rank_files(bug_id, top_k=3)
    
    print(f"\nExpected file: {expected_file}")
    display_ranked_results(bug_id, ranked_files, bug_report)
    
    # Test all bugs
    print("\n4. Testing all bug reports...")
    all_results = similarity_engine.rank_all_bugs(top_k=3)
    
    for bug_id, ranked_files in all_results.items():
        bug_report, _ = dataset.get_bug_report(bug_id)
        print(f"\nBug {bug_id}: Top file is {ranked_files[0]['filepath']}")
    
    # Test similarity matrix
    print("\n5. Testing similarity matrix...")
    sim_matrix = similarity_engine.get_similarity_matrix()
    print(f"   Similarity matrix shape: {sim_matrix.shape}")
    print(f"   (Bugs: {sim_matrix.shape[0]}, Files: {sim_matrix.shape[1]})")
    
    # Find most similar pair
    print("\n6. Finding most similar bug-code pair...")
    best_bug_id, best_filepath, best_similarity = similarity_engine.get_most_similar_pair()
    print(f"   Bug ID: {best_bug_id}")
    print(f"   File: {best_filepath}")
    print(f"   Similarity: {best_similarity:.4f}")
    
    # Test with different top_k values
    print("\n7. Testing different top_k values...")
    for k in [1, 3, 5]:
        ranked = similarity_engine.rank_files(1, top_k=k)
        print(f"   Top-{k}: {[r['filepath'] for r in ranked]}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 6 Complete! Similarity engine is ready.")
    print("=" * 80)
    print("\nKey Points:")
    print("  - Cosine similarity measures how similar two embeddings are")
    print("  - Higher scores (closer to 1) indicate more similar items")
    print("  - Files are ranked by similarity to the bug report")
    print("  - Top-K results show the most likely buggy files")


if __name__ == "__main__":
    main()