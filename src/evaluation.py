"""
Evaluation metrics module.

This module implements evaluation metrics for bug localization:
- Top-1 Accuracy
- Top-5 Accuracy
- Mean Reciprocal Rank (MRR)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from similarity import SimilarityEngine


class BugLocalizationEvaluator:
    """
    A class to evaluate bug localization performance.
    
    Attributes:
        similarity_engine (SimilarityEngine): Engine for computing similarities
        dataset: Dataset containing bug reports and ground truth
    """
    
    def __init__(self, similarity_engine: SimilarityEngine, dataset):
        """
        Initialize the evaluator.
        
        Args:
            similarity_engine: SimilarityEngine instance
            dataset: BugDataset instance with ground truth
        """
        self.similarity_engine = similarity_engine
        self.dataset = dataset
    
    def compute_top_k_accuracy(self, bug_id: int, top_k: int = 5) -> bool:
        """
        Check if the correct file is in the top-k predictions.
        
        Args:
            bug_id: ID of the bug report
            top_k: Number of top files to consider
            
        Returns:
            True if correct file is in top-k, False otherwise
        """
        # Get expected file
        _, expected_file = self.dataset.get_bug_report(bug_id)
        
        # Get top-k predictions
        ranked_files = self.similarity_engine.rank_files(bug_id, top_k=top_k)
        top_files = [r['filepath'] for r in ranked_files]
        
        # Check if expected file is in top-k
        return expected_file in top_files
    
    def compute_reciprocal_rank(self, bug_id: int) -> float:
        """
        Compute the reciprocal rank for a bug report.
        
        Reciprocal Rank = 1 / rank of the first correct result
        
        Args:
            bug_id: ID of the bug report
            
        Returns:
            Reciprocal rank (0 if not found, 1/rank if found)
        """
        # Get expected file
        _, expected_file = self.dataset.get_bug_report(bug_id)
        
        # Get all ranked files
        ranked_files = self.similarity_engine.rank_files(bug_id, top_k=len(self.similarity_engine.code_embeddings))
        
        # Find rank of expected file
        for result in ranked_files:
            if result['filepath'] == expected_file:
                return 1.0 / result['rank']
        
        # Not found
        return 0.0
    
    def evaluate_all_bugs(self, top_k_values: List[int] = [1, 5]) -> Dict:
        """
        Evaluate all bug reports in the dataset.
        
        Args:
            top_k_values: List of k values for Top-K accuracy
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'top_k_accuracy': {k: [] for k in top_k_values},
            'reciprocal_ranks': [],
            'bug_ids': []
        }
        
        for bug_id in self.dataset.df['bug_id']:
            results['bug_ids'].append(bug_id)
            
            # Compute Top-K accuracy for each k
            for k in top_k_values:
                is_correct = self.compute_top_k_accuracy(bug_id, top_k=k)
                results['top_k_accuracy'][k].append(1 if is_correct else 0)
            
            # Compute reciprocal rank
            rr = self.compute_reciprocal_rank(bug_id)
            results['reciprocal_ranks'].append(rr)
        
        return results
    
    def compute_metrics(self, top_k_values: List[int] = [1, 5]) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            top_k_values: List of k values for Top-K accuracy
            
        Returns:
            Dictionary containing computed metrics
        """
        # Evaluate all bugs
        results = self.evaluate_all_bugs(top_k_values)
        
        # Compute metrics
        metrics = {}
        
        # Top-K Accuracy
        for k in top_k_values:
            accuracy = np.mean(results['top_k_accuracy'][k])
            metrics[f'top_{k}_accuracy'] = accuracy
        
        # Mean Reciprocal Rank (MRR)
        mrr = np.mean(results['reciprocal_ranks'])
        metrics['mrr'] = mrr
        
        # Store detailed results
        metrics['detailed_results'] = results
        
        return metrics
    
    def print_metrics(self, metrics: Dict) -> None:
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary containing computed metrics
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        print("\nAccuracy Metrics:")
        print("-" * 80)
        for key, value in metrics.items():
            if key.startswith('top_') and key.endswith('_accuracy'):
                k = key.replace('top_', '').replace('_accuracy', '')
                print(f"  Top-{k} Accuracy: {value:.2%} ({int(value * len(metrics['detailed_results']['bug_ids']))}/{len(metrics['detailed_results']['bug_ids'])})")
        
        print(f"\n  Mean Reciprocal Rank (MRR): {metrics['mrr']:.4f}")
        
        # Print detailed results
        print("\n" + "-" * 80)
        print("Detailed Results by Bug:")
        print("-" * 80)
        
        detailed = metrics['detailed_results']
        for i, bug_id in enumerate(detailed['bug_ids']):
            print(f"\nBug {bug_id}:")
            for k in sorted([int(key.replace('top_', '').replace('_accuracy', '')) for key in metrics.keys() if key.startswith('top_')]):
                correct = detailed['top_k_accuracy'][k][i]
                status = "[OK]" if correct else "[FAIL]"
                print(f"  Top-{k}: {status}")
            print(f"  Reciprocal Rank: {detailed['reciprocal_ranks'][i]:.4f}")
        
        print("\n" + "=" * 80)
    
    def generate_comparison_table(self, metrics_list: List[Tuple[str, Dict]]) -> pd.DataFrame:
        """
        Generate a comparison table for multiple models.
        
        Args:
            metrics_list: List of (model_name, metrics) tuples
            
        Returns:
            DataFrame containing comparison table
        """
        rows = []
        
        for model_name, metrics in metrics_list:
            row = {'Model': model_name}
            
            # Add Top-K accuracy
            for key, value in metrics.items():
                if key.startswith('top_') and key.endswith('_accuracy'):
                    k = key.replace('top_', '').replace('_accuracy', '')
                    row[f'Top-{k} Accuracy'] = f"{value:.2%}"
                elif key == 'mrr':
                    row['MRR'] = f"{value:.4f}"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df


def main():
    """Main function to test evaluation metrics."""
    print("=" * 80)
    print("Phase 8: Evaluation Metrics - Testing")
    print("=" * 80)
    
    # Import required modules
    from codebert_model import CodeBERTModel
    from embedding_generator import EmbeddingGenerator
    from dataset import BugDataset, create_sample_dataset
    from extractor import RepositoryExtractor, create_test_repository
    
    # Setup pipeline components
    print("\n1. Setting up components...")
    
    # Load model
    codebert = CodeBERTModel()
    codebert.load_model()
    
    # Load data
    create_sample_dataset()
    dataset = BugDataset("data/raw/bug_reports.csv")
    bug_reports_df = dataset.load()
    
    repo_path = create_test_repository()
    extractor = RepositoryExtractor(repo_path)
    files_data = extractor.extract_python_files()
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator(codebert)
    bug_embeddings = embedding_gen.generate_bug_embeddings(bug_reports_df)
    code_embeddings = embedding_gen.generate_code_embeddings(files_data)
    
    # Initialize similarity engine
    similarity_engine = SimilarityEngine(bug_embeddings, code_embeddings)
    
    # Initialize evaluator
    print("\n2. Initializing evaluator...")
    evaluator = BugLocalizationEvaluator(similarity_engine, dataset)
    print("[OK] Evaluator initialized")
    
    # Compute metrics
    print("\n3. Computing evaluation metrics...")
    metrics = evaluator.compute_metrics(top_k_values=[1, 3, 5])
    
    # Print metrics
    print("\n4. Displaying results...")
    evaluator.print_metrics(metrics)
    
    # Test with different top-k values
    print("\n5. Testing with different Top-K values...")
    for k in [1, 2, 3, 5]:
        metrics_k = evaluator.compute_metrics(top_k_values=[k])
        accuracy = metrics_k[f'top_{k}_accuracy']
        print(f"   Top-{k} Accuracy: {accuracy:.2%}")
    
    # Summary
    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 8 Complete! Evaluation metrics are ready.")
    print("=" * 80)
    print("\nKey Points:")
    print("  - Top-K Accuracy: Percentage of bugs where correct file is in top-K")
    print("  - MRR: Mean Reciprocal Rank - average of 1/rank for correct predictions")
    print("  - Higher values indicate better performance")
    print("  - These metrics are standard in information retrieval tasks")


if __name__ == "__main__":
    main()