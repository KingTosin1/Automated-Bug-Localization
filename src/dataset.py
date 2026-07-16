"""
Dataset preparation and loading module.

This module handles loading and preparing bug report datasets for the bug localization system.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


class BugDataset:
    """
    A class to handle bug report datasets.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing bug reports
        data_path (Path): Path to the dataset file
    """
    
    def __init__(self, data_path: str = "data/raw/bug_reports.csv"):
        """
        Initialize the BugDataset.
        
        Args:
            data_path: Path to the CSV file containing bug reports
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load the bug report dataset from CSV.
        
        Returns:
            DataFrame containing bug reports
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please ensure the CSV file exists."
            )
        
        # Load CSV
        self.df = pd.read_csv(self.data_path)
        
        # Validate required columns
        required_columns = ['bug_id', 'repository', 'bug_report', 'file_path']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Dataset must have: {required_columns}"
            )
        
        print(f"[OK] Loaded {len(self.df)} bug reports from {self.data_path}")
        return self.df
    
    def get_bug_report(self, bug_id: int) -> Tuple[str, str]:
        """
        Get a specific bug report by ID.
        
        Args:
            bug_id: The ID of the bug report to retrieve
            
        Returns:
            Tuple of (bug_report_text, file_path)
            
        Raises:
            ValueError: If bug_id is not found
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        bug_row = self.df[self.df['bug_id'] == bug_id]
        
        if bug_row.empty:
            raise ValueError(f"Bug ID {bug_id} not found in dataset.")
        
        bug_report = bug_row.iloc[0]['bug_report']
        file_path = bug_row.iloc[0]['file_path']
        
        return bug_report, file_path
    
    def get_all_bug_reports(self) -> pd.DataFrame:
        """
        Get all bug reports.
        
        Returns:
            DataFrame containing all bug reports
            
        Raises:
            ValueError: If dataset is not loaded
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        return self.df
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        stats = {
            'total_bugs': len(self.df),
            'unique_repositories': self.df['repository'].nunique(),
            'repositories': self.df['repository'].unique().tolist(),
            'avg_bug_report_length': self.df['bug_report'].str.len().mean(),
            'avg_file_path_length': self.df['file_path'].str.len().mean()
        }
        
        return stats
    
    def display_sample(self, n: int = 5) -> None:
        """
        Display sample bug reports.
        
        Args:
            n: Number of samples to display
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        print(f"\n{'='*80}")
        print(f"Sample Bug Reports (showing {min(n, len(self.df))} of {len(self.df)})")
        print(f"{'='*80}")
        
        for idx, row in self.df.head(n).iterrows():
            print(f"\nBug ID: {row['bug_id']}")
            print(f"Repository: {row['repository']}")
            print(f"File Path: {row['file_path']}")
            print(f"Bug Report: {row['bug_report'][:200]}...")
            print(f"{'-'*80}")


def create_sample_dataset(output_path: str = "data/raw/bug_reports.csv") -> None:
    """
    Create a sample bug report dataset for testing.
    
    Args:
        output_path: Path to save the sample dataset
    """
    # Sample bug reports from popular Python repositories
    sample_data = {
        'bug_id': [1, 2, 3, 4, 5],
        'repository': [
            'requests',
            'flask',
            'django',
            'pandas',
            'numpy'
        ],
        'bug_report': [
            "Connection timeout when making HTTP requests to slow servers. The request hangs indefinitely instead of timing out after the specified timeout period.",
            "Blueprint routes are not registering correctly when using dynamic URL parameters. The route pattern matching fails for URLs with multiple parameters.",
            "Django ORM query returns incorrect results when using filter with Q objects and multiple conditions. The AND/OR logic is not working as expected.",
            "DataFrame merge operation produces duplicate columns when merging on index with overlapping column names. The suffixes parameter is not being applied correctly.",
            "Array indexing with negative steps produces wrong results. When using slicing with negative step values, the array is not reversed correctly."
        ],
        'file_path': [
            'requests/sessions.py',
            'flask/blueprints.py',
            'django/db/models/query.py',
            'pandas/core/reshape/merge.py',
            'numpy/core/fromnumeric.py'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[OK] Sample dataset created at {output_path}")
    print(f"   Total bug reports: {len(df)}")


def main():
    """Main function to test dataset loading."""
    print("=" * 80)
    print("Phase 2: Dataset Preparation - Testing")
    print("=" * 80)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    create_sample_dataset()
    
    # Load dataset
    print("\n2. Loading dataset...")
    dataset = BugDataset("data/raw/bug_reports.csv")
    df = dataset.load()
    
    # Display statistics
    print("\n3. Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Display sample
    print("\n4. Sample bug reports:")
    dataset.display_sample(n=3)
    
    # Test getting specific bug report
    print("\n5. Testing get_bug_report()...")
    bug_report, file_path = dataset.get_bug_report(1)
    print(f"   Bug ID 1:")
    print(f"   File: {file_path}")
    print(f"   Report: {bug_report[:100]}...")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 2 Complete! Dataset is ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()