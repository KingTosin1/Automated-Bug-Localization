"""
Repository file extraction module.

This module handles reading Python source files from repositories and extracting their content.
"""

from pathlib import Path
from typing import List, Dict, Optional
import os


class RepositoryExtractor:
    """
    A class to extract Python files from repositories.
    
    Attributes:
        repo_path (Path): Path to the repository
        files_data (List[Dict]): List of extracted file information
    """
    
    def __init__(self, repo_path: str):
        """
        Initialize the RepositoryExtractor.
        
        Args:
            repo_path: Path to the repository directory
        """
        self.repo_path = Path(repo_path)
        self.files_data: List[Dict] = []
    
    def extract_python_files(self) -> List[Dict]:
        """
        Extract all Python files from the repository.
        
        Returns:
            List of dictionaries containing file information:
            - filename: Name of the file
            - filepath: Relative path from repo root
            - full_path: Absolute path
            - source_code: Content of the file
            - size_bytes: File size in bytes
            - lines: Number of lines
            
        Raises:
            ValueError: If repository path doesn't exist
        """
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        
        if not self.repo_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.repo_path}")
        
        self.files_data = []
        
        # Walk through repository and find all .py files
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden directories (like .git, __pycache__, etc.)
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                # Skip __init__.py files as they typically don't contain bugs
                if file == '__init__.py':
                    continue
                
                if file.endswith('.py'):
                    full_path = Path(root) / file
                    relative_path = full_path.relative_to(self.repo_path)
                    
                    try:
                        # Read file content
                        with open(full_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()
                        
                        # Skip files with only comments or very short files
                        lines = source_code.splitlines()
                        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                        
                        if len(code_lines) < 3:  # Skip files with less than 3 lines of actual code
                            continue
                        
                        # Get file stats
                        file_size = full_path.stat().st_size
                        num_lines = len(lines)
                        
                        file_info = {
                            'filename': file,
                            'filepath': str(relative_path),
                            'full_path': str(full_path),
                            'source_code': source_code,
                            'size_bytes': file_size,
                            'lines': num_lines
                        }
                        
                        self.files_data.append(file_info)
                        
                    except Exception as e:
                        print(f"[WARNING] Could not read {full_path}: {e}")
                        continue
        
        print(f"[OK] Extracted {len(self.files_data)} Python files from {self.repo_path}")
        return self.files_data
    
    def get_file(self, filepath: str) -> Optional[Dict]:
        """
        Get a specific file by its relative path.
        
        Args:
            filepath: Relative path of the file to retrieve
            
        Returns:
            Dictionary containing file information, or None if not found
        """
        for file_info in self.files_data:
            if file_info['filepath'] == filepath:
                return file_info
        return None
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the extracted files.
        
        Returns:
            Dictionary containing statistics
        """
        if not self.files_data:
            return {
                'total_files': 0,
                'total_lines': 0,
                'total_size_bytes': 0,
                'avg_lines_per_file': 0,
                'avg_size_per_file': 0
            }
        
        total_lines = sum(f['lines'] for f in self.files_data)
        total_size = sum(f['size_bytes'] for f in self.files_data)
        
        stats = {
            'total_files': len(self.files_data),
            'total_lines': total_lines,
            'total_size_bytes': total_size,
            'avg_lines_per_file': total_lines / len(self.files_data),
            'avg_size_per_file': total_size / len(self.files_data),
            'files': [f['filepath'] for f in self.files_data]
        }
        
        return stats
    
    def display_files(self, n: int = 10) -> None:
        """
        Display information about extracted files.
        
        Args:
            n: Number of files to display
        """
        if not self.files_data:
            print("[WARNING] No files extracted. Call extract_python_files() first.")
            return
        
        print(f"\n{'='*80}")
        print(f"Extracted Python Files (showing {min(n, len(self.files_data))} of {len(self.files_data)})")
        print(f"{'='*80}")
        
        for file_info in self.files_data[:n]:
            print(f"\nFile: {file_info['filepath']}")
            print(f"  Size: {file_info['size_bytes']} bytes")
            print(f"  Lines: {file_info['lines']}")
            print(f"  Preview: {file_info['source_code'][:100]}...")
            print(f"{'-'*80}")
    
    def save_to_csv(self, output_path: str = "data/processed/extracted_files.csv") -> None:
        """
        Save extracted file information to CSV (without source code).
        
        Args:
            output_path: Path to save the CSV file
        """
        if not self.files_data:
            raise ValueError("No files extracted. Call extract_python_files() first.")
        
        import pandas as pd
        
        # Create DataFrame without source_code (too large for CSV)
        df_data = []
        for f in self.files_data:
            df_data.append({
                'filename': f['filename'],
                'filepath': f['filepath'],
                'full_path': f['full_path'],
                'size_bytes': f['size_bytes'],
                'lines': f['lines']
            })
        
        df = pd.DataFrame(df_data)
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved file list to {output_path}")


def create_test_repository(output_dir: str = "data/raw/test_repo") -> str:
    """
    Create a test repository with sample Python files.
    
    Args:
        output_dir: Directory to create the test repository
        
    Returns:
        Path to the created test repository
    """
    repo_path = Path(output_dir)
    repo_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample Python files
    sample_files = {
        'main.py': '''#!/usr/bin/env python3
"""Main module for the application."""

def main():
    """Main function."""
    print("Hello, World!")
    process_data()

def process_data():
    """Process some data."""
    data = [1, 2, 3, 4, 5]
    result = sum(data)
    return result

if __name__ == "__main__":
    main()
''',
        'utils.py': '''"""Utility functions."""

def helper_function(x):
    """A helper function."""
    return x * 2

def another_helper(y):
    """Another helper function."""
    return y + 10
''',
        'models/user.py': '''"""User model."""

class User:
    """User class."""
    
    def __init__(self, name, email):
        """Initialize user."""
        self.name = name
        self.email = email
    
    def get_info(self):
        """Get user info."""
        return f"{self.name} ({self.email})"
''',
        'models/product.py': '''"""Product model."""

class Product:
    """Product class."""
    
    def __init__(self, name, price):
        """Initialize product."""
        self.name = name
        self.price = price
    
    def get_price(self):
        """Get product price."""
        return self.price
''',
        'api/endpoints.py': '''"""API endpoints."""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users."""
    return jsonify([])

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products."""
    return jsonify([])
'''
    }
    
    # Write files
    for file_path, content in sample_files.items():
        full_path = repo_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')
    
    print(f"[OK] Created test repository at {repo_path}")
    print(f"   Total files: {len(sample_files)}")
    
    return str(repo_path)


def main():
    """Main function to test repository extraction."""
    print("=" * 80)
    print("Phase 3: Repository Processing - Testing")
    print("=" * 80)
    
    # Create test repository
    print("\n1. Creating test repository...")
    repo_path = create_test_repository()
    
    # Extract files
    print("\n2. Extracting Python files...")
    extractor = RepositoryExtractor(repo_path)
    files = extractor.extract_python_files()
    
    # Display statistics
    print("\n3. Repository statistics:")
    stats = extractor.get_statistics()
    for key, value in stats.items():
        if key != 'files':
            print(f"   {key}: {value}")
    
    # Display files
    print("\n4. Extracted files:")
    extractor.display_files(n=3)
    
    # Test getting specific file
    print("\n5. Testing get_file()...")
    file_info = extractor.get_file('models/user.py')
    if file_info:
        print(f"   Found: {file_info['filepath']}")
        print(f"   Lines: {file_info['lines']}")
        print(f"   Preview: {file_info['source_code'][:100]}...")
    
    # Save to CSV
    print("\n6. Saving file list to CSV...")
    extractor.save_to_csv()
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 3 Complete! Repository processing is ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()