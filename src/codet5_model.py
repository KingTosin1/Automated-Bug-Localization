"""
CodeT5 model wrapper for generating embeddings.

This module provides functions to load CodeT5 and generate embeddings for
both natural language (bug reports) and code (source files).
"""

import torch
from transformers import RobertaTokenizer, T5EncoderModel
from typing import List, Union
import numpy as np


class CodeT5Model:
    """
    A wrapper class for CodeT5 model.
    
    CodeT5 is a pre-trained model by Salesforce specifically designed for
    code understanding and generation. It's based on the T5 architecture and
    trained on a large corpus of code.
    
    Attributes:
        model_name (str): Name of the pre-trained model
        tokenizer: Tokenizer for the model
        model: The pre-trained CodeT5 model
        device: Device to run the model on (CPU/GPU)
    """
    
    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        """
        Initialize the CodeT5 model.
        
        Args:
            model_name: Name of the pre-trained model from Hugging Face
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self) -> None:
        """
        Load the CodeT5 model and tokenizer.
        
        This downloads the model from Hugging Face if it's not cached locally.
        The model will be moved to GPU if available, otherwise CPU.
        """
        print(f"[INFO] Loading CodeT5 model: {self.model_name}")
        print(f"[INFO] Using device: {self.device}")
        
        # Load tokenizer and model
        # CodeT5 uses Roberta tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        # Use T5EncoderModel to get encoder outputs only
        self.model = T5EncoderModel.from_pretrained(self.model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("[OK] CodeT5 model loaded successfully")
    
    def encode_text(self, text: Union[str, List[str]], max_length: int = 512) -> np.ndarray:
        """
        Encode text (natural language) into embeddings.
        
        Args:
            text: Single text string or list of texts to encode
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Numpy array of embeddings (shape: [batch_size, hidden_size])
            
        Note:
            - Uses the encoder's last hidden state, averaged over all tokens
            - Hidden size for CodeT5-base is 768
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model.encoder(**inputs)
        
        # Use mean pooling of last hidden state as the text representation
        # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings
    
    def encode_code(self, code: Union[str, List[str]], max_length: int = 512) -> np.ndarray:
        """
        Encode code into embeddings.
        
        This is similar to encode_text() but specifically for code.
        CodeT5 was trained on code, so it understands code structure.
        
        Args:
            code: Single code string or list of code snippets to encode
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Numpy array of embeddings (shape: [batch_size, hidden_size])
            
        Note:
            - Uses the encoder's last hidden state, averaged over all tokens
            - Hidden size for CodeT5-base is 768
        """
        # For CodeT5, encoding code is the same as encoding text
        # The model was trained on both and handles them similarly
        return self.encode_text(code, max_length=max_length)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (between -1 and 1)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Reshape if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        return float(similarity)


def main():
    """Main function to test CodeT5 model."""
    print("=" * 80)
    print("Phase 9: CodeT5 Model - Testing")
    print("=" * 80)
    
    # Initialize model
    print("\n1. Initializing CodeT5 model...")
    codet5 = CodeT5Model()
    
    # Load model
    print("\n2. Loading model (this may take a minute on first run)...")
    codet5.load_model()
    
    # Test encoding text (bug report)
    print("\n3. Testing text encoding (bug report)...")
    bug_report = "Connection timeout when making HTTP requests to slow servers"
    text_embedding = codet5.encode_text(bug_report)
    print(f"   Bug report: {bug_report[:50]}...")
    print(f"   Embedding shape: {text_embedding.shape}")
    print(f"   Embedding dimension: {text_embedding.shape[1]}")
    print(f"   First 5 values: {text_embedding[0][:5]}")
    
    # Test encoding code
    print("\n4. Testing code encoding (source file)...")
    code_snippet = """
def connect(url, timeout=30):
    \"\"\"Connect to a URL with timeout.\"\"\"
    import requests
    response = requests.get(url, timeout=timeout)
    return response
"""
    code_embedding = codet5.encode_code(code_snippet)
    print(f"   Code snippet: {code_snippet[:50].strip()}...")
    print(f"   Embedding shape: {code_embedding.shape}")
    print(f"   Embedding dimension: {code_embedding.shape[1]}")
    print(f"   First 5 values: {code_embedding[0][:5]}")
    
    # Test batch encoding
    print("\n5. Testing batch encoding...")
    texts = [
        "Database connection error",
        "API endpoint not found",
        "Authentication failed"
    ]
    batch_embeddings = codet5.encode_text(texts)
    print(f"   Encoded {len(texts)} texts")
    print(f"   Batch embedding shape: {batch_embeddings.shape}")
    
    # Test similarity computation
    print("\n6. Testing similarity computation...")
    similarity = codet5.compute_similarity(text_embedding, code_embedding)
    print(f"   Similarity between bug report and code: {similarity:.4f}")
    
    # Test with different code
    print("\n7. Testing with unrelated code...")
    unrelated_code = """
def calculate_sum(a, b):
    \"\"\"Calculate sum of two numbers.\"\"\"
    return a + b
"""
    unrelated_embedding = codet5.encode_code(unrelated_code)
    unrelated_similarity = codet5.compute_similarity(text_embedding, unrelated_embedding)
    print(f"   Similarity with unrelated code: {unrelated_similarity:.4f}")
    print(f"   Difference: {abs(similarity - unrelated_similarity):.4f}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Phase 9 Complete! CodeT5 model is ready.")
    print("=" * 80)
    print("\nKey Points:")
    print("  - CodeT5 generates 768-dimensional embeddings")
    print("  - Uses mean pooling of encoder outputs")
    print("  - Can encode both natural language and code")
    print("  - Similarity scores help match bug reports to code files")


if __name__ == "__main__":
    main()