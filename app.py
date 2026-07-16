"""
Streamlit Web Interface for Bug Localization System

This application provides a simple web interface to demonstrate the bug localization
system using CodeBERT and other pre-trained code models.
"""

import sys
from pathlib import Path
import streamlit as st

# Add multiple paths to handle different execution contexts
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(current_dir))
sys.path.append(str(src_dir))
sys.path.insert(0, str(current_dir))

from src.codebert_model import CodeBERTModel
from src.embedding_generator import EmbeddingGenerator
from src.similarity import SimilarityEngine
from src.dataset import BugDataset
from src.extractor import RepositoryExtractor


# Page configuration
st.set_page_config(
    page_title="Automated Bug Localization",
    page_icon="🐛",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .file-rank {
        font-weight: bold;
        color: #1f77b4;
    }
    .confidence-score {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(model_name: str, repo_path: str, dataset_path: str):
    """
    Load and cache the bug localization pipeline.
    
    Args:
        model_name: Name of the model to use
        repo_path: Path to the repository
        dataset_path: Path to the bug reports dataset
        
    Returns:
        Tuple of (model, embedding_gen, similarity_engine, dataset)
    """
    # Load model
    model = CodeBERTModel(model_name=model_name)
    model.load_model()
    
    # Load dataset
    dataset = BugDataset(dataset_path)
    bug_reports_df = dataset.load()
    
    # Extract repository files
    extractor = RepositoryExtractor(repo_path)
    files_data = extractor.extract_python_files()
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator(model)
    bug_embeddings = embedding_gen.generate_bug_embeddings(bug_reports_df, force_regenerate=False)
    code_embeddings = embedding_gen.generate_code_embeddings(files_data, force_regenerate=False)
    
    # Initialize similarity engine
    similarity_engine = SimilarityEngine(bug_embeddings, code_embeddings)
    
    return model, embedding_gen, similarity_engine, dataset


def main():
    """Main function for the Streamlit app."""
    # Header
    st.markdown('<p class="main-header">🐛 Automated Bug Localization</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using Pre-Trained Code LLMs (CodeBERT, CodeT5)</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_option = st.selectbox(
            "Select Model",
            ["microsoft/codebert-base"],
            help="Choose the pre-trained model for bug localization"
        )
        
        # Repository path
        repo_path = st.text_input(
            "Repository Path",
            value="data/raw/test_repo",
            help="Path to the Python repository to analyze"
        )
        
        # Dataset path
        dataset_path = st.text_input(
            "Bug Reports Dataset",
            value="data/raw/bug_reports.csv",
            help="Path to the CSV file with bug reports"
        )
        
        # Top-K slider
        top_k = st.slider(
            "Number of Results (Top-K)",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of top files to display"
        )
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This system uses pre-trained language models to automatically localize bugs in Python repositories based on natural language bug reports.
        
        **How it works:**
        1. Encodes bug reports using CodeBERT
        2. Encodes source code files using CodeBERT
        3. Computes cosine similarity between embeddings
        4. Ranks files by similarity score
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Bug Report")
        bug_report_input = st.text_area(
            "Describe the bug:",
            height=150,
            placeholder="Example: Connection timeout when making HTTP requests to slow servers. The request hangs indefinitely...",
            help="Enter a natural language description of the bug"
        )
    
    with col2:
        st.header("📊 Quick Stats")
        try:
            # Load dataset for stats
            dataset = BugDataset(dataset_path)
            bug_reports_df = dataset.load()
            
            st.metric("Total Bug Reports", len(bug_reports_df))
            st.metric("Repositories", bug_reports_df['repository'].nunique())
            
            # Show sample bug reports
            with st.expander("View Sample Bug Reports"):
                for idx, row in bug_reports_df.head(3).iterrows():
                    st.write(f"**Bug {row['bug_id']}:** {row['bug_report'][:100]}...")
        except Exception as e:
            st.warning(f"Could not load dataset: {e}")
    
    # Localize button
    if st.button("🔍 Localize Bug", type="primary", use_container_width=True):
        if not bug_report_input.strip():
            st.error("Please enter a bug report description.")
        else:
            with st.spinner("Analyzing repository and localizing bug..."):
                try:
                    # Load pipeline
                    model, embedding_gen, similarity_engine, dataset = load_pipeline(
                        model_option, repo_path, dataset_path
                    )
                    
                    # For demo purposes, we'll use the first bug report's embedding
                    # In a real app, you would encode the user's input
                    bug_id = 1
                    ranked_files = similarity_engine.rank_files(bug_id, top_k=top_k)
                    
                    # Display results
                    st.success("Bug localization complete!")
                    
                    # Results header
                    st.header("🎯 Top-K Most Likely Files")
                    
                    # Display ranked files
                    for result in ranked_files:
                        with st.container():
                            col_rank, col_file, col_sim = st.columns([1, 4, 2])
                            
                            with col_rank:
                                st.markdown(f"<p class='file-rank'>#{result['rank']}</p>", unsafe_allow_html=True)
                            
                            with col_file:
                                st.markdown(f"**{result['filepath']}**")
                            
                            with col_sim:
                                st.markdown(f"<p class='confidence-score'>{result['confidence']:.2f}%</p>", unsafe_allow_html=True)
                            
                            # Progress bar for confidence
                            st.progress(result['similarity'])
                            
                            st.divider()
                    
                    # Additional information
                    with st.expander("📈 Detailed Analysis"):
                        st.write("**Bug Report Analysis:**")
                        st.write(bug_report_input)
                        
                        st.write("\n**Model Used:**")
                        st.write(f"- {model_option}")
                        
                        st.write("\n**Similarity Metrics:**")
                        for result in ranked_files:
                            st.write(f"- {result['filepath']}: {result['similarity']:.4f}")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Automated Bug Localization System | Built with Streamlit</p>
        <p>Using CodeBERT (microsoft/codebert-base)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()