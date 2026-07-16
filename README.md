# Automated Bug Localization with Pre-Trained Code LLMs

A final-year project that uses CodeBERT and CodeT5 to automatically localize bugs in Python repositories based on natural language bug reports.

## Project Overview

This system takes a bug report written in natural language and identifies which Python source files in a repository are most likely to contain the bug. It uses pre-trained language models (CodeBERT and CodeT5) to generate embeddings for both bug reports and source code, then ranks files using cosine similarity.

## Features

- Accepts natural language bug reports
- Processes Python repositories
- Generates embeddings using CodeBERT and CodeT5
- Ranks files by relevance using cosine similarity
- Displays Top-K most suspicious files
- Compares performance between CodeBERT and CodeT5
- Simple web interface using Streamlit

## Technology Stack

- **Language**: Python 3.8+
- **Models**: microsoft/codebert-base, Salesforce/codet5-base
- **Libraries**: torch, transformers, pandas, numpy, scikit-learn, matplotlib, streamlit, tqdm

## Project Structure

```
automated-bug-localization/
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── dataset.py           # Dataset preparation
│   ├── extractor.py         # Repository file extraction
│   ├── codebert_model.py    # CodeBERT embedding generation
│   ├── codet5_model.py      # CodeT5 embedding generation
│   ├── similarity.py        # Cosine similarity computation
│   ├── localization.py      # Bug localization pipeline
│   ├── evaluation.py        # Evaluation metrics
│   └── utils.py             # Utility functions
├── data/
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Processed embeddings and results
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks for exploration
├── venv/                    # Python virtual environment
├── requirements.txt         # Python dependencies
├── main.py                  # Main entry point
└── README.md               # This file
```

## Installation

1. Clone or download this repository
2. Create a virtual environment (already done):
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Phase 1: Project Initialization
```bash
python main.py
```

### Phase 2-10: Run individual modules
Each phase will have its own script that can be run independently.

### Web Interface (Phase 10)
```bash
streamlit run app.py
```

## Development Phases

This project is built in 10 phases:

1. **Project Initialization** - Setup folder structure and dependencies
2. **Dataset Preparation** - Create and load bug report dataset
3. **Repository Processing** - Extract Python files from repositories
4. **CodeBERT** - Load and test CodeBERT model
5. **Embedding Generation** - Generate embeddings for bug reports and code
6. **Similarity Engine** - Implement cosine similarity ranking
7. **Bug Localization** - Combine all components into a pipeline
8. **Evaluation** - Implement Top-1, Top-5, and MRR metrics
9. **CodeT5** - Repeat pipeline with CodeT5 and compare results
10. **Web Interface** - Build Streamlit demo application

## Model Information

### CodeBERT
- **Model**: microsoft/codebert-base
- **Type**: BERT-based model pre-trained on code
- **Purpose**: Generates contextual embeddings for both natural language and code

### CodeT5
- **Model**: Salesforce/codet5-base
- **Type**: T5-based model pre-trained on code
- **Purpose**: Text-to-text model that can generate embeddings for code understanding

## Evaluation Metrics

- **Top-1 Accuracy**: Percentage of bugs where the correct file is ranked first
- **Top-5 Accuracy**: Percentage of bugs where the correct file is in the top 5
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of correct files

## Academic Context

This project demonstrates:
- Application of pre-trained language models to software engineering tasks
- Information retrieval techniques for bug localization
- Comparison of different model architectures
- Practical implementation of ML-based software engineering tools

## Future Extensions

Possible improvements for future work:
- Fine-tune models on bug-specific data
- Use more sophisticated ranking algorithms
- Incorporate code structure information
- Add support for multiple programming languages
- Implement hybrid approaches combining multiple models

## License

This is an academic project for educational purposes.

## Author

Final-year Computer Science Project