# FastText Semantic Similarity Lab

A local NLP mini project demonstrating FastText word embeddings, character n-grams, average sentence vectors, and cosine similarity.

## Features
- Compare two sentences with real calculated cosine similarity.
- Explore word vectors and nearest vocabulary words.
- Demonstrate FastText's ability to construct vectors for unseen words.
- Learn the Word2Vec vs FastText differences.
- Dark navy and teal Streamlit UI.
- Local training; no API key required.

## Requirements
- Python 3.11 or newer
- Internet connection for initial package installation only

## Run on Windows (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If `python` is not recognized, install Python from https://www.python.org/downloads/ and enable **Add Python to PATH**, then reopen the terminal.

Open the local address printed in the terminal, usually http://localhost:8501.

## Tests
```cmd
python -m pip install pytest
python -m pytest -q
```

## Pipeline
Input text -> tokenization -> FastText word vectors -> average sentence vectors -> cosine similarity -> interpretation.

## Model settings
The model trains on `corpus.txt` using 100-dimensional skip-gram FastText embeddings and character n-grams of lengths 3–6. The trained model is saved in `models/fasttext_demo.model` and reused on subsequent launches.

## Important limitations
The bundled corpus is small and designed for an easy classroom demo. Similarity scores may be poor or surprising; they are not probabilities or guaranteed percentages of meaning. Sentence averaging loses word order and contextual meaning. Use a much larger domain-relevant corpus or a suitable pretrained model for more realistic results.

## PPT alignment
- Word embeddings and distributional hypothesis
- FastText character n-grams
- Skip-gram training
- FastText vs Word2Vec
- Cosine similarity formula and worked example
- Sentence-level vector averaging
- Applications and limitations
