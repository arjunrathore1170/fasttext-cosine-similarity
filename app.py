from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.model import load_or_train_model, read_corpus
from src.preprocessing import tokenize
from src.similarity import sentence_vector, cosine_score

st.set_page_config(
    page_title="FastText Semantic Similarity Lab",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background: #071321; color: #e8f1f8; }
[data-testid="stSidebar"] { background: #0b1b2b; }
h1, h2, h3 { color: #e8f1f8 !important; }
p, label, li { color: #c7d6e4; }
div[data-testid="stMetric"] {
    background: #10283a; border: 1px solid #1c5361;
    padding: 16px; border-radius: 14px;
}
.stButton > button {
    background: #16b8a6; color: #061321; border: 0;
    border-radius: 9px; font-weight: 700;
}
.stButton > button:hover { background: #56e0cf; color: #061321; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_or_train_model()

st.sidebar.title("FastText Lab")
page = st.sidebar.radio(
    "Navigate",
    ["Similarity Analyzer", "Word Explorer", "Word2Vec vs FastText", "How It Works"]
)
st.sidebar.caption("Local NLP mini project · No API key required")

try:
    model = get_model()
except Exception as exc:
    st.error(f"Could not load/train the model: {exc}")
    st.info("Check requirements.txt and corpus.txt, then run: python -m pip install -r requirements.txt")
    st.stop()

st.title("FastText Semantic Similarity Lab")
st.caption("Explore word embeddings, character n-grams and cosine similarity")

examples = {
    "Car vs vehicle": ("The car is fast", "The vehicle moves quickly"),
    "Cat vs feline": ("The cat sits on the mat", "A feline rests on a rug"),
    "Machine learning": ("I love machine learning", "Artificial intelligence is interesting"),
    "Different topics": ("The weather is sunny today", "Databases store structured information"),
}

if page == "Similarity Analyzer":
    st.subheader("Compare two pieces of text")
    example = st.selectbox("Load a demo example", ["Choose an example"] + list(examples))
    default_a, default_b = examples.get(example, ("The car is fast", "The vehicle moves quickly"))
    left, right = st.columns(2)
    with left:
        text_a = st.text_area("Text A", default_a, height=140, key="text_a")
    with right:
        text_b = st.text_area("Text B", default_b, height=140, key="text_b")
    if st.button("Calculate Similarity", type="primary"):
        if not text_a.strip() or not text_b.strip():
            st.warning("Please enter both texts.")
        else:
            try:
                vec_a = sentence_vector(text_a, model, tokenize)
                vec_b = sentence_vector(text_b, model, tokenize)
                score = cosine_score(text_a, text_b, model, tokenize)
                c1, c2, c3 = st.columns(3)
                c1.metric("Cosine score", f"{score:.4f}")
                c2.metric("Vector dimensions", str(model.vector_size))
                c3.metric("Tokens compared", f"{len(tokenize(text_a))} + {len(tokenize(text_b))}")
                st.progress(float(np.clip((score + 1) / 2, 0, 1)))
                st.caption("The progress bar maps the mathematical range −1 to +1 onto 0% to 100%; it is not a probability.")
                if score >= 0.70:
                    st.success("High vector similarity (for this trained model)")
                elif score >= 0.30:
                    st.info("Moderate vector similarity (for this trained model)")
                else:
                    st.warning("Low vector similarity (for this trained model)")
                with st.expander("Inspect sentence vectors"):
                    st.write("Sentence A vector (first 12 values)")
                    st.code(np.array2string(vec_a[:12], precision=4))
                    st.write("Sentence B vector (first 12 values)")
                    st.code(np.array2string(vec_b[:12], precision=4))
                st.caption("This model is trained on a small demonstration corpus. Scores are illustrative and are not guaranteed to match human judgement.")
            except ValueError as exc:
                st.error(str(exc))

elif page == "Word Explorer":
    st.subheader("Explore FastText word embeddings")
    word = st.text_input("Enter a word", "connected")
    if st.button("Explore Word"):
        tokens = tokenize(word)
        if len(tokens) != 1:
            st.warning("Enter one word using letters or numbers.")
        else:
            token = tokens[0]
            vector = model.wv[token]
            in_vocab = token in model.wv.key_to_index
            st.metric("Embedding dimension", model.vector_size)
            st.write("Vocabulary word:", "Yes" if in_vocab else "No — vector generated from character n-grams")
            st.write("Vector values (first 20)")
            st.code(np.array2string(vector[:20], precision=4))
            st.write("Nearest known vocabulary words")
            try:
                neighbors = model.wv.similar_by_vector(vector, topn=6)
                st.table([{"Word": w, "Cosine similarity": round(float(s), 4)} for w, s in neighbors if w != token])
            except Exception:
                st.info("No nearest words could be returned.")
            st.caption("FastText can construct vectors for out-of-vocabulary words from character n-grams. Nearest-word quality depends on the small training corpus.")

elif page == "Word2Vec vs FastText":
    st.subheader("Word2Vec vs FastText")
    st.markdown("""
| Feature | Word2Vec | FastText |
|---|---|---|
| Representation | Whole-word vectors | Word plus character n-grams |
| Rare words | Often weaker with few examples | Can share subword information |
| Unseen words | Standard vocabulary has no vector | Can construct vectors from n-grams |
| Morphology | Not explicitly modelled | Captured through shared subwords |
| Model size | Often smaller | Usually larger due to n-gram table |
""")
    st.info("This page explains the conceptual comparison. The current application trains FastText only; it does not claim a measured benchmark against Word2Vec.")

elif page == "How It Works":
    st.subheader("Pipeline")
    st.markdown("""
**Input text → Tokenization → FastText embeddings → Average word vectors → Cosine similarity → Interpretation**

1. **Word embeddings:** represent words as dense numerical vectors.
2. **Character n-grams:** FastText represents a word using character sequences, typically lengths 3 to 6, plus the whole word.
3. **Sentence vector:** this demo averages the vectors for the sentence's tokens.
4. **Cosine similarity:** compares vector direction, not magnitude.
5. **Interpretation:** a score is a mathematical similarity measure, not a guaranteed measure of human understanding.
""")
    st.latex(r"\cos(\theta)=\frac{A\cdot B}{\|A\|\|B\|}")
    st.markdown("**Worked example**")
    st.code("A = [1, 2], B = [2, 4]\nA · B = 1×2 + 2×4 = 10\n||A|| = √5, ||B|| = √20\ncos(θ) = 10 / (√5 × √20) = 1")
    st.markdown("""
**Limitations**
- Averaging loses word order.
- Static vectors do not fully represent context-dependent meanings.
- Model quality depends on training corpus size and domain.
- This bundled corpus is intentionally small for a fast classroom demo.
""")

st.divider()
st.caption("NLP Mini Project · FastText + Cosine Similarity · Local demonstration")
