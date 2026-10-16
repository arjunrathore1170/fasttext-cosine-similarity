import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def sentence_vector(text: str, model, tokenizer) -> np.ndarray:
    """Average the FastText word vectors for tokens in a sentence."""
    tokens = tokenizer(text)
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    vectors = [model.wv[token] for token in tokens]
    return np.mean(vectors, axis=0)

def cosine_score(text_a: str, text_b: str, model, tokenizer) -> float:
    """Return cosine similarity between two averaged sentence vectors."""
    a = sentence_vector(text_a, model, tokenizer)
    b = sentence_vector(text_b, model, tokenizer)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compare an empty or zero-length sentence vector.")
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])
