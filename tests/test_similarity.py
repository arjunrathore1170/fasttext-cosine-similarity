import numpy as np
import pytest

from src.preprocessing import tokenize
from src.similarity import sentence_vector, cosine_score
from src.model import load_or_train_model

@pytest.fixture(scope="module")
def model():
    return load_or_train_model()

def test_tokenizer_lowercases():
    assert tokenize("Hello WORLD!") == ["hello", "world"]

def test_sentence_vector_dimension(model):
    vector = sentence_vector("the car is fast", model, tokenize)
    assert vector.shape == (model.vector_size,)

def test_identical_sentence_score_is_one(model):
    score = cosine_score("the car is fast", "the car is fast", model, tokenize)
    assert score == pytest.approx(1.0, abs=1e-5)

def test_cosine_score_bounds(model):
    score = cosine_score("the cat sits on the mat", "the weather is sunny", model, tokenize)
    assert -1.00001 <= score <= 1.00001

def test_empty_sentence_rejected(model):
    with pytest.raises(ValueError):
        cosine_score("", "the car is fast", model, tokenize)

def test_fasttext_can_vectorize_unknown_word(model):
    vector = model.wv["qzxunseenword"]
    assert vector.shape == (model.vector_size,)
    assert np.isfinite(vector).all()
