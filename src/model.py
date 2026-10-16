from pathlib import Path
from gensim.models import FastText
from .preprocessing import tokenize

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpus.txt"
MODEL_PATH = ROOT / "models" / "fasttext_demo.model"

def read_corpus(path=CORPUS_PATH):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    sentences = [tokenize(line) for line in lines if line.strip()]
    return [s for s in sentences if s]

def load_or_train_model(force_retrain=False):
    """Load the saved local model or train it once from corpus.txt."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force_retrain:
        try:
            return FastText.load(str(MODEL_PATH))
        except Exception:
            MODEL_PATH.unlink(missing_ok=True)

    sentences = read_corpus()
    if not sentences:
        raise ValueError("corpus.txt has no usable training sentences.")

    model = FastText(
        vector_size=100,
        window=5,
        min_count=1,
        min_n=3,
        max_n=6,
        sg=1,
        workers=1,
        seed=42,
        epochs=25
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(
        corpus_iterable=sentences,
        total_examples=len(sentences),
        epochs=model.epochs
    )
    model.save(str(MODEL_PATH))
    return model
