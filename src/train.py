import argparse
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Allow running the script directly (python src\train.py) or as a module (python -m src.train).
# When executed directly, the package `src` may not be importable; ensure project root is on sys.path.
if __package__ is None:
    # Insert project root (parent of src) to sys.path so `import src.utils` works.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.utils import load_dataset

def train_model(data_path: str, out_path: str, test_size: float = 0.2, random_state: int = 42):
    df = load_dataset(data_path)
    X = df['text'].values
    y = df['lang'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
        ('clf', LogisticRegression(max_iter=200))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Ensure models directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Saved model to {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Train language detection model")
    parser.add_argument('--data', required=True, help="Path to CSV dataset (text,lang)")
    parser.add_argument('--out', required=True, help="Output joblib model path")
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()
    train_model(args.data, args.out, test_size=args.test_size)

if __name__ == '__main__':
    main()