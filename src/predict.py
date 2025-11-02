import argparse
import joblib
import os
import sys
from typing import List

# Support running as script or module. Ensure project root is on sys.path so `from src.utils` would work
# if needed in the future.
if __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def predict_texts(model_path: str, texts: List[str]):
    """
    Load pipeline from model_path and predict labels for texts.
    Returns a list of predicted language codes.
    """
    pipeline = joblib.load(model_path)
    preds = pipeline.predict(texts)
    # If classifier supports predict_proba, we could also return confidences.
    return preds.tolist()

def main():
    parser = argparse.ArgumentParser(description="Predict language for given text")
    parser.add_argument('--model', required=True, help="Path to saved joblib model")
    parser.add_argument('--text', help="Text to classify (enclose in quotes). If omitted, reads from stdin.")
    args = parser.parse_args()

    if args.text:
        texts = [args.text]
    else:
        import sys
        print("Reading lines from stdin. Press Ctrl+Z (Windows) or Ctrl+D (Unix) to finish.")
        texts = [line.strip() for line in sys.stdin if line.strip()]

    preds = predict_texts(args.model, texts)
    for t, p in zip(texts, preds):
        print(f"{p}\t{t}")

if __name__ == '__main__':
    main()