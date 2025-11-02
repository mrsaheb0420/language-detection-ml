# AI Agent Instructions for Language Detection Project

## Project Overview

This is a machine learning project for detecting text language using scikit-learn. Key components:

- TF-IDF vectorization + LogisticRegression classifier pipeline
- Training script with evaluation metrics
- Command-line prediction interface
- Test suite validating core functionality

## Architecture & Data Flow

1. Data Processing (`src/utils.py`)

   - Handles CSV dataset loading with required 'text' and 'lang' columns
   - Dataset validation and cleaning (NaN removal)

2. Model Pipeline (`src/train.py`)

   - Features: TF-IDF with word bigrams (ngram_range=(1,2))
   - Classifier: LogisticRegression with increased max_iter
   - Evaluation: Prints accuracy and classification report
   - Saves: Complete sklearn Pipeline using joblib

3. Inference (`src/predict.py`)
   - Loads trained Pipeline model
   - Supports both CLI text input and stdin for batch processing
   - Returns language codes matching training data

## Key Development Workflows

### Setup (Windows)

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Training

```cmd
python src\train.py --data data\sample_data.csv --out models\lang_detector.joblib
```

- Monitor the classification report for per-language performance
- Model artifacts are saved to `models/` directory

### Testing

```cmd
python -m pytest -q
```

- Tests use temporary directories for model artifacts
- Validates end-to-end training and prediction
- Checks prediction quality on known samples

## Project Conventions

### Directory Structure

- `src/`: Core Python modules
- `data/`: Training/test datasets (CSV format)
- `models/`: Saved model artifacts (gitignored)
- `tests/`: pytest test suite

### Data Format

- CSVs must have 'text' and 'lang' columns
- Language codes are 2-letter (en, fr, es, etc.)
- Text should be cleaned/normalized before training

### Model Pipeline

- Uses scikit-learn Pipeline for reproducibility
- TF-IDF params tuned for short text classification
- LogisticRegression preferred for interpretable confidences

## Common Tasks & Patterns

### Adding New Languages

1. Update `data/sample_data.csv` with examples
2. Ensure balanced class distribution
3. Retrain model with updated dataset
4. Update tests with new language codes

### Modifying the Model

- Pipeline components in `src/train.py`
- Consider character n-grams for short texts
- Tune max_features in TfidfVectorizer for memory/speed

### Error Handling

- Dataset validation in utils.py
- Proper CLI argument handling
- Informative error messages for missing files

## Integration Points

- CSV data loading (pandas)
- Model serialization (joblib)
- scikit-learn Pipeline API
- Command-line interfaces (argparse)

## Notes

- The sample dataset is minimal - replace with larger corpus for production
- Model selection prioritizes simplicity and interpretability
- Designed for easy experimentation and extension
