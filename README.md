<<<<<<< HEAD
# Language Detection (ML)

# 1. INTRODUCTION

This project implements a robust multi-language detection system using machine learning techniques, specifically designed to identify text across a wide range of languages including Asian and Indian languages.

## Project Overview

The system employs a sophisticated machine learning pipeline built with scikit-learn, incorporating:

- TF-IDF vectorization with word bigrams for feature extraction
- LogisticRegression classifier for language prediction
- Support for multiple detection backends (sklearn, langdetect, fastText)
- Web-based user interface for easy access and testing

### Key Features

1. **Multi-Language Support**

   - Covers major world languages
   - Special focus on 22 Indian languages
   - Robust detection of Asian languages
   - Extensible language model support

2. **Technical Implementation**

   - Flask-based web interface
   - RESTful API endpoints
   - Real-time language detection
   - High accuracy text classification

3. **System Architecture**

   - Modular design with separate components:
     - Data processing utilities
     - Model training pipeline
     - Prediction interface
     - Web service layer

4. **User Interface**
   - Clean, intuitive web interface
   - Support for text input of any length
   - Instant language detection results
   - Browser-based accessibility

### Technology Stack

- **Backend**: Python 3.8+
- **Web Framework**: Flask
- **ML Libraries**: scikit-learn, langdetect
- **Data Processing**: pandas, joblib
- **Translation Support**: googletrans 3.1.0

### Performance Highlights

- Fast response times for real-time detection
- High accuracy across diverse languages
- Scalable architecture for growing language support
- Efficient resource utilization

---

## Implementation Details

This project is a language-detection example using multiple detection backends:

- sklearn TF-IDF + LogisticRegression pipeline (trainable)
- optional fastText pretrained language identifier (recommended for wide coverage of Indic languages)
- langdetect as a final fallback (quick, limited accuracy)
- optional translation via `googletrans` (requires internet)

What you get

- Training script: `src/train.py` — trains a TF-IDF + classifier pipeline and saves it to `models/`.
- Prediction CLI: `src/predict.py` — loads the saved model and predicts language for given text.
- Web UI & API: `src/app.py` — small Flask app with frontend and JSON API. Supports backend selection (auto/fasttext/sklearn/langdetect) and optional translation to English.
- Sample dataset: `data/sample_data.csv` (small phrases across many languages).

Quick start (Windows cmd.exe)

1. Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```cmd
pip install -r requirements.txt
```

3. (Optional but recommended) Download fastText language identification model for better coverage:

- Visit https://fasttext.cc/docs/en/language-identification.html and download `lid.176.ftz` or `lid.176.bin`.
- Place the downloaded file into the `models/` folder.

4. Train the sklearn model (optional):

```cmd
python src\train.py --data data\sample_data.csv --out models\lang_detector.joblib
```

5. Start the web UI:

```cmd
python src\app.py
```

Open http://127.0.0.1:5000 in your browser. Choose the backend (Auto/fastText/sklearn/langdetect). For detecting Indian languages, choose `fastText` if you placed the model in `models/`.

API usage examples

- Basic detection (JSON):

```cmd
curl -X POST -H "Content-Type: application/json" -d "{\"text\":\"こんにちは\"}" http://127.0.0.1:5000/api/predict
```

- Detection + translate:

```cmd
curl -X POST -H "Content-Type: application/json" -d "{\"text\":\"こんにちは\", \"translate\": true}\"" http://127.0.0.1:5000/api/predict
```

Notes and limitations

- The provided `data/sample_data.csv` contains minimal examples for many languages — it's not a balanced dataset. For production, use sizeable curated corpora per language.
- fastText (lid.176) is recommended for offline, high-coverage detection across Indic languages. It requires downloading the pretrained model (about 100MB).
- `googletrans` is an unofficial translator wrapper; it may be unstable. For production translations, use an official cloud API.

Next steps

- Replace the tiny sample dataset with larger labeled corpora per language.
- Optionally integrate fastText model for highest offline accuracy.
- Add a backend selection preference and caching for performance.
=======
# language-detection-ml
A machine learning–based multilingual text detection system using Python, Flask, and scikit-learn. Supports 100+ languages with special focus on 22 Indian languages, real-time web UI, REST API, and optional translation using Google Translate and fastText integration.
>>>>>>> b2d53e116b03622214c492cb0b8f9a0f3c111397
