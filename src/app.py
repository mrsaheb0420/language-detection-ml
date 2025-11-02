import flask
from flask import Flask, request, render_template_string, jsonify
import joblib
import os
import sys
from typing import Optional, List

# Make sure project root is on sys.path when running script directly
if __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Optional native fastText module import. Wrapped in try/except so the app still runs
# when fasttext is not installed. Use `FASTTEXT_AVAILABLE` to check presence.
try:
    import fasttext  # type: ignore
    FASTTEXT_AVAILABLE = True
except Exception:
    fasttext = None  # type: ignore
    FASTTEXT_AVAILABLE = False

APP = Flask(__name__)
MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'lang_detector.joblib')

# Optional fastText model paths (place lid.176.ftz or lid.176.bin in models/)
FASTTEXT_MODEL_PATHS: List[str] = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'lid.176.ftz'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'lid.176.bin'),
]
FASTTEXT_MODEL = None

# Map 2-letter codes to full language names used in sample dataset
LANG_FULL = {
    # European / common
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'sv': 'Swedish',
    'nl': 'Dutch',
    # Common Asian languages (expanded)
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ur': 'Urdu',
    'ar': 'Arabic',
    'fa': 'Persian',
    'tr': 'Turkish',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'lo': 'Lao',
    'km': 'Khmer',
    'my': 'Burmese',
    'mn': 'Mongolian',
    'az': 'Azerbaijani',
    'he': 'Hebrew',
    # Indian languages (major 22 + related codes)
    'sa': 'Sanskrit',
    'as': 'Assamese',
    'brx': 'Bodo',
    'doi': 'Dogri',
    'ks': 'Kashmiri',
    'kok': 'Konkani',
    'mai': 'Maithili',
    'mni': 'Manipuri',
    'sat': 'Santali',
    'sd': 'Sindhi',
    # Note: many Indian languages also have two-letter codes used elsewhere (e.g. 'bn','hi','pa','or','te','ta','kn','ml','mr','gu','ne','sa')
}

HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Language Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }
        textarea { width: 100%; font-size: 16px; }
        .controls { margin-top: 8px; display:flex; gap:12px; align-items:center }
        .result { background:#f7f7f7; padding:12px; border-radius:6px; margin-top:12px }
        label { font-size:14px }
        .meta { color:#666; font-size:13px }
    </style>
</head>
<body>
<h1>Language Detector</h1>
<form method=post>
    <textarea name=text rows=6 placeholder="Type or paste text here..."></textarea>
    <div class="controls">
        <label>Backend:
            <select name=backend>
                <option value="auto">Auto (sklearn → fastText → langdetect)</option>
                <option value="sklearn">sklearn pipeline</option>
                <option value="fasttext">fastText (if available)</option>
                <option value="langdetect">langdetect (fallback)</option>
            </select>
        </label>
        <label><input type=checkbox name=translate> Translate to English</label>
        <input type=submit value="Detect">
    </div>
</form>
{% if fasttext_available is not none %}
    <p class="meta">fastText model present: {{ 'yes' if fasttext_available else 'no' }} — put <code>lid.176.ftz</code> or <code>lid.176.bin</code> in <code>models/</code></p>
{% endif %}
{% if result %}
    <div class="result">
        <h2>Result</h2>
        <p><strong>Language code:</strong> {{ result.lang }}</p>
        <p><strong>Language name:</strong> {{ result.lang_full }}</p>
        <p><strong>Confidence:</strong> {{ '{:.2f}'.format(result.confidence) }}</p>
        <p><strong>Text:</strong> {{ result.text }}</p>
        {% if result.translation %}
            <h3>Translation (English)</h3>
            <p>{{ result.translation }}</p>
        {% endif %}
    </div>
{% endif %}
<p class="meta">API: POST JSON {"text":"...","translate":true,"backend":"fasttext"} to <code>/api/predict</code></p>
</body>
</html>
"""


def load_model(path: Optional[str] = None):
    global MODEL
    if MODEL is not None:
        return MODEL
    p = path or MODEL_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file not found: {p}. Train and save the model first.")
    MODEL = joblib.load(p)
    return MODEL

def load_fasttext_model():
    """Try to load fastText language ID model if available in models/ as lid.176.ftz or lid.176.bin.
    Returns the model or None if not available."""
    global FASTTEXT_MODEL
    if FASTTEXT_MODEL is not None:
        return FASTTEXT_MODEL
    # If the fasttext module isn't available (checked at import time), bail out early
    if not FASTTEXT_AVAILABLE:
        return None
    for p in FASTTEXT_MODEL_PATHS:
        if os.path.exists(p):
            try:
                FASTTEXT_MODEL = fasttext.load_model(p)
                return FASTTEXT_MODEL
            except Exception:
                continue
    return None


def predict(text: str, backend: str = 'auto'):
    """Return (code, full_name, confidence) for the given text.

    backend: 'auto' (sklearn -> fasttext -> langdetect), 'fasttext', 'langdetect', or 'sklearn'
    """
    backend = (backend or 'auto').lower()

    # 1) sklearn pipeline
    if backend in ('sklearn', 'auto'):
        try:
            model = load_model()
            preds = model.predict([text])
            code = preds[0]
            confidence = None
            try:
                probs = model.predict_proba([text])[0]
                classes = list(model.classes_)
                idx = classes.index(code) if code in classes else None
                if idx is not None:
                    confidence = float(probs[idx])
            except Exception:
                confidence = 1.0
            name = LANG_FULL.get(code, code)
            return code, name, confidence
        except FileNotFoundError:
            if backend == 'sklearn':
                return 'unknown', 'Unknown', 0.0
            # else fall through

    # 2) fastText
    if backend in ('fasttext', 'auto'):
        try:
            ft = load_fasttext_model()
            if ft is not None:
                labels, probs = ft.predict(text, k=1)
                if labels:
                    lbl = labels[0]
                    code = lbl.replace('__label__', '')
                    conf = float(probs[0]) if probs else 0.0
                    name = LANG_FULL.get(code, code)
                    return code, name, conf
            elif backend == 'fasttext':
                return 'unknown', 'Unknown', 0.0
        except Exception:
            if backend == 'fasttext':
                return 'unknown', 'Unknown', 0.0

    # 3) langdetect
    if backend in ('langdetect', 'auto'):
        try:
            from langdetect import detect_langs, DetectorFactory
            DetectorFactory.seed = 0
            langs = detect_langs(text)
            if not langs:
                return 'unknown', 'Unknown', 0.0
            best = langs[0]
            code = best.lang
            name = LANG_FULL.get(code, LANG_FULL.get(code.lower(), code))
            confidence = float(best.prob)
            return code, name, confidence
        except Exception:
            return 'unknown', 'Unknown', 0.0

    return 'unknown', 'Unknown', 0.0


def translate_text(text: str, target: str = 'en'):
    """Translate text to target language using googletrans if available.
    Returns translated text or None if translator not available."""
    try:
        from googletrans import Translator
    except Exception:
        return None
    try:
        translator = Translator()
        res = translator.translate(text, dest=target)
        return res.text
    except Exception:
        return None


@APP.route('/', methods=['GET', 'POST'])
def index():
    result = None
    fasttext_available = load_fasttext_model() is not None
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        backend = request.form.get('backend', 'auto')
        if text:
            try:
                code, name, conf = predict(text, backend=backend)
                translated = None
                # If the form included a translate checkbox, translate to English (if possible)
                if request.form.get('translate') == 'on':
                    translated = translate_text(text, target='en')
                result = {
                    'lang': code,
                    'lang_full': name,
                    'confidence': conf if conf is not None else 0.0,
                    'text': text,
                    'translation': translated
                }
            except FileNotFoundError as e:
                result = {'error': str(e)}
    return render_template_string(HTML, result=result, fasttext_available=fasttext_available)


@APP.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get('text') or request.form.get('text') or request.args.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # detect if translation requested: JSON field `translate` (bool), form field `translate`, or query param `translate` (true/1)
    translate_requested = False
    if isinstance(data.get('translate'), bool):
        translate_requested = data.get('translate')
    elif request.form.get('translate') in ('on', 'true', '1'):
        translate_requested = True
    elif str(request.args.get('translate', '')).lower() in ('true', '1', 'yes'):
        translate_requested = True

    backend = data.get('backend') or request.form.get('backend') or request.args.get('backend') or 'auto'
    try:
        code, name, conf = predict(text, backend=backend)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500

    resp = {
        'text': text,
        'lang': code,
        'lang_full': name,
        'confidence': conf if conf is not None else 0.0
    }

    if translate_requested:
        translated = translate_text(text, target='en')
        resp['translation'] = translated

    return jsonify(resp)


if __name__ == '__main__':
    # Run development server
    APP.run(host='127.0.0.1', port=5000, debug=True)
