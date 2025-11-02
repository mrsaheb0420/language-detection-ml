import os
import sys
import tempfile
import joblib
import runpy

def test_train_and_predict_creates_model_and_predicts():
    # Ensure running from repo root so paths match
    repo_root = os.path.abspath(os.path.dirname(__file__))  # tests folder
    repo_root = os.path.dirname(repo_root)  # project root
    data_path = os.path.join(repo_root, 'data', 'sample_data.csv')

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, 'model.joblib')

        # Prepare sys.argv for train script
        old_argv = sys.argv[:]
        sys.argv = ['train.py', '--data', data_path, '--out', model_path, '--test-size', '0.2']
        try:
            # run the train script (it will execute train_model and save the model)
            runpy.run_path(os.path.join(repo_root, 'src', 'train.py'), run_name="__main__")
        finally:
            sys.argv = old_argv

        assert os.path.exists(model_path), "Model file was not created."

        # Load model and make a couple of predictions
        model = joblib.load(model_path)

        sample_texts = [
            "This is a short English sentence.",
            "Bonjour, comment allez-vous?",
            "Hola amigo, ¿cómo estás?"
        ]

        preds = model.predict(sample_texts)
        # All predictions should be among known labels
        known_labels = {'en','fr','es','de','it','pt','nl','sv'}
        for p in preds:
            assert p in known_labels